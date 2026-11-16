"""The per-frame signal table every model reads: one row per video frame, one column per signal.

    python model/features.py              # builds analysis_out/features.npz (about a minute)
    python model/features.py --rebuild    # ignore the saved table

    import features; table = features.load()     # a pandas DataFrame, 199319 rows

Columns (distances in metres, speeds in m/s, NaN where a value does not exist):

  frame             video frame, the submission row
  half              1 or 2 inside the graded halves, 0 outside
  ball_seen         the tracking has a ball position
  ball_x, ball_y    ball position in metres, origin at a corner
  ball_outside      ball seen and more than 0.5 m outside the pitch lines
  ball_speed        ball speed from consecutive positions, median over 5 frames
  d_home, d_away    distance from the ball to the nearest tracked player of each team
  nearest_team      0 home, 1 away, -1 no ball
  n_home, n_away    tracked objects per team
  speed_mean        mean speed of all tracked players
  speed_mean_2s     the same, averaged over a centred 2-second window
  gap_len           length in frames of the ball gap this frame is in (0 if the ball is seen)
  gap_pos           frames since that gap started (0 if the ball is seen)
  team_before_gap   nearest team in the last second before the gap (-1 if unknown or no gap)
  team_after_gap    nearest team in the first second after the gap
  tag_home          Smart Tagging phases active now that say "home has the ball"
  tag_away          the same for away. Attacking phases count for their team, "Defending ..." and
                    "Defensive transition" for the other team, as the code definitions in
                    OVERVIEW.md describe them
  setpiece          a Smart Tagging set piece is active

Everything is derived from the gated data, so the table lives in analysis_out/ (gitignored).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
import load_smart_tagging as lst  # noqa: E402
from analyze_data import load_atd, team_code  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
LENGTH, WIDTH = games.GAMES[GAME]["pitch"]
N_FRAMES = games.source_frames("tactical", GAME)
TABLE_PATH = os.path.join(ROOT, "analysis_out", "features.npz")
OUTSIDE_MARGIN_M = 0.5
BALL_SPEED_MEDIAN = 5
SPEED_WINDOW = 2 * FPS
GAP_CONTEXT = FPS
ATTACKING_CODES = ("build up", "progression", "creating chances", "attacking transition",
                   "long ball", "ball in final third", "ball in the box", "players in the box", "goals")
DEFENDING_CODES = ("defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def majority(values):
    values = values[values >= 0]
    return int(np.bincount(values, minlength=2).argmax()) if len(values) >= 5 else -1


def build(data_dir=None):
    """The table for the challenge match: tracking signals plus the Smart Tagging columns."""
    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir)
    atd = load_atd(raw, N_FRAMES, os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz"), False)
    half = np.zeros(N_FRAMES, dtype=np.int8)
    for h, first, last in games.half_video_blocks(GAME):
        half[first:last + 1] = h
    out = compute(atd["px"], atd["py"], atd["ps"], atd["bx"], atd["by"],
                  bl.team_of_each_track(meta), half, LENGTH, WIDTH)
    out.update(tag_columns(data_dir))
    return out


def compute(px, py, ps, bx, by, team, half, length, width):
    """Tracking signals for any match in the ATD layout.

    px, py: (frames, slots) positions normalised 0-1, NaN when missing; ps: speeds in m/s;
    bx, by: ball, normalised; team: 0/1 per slot; half: 1/2 for frames to label, 0 otherwise.
    The challenge match and the external matches go through this same function, so a parameter
    tuned on one means the same thing on the other.
    """
    n_frames = len(bx)
    home, away = team == bl.HOME, team == bl.AWAY
    out = {"frame": np.arange(n_frames), "half": half.astype(np.int8)}
    LENGTH, WIDTH = length, width  # noqa: N806 -- same names as the module constants on purpose

    seen = np.isfinite(bx) & np.isfinite(by)
    out["ball_seen"] = seen
    out["ball_x"] = np.where(seen, bx * LENGTH, np.nan)
    out["ball_y"] = np.where(seen, by * WIDTH, np.nan)
    margin_x, margin_y = OUTSIDE_MARGIN_M / LENGTH, OUTSIDE_MARGIN_M / WIDTH
    out["ball_outside"] = seen & ((bx < -margin_x) | (bx > 1 + margin_x) | (by < -margin_y) | (by > 1 + margin_y))

    step = np.full(n_frames, np.nan)
    both = seen[1:] & seen[:-1]
    step[1:][both] = np.hypot(np.diff(bx.astype(np.float64)) * LENGTH,
                              np.diff(by.astype(np.float64)) * WIDTH)[both] * FPS
    out["ball_speed"] = pd.Series(step).rolling(BALL_SPEED_MEDIAN, center=True, min_periods=1).median().values

    dist = np.sqrt(((px - bx[:, None]) * LENGTH) ** 2 + ((py - by[:, None]) * WIDTH) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = dist[:, home].min(axis=1)
    d_away = dist[:, away].min(axis=1)
    out["d_home"] = np.where(np.isfinite(d_home), d_home, np.nan)
    out["d_away"] = np.where(np.isfinite(d_away), d_away, np.nan)
    nearest_slot = dist.argmin(axis=1)
    nearest = np.where(np.isfinite(dist.min(axis=1)), team[nearest_slot], -1).astype(np.int8)
    out["nearest_team"] = nearest

    occupied = np.isfinite(px) & np.isfinite(py)
    out["n_home"] = (occupied & home).sum(axis=1).astype(np.int8)
    out["n_away"] = (occupied & away).sum(axis=1).astype(np.int8)
    speed_ok = np.isfinite(ps)
    count = speed_ok.sum(axis=1)
    mean_speed = np.where(count > 0, np.where(speed_ok, ps, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    out["speed_mean"] = mean_speed
    out["speed_mean_2s"] = pd.Series(mean_speed).rolling(SPEED_WINDOW, center=True, min_periods=1).mean().values

    gap_len = np.zeros(n_frames, dtype=np.int32)
    gap_pos = np.zeros(n_frames, dtype=np.int32)
    before = np.full(n_frames, -1, dtype=np.int8)
    after = np.full(n_frames, -1, dtype=np.int8)
    for s, e in zip(*runs(~seen)):
        gap_len[s:e] = e - s
        gap_pos[s:e] = np.arange(e - s)
        before[s:e] = majority(nearest[max(s - GAP_CONTEXT, 0):s])
        after[s:e] = majority(nearest[e:e + GAP_CONTEXT])
    out["gap_len"], out["gap_pos"] = gap_len, gap_pos
    out["team_before_gap"], out["team_after_gap"] = before, after
    return out


def tag_columns(data_dir=None):
    """Smart Tagging as per-frame counts. Only the challenge match has it.

    Which team a tag points to follows the code definitions in OVERVIEW.md: attacking phases and
    set pieces are performed by the team with the ball; "Defending in ... third" and "Defensive
    transition" are performed by the team without it, so they count for the other side.
    """
    out = {}
    tag_home = np.zeros(N_FRAMES + 1, dtype=np.int32)
    tag_away = np.zeros(N_FRAMES + 1, dtype=np.int32)
    setpiece = np.zeros(N_FRAMES + 1, dtype=np.int32)
    for event in lst.load(GAME, data_dir):
        code = (event["code"] or "").strip().lower()
        a, b = max(event["start_frame"], 0), min(event["end_frame"], N_FRAMES)
        if b <= a:
            continue
        if code == "set pieces":
            setpiece[a] += 1
            setpiece[b] -= 1
        side = team_code(event["tags"].get("Team"))
        if side is None:
            continue
        if code in DEFENDING_CODES:
            side = 1 - side
        elif code not in ATTACKING_CODES and code != "set pieces":
            continue
        target = tag_home if side == 0 else tag_away
        target[a] += 1
        target[b] -= 1
    out["tag_home"] = np.cumsum(tag_home)[:N_FRAMES].astype(np.int8)
    out["tag_away"] = np.cumsum(tag_away)[:N_FRAMES].astype(np.int8)
    out["setpiece"] = np.cumsum(setpiece)[:N_FRAMES] > 0
    return out


def load(rebuild=False, data_dir=None):
    if rebuild or not os.path.isfile(TABLE_PATH):
        columns = build(data_dir)
        os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
        np.savez_compressed(TABLE_PATH, **columns)
    with np.load(TABLE_PATH) as saved:
        return pd.DataFrame({name: saved[name] for name in saved.files})


def check(table):
    """The nearest team in the table must reproduce the official raw baseline exactly."""
    labels = bl.smooth(table["nearest_team"].values.astype(np.int8), 0)
    committed = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv"))["label"].values
    return float((labels == committed).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    table = load(args.rebuild, args.data_dir)
    print("{} rows x {} columns -> {}".format(len(table), len(table.columns), TABLE_PATH))
    print("nearest team reproduces baseline_raw on {:.3%} of frames".format(check(table)))
