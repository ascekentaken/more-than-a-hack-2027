"""Read the Metrica Sports sample games into the same layout as the DFL matches (external/dfl.py).

    python external/metrica.py         # downloads, parses, caches and checks sample games 1 and 2

This is EXTERNAL data: two anonymised full matches released by Metrica Sports at
https://github.com/metrica-sports/sample-data ("please be responsible with the use of this data; if
you use it for anything public, please acknowledge the source"). Tracking at 25 fps for every player
and the ball, coordinates 0-1 on a 105 x 68 m pitch, and a manual event log.

Unlike the DFL files there is no per-frame possession or ball status, so H / A / D is rebuilt from
the event log:

  dead ball   from a stoppage -- the ball crosses a line (BALL OUT, a shot that goes OUT or into the
              GOAL: at the event's end frame), a foul (FAULT RECEIVED), an offside, a hand ball or the
              end of a half -- until the next restart (a SET PIECE, or a pass marked GOAL KICK); if
              something else is logged first, until that event
  in play     the team of the last event played on the ball (PASS, RECOVERY, SHOT, BALL LOST,
              SET PIECE, BALL OUT), from its start frame; duels (CHALLENGE) and cards do not move it

This is close to the DFL definition (the team last in control; dead between a stoppage and its
restart), but rebuilt by rule, so the labels are coarser at the edges of each possession.

Layout returned by load(game): as external/dfl.load; ps is the speed from the positions (m/s,
centred difference over SPEED_FRAMES). y is flipped so that it grows upwards like the DFL data.
"""

import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "external_data", "metrica")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SOURCE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{0}/Sample_Game_{0}_{1}.csv"

# Params
GAMES = {"MET1": 1, "MET2": 2}
LENGTH, WIDTH = 105.0, 68.0
FPS = 25
SPEED_FRAMES = 5
H, A, D = 0, 1, 2
ON_BALL = ("PASS", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE", "BALL OUT")
FILES = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")


def path(game, kind):
    return os.path.join(DATA_DIR, "Sample_Game_{}_{}.csv".format(GAMES[game], kind))


def download(game):
    os.makedirs(DATA_DIR, exist_ok=True)
    for kind in FILES:
        if not os.path.isfile(path(game, kind)):
            urllib.request.urlretrieve(SOURCE.format(GAMES[game], kind), path(game, kind))


def tracking(file):
    """(period, frame, players x, players y, ball x, ball y) from one team's file."""
    raw = pd.read_csv(file, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    x = raw[players].values.astype(np.float32)
    y = raw[[cols[cols.index(p) + 1] for p in players]].values.astype(np.float32)
    ball = cols.index("Ball")
    return (raw["Period"].values, raw["Frame"].values, x, y,
            raw.iloc[:, ball].values.astype(np.float32), raw.iloc[:, ball + 1].values.astype(np.float32))


def stoppage_frame(event):
    """Frame where the ball goes dead, or None if the event does not stop play."""
    kind, sub = event["Type"], str(event["Subtype"]) if pd.notna(event["Subtype"]) else ""
    end = event["End Frame"] if event["End Frame"] > 0 else event["Start Frame"]
    if kind == "BALL OUT":
        return end
    if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return end
    if kind == "FAULT RECEIVED":
        return event["Start Frame"]
    if kind == "BALL LOST" and any(w in sub for w in ("OFFSIDE", "HAND BALL", "END HALF")):
        return end
    return None


def is_restart(event):
    sub = str(event["Subtype"]) if pd.notna(event["Subtype"]) else ""
    return event["Type"] == "SET PIECE" or (event["Type"] in ("PASS", "BALL LOST") and "GOAL KICK" in sub)


def labels(events, frames):
    """H / A / D for every tracking frame number in `frames`, from the event log."""
    first, last = int(frames.min()), int(frames.max())
    label = np.full(last + 2, D, dtype=np.int8)
    events = events.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)
    team, dead_from = None, None
    marks = []  # (frame, code) changes, in order
    for _, e in events.iterrows():
        start = int(e["Start Frame"])
        if dead_from is not None:
            # dead until a restart; anything else logged on the ball also ends it (a missed restart)
            if is_restart(e) or e["Type"] in ON_BALL:
                dead_from = None
            else:
                continue
        if e["Type"] in ON_BALL:
            team = H if e["Team"] == "Home" else A
            marks.append((start, team))
        stop = stoppage_frame(e)
        if stop is not None:
            marks.append((int(stop), D))
            dead_from = int(stop)
    marks.sort(key=lambda m: m[0])
    code, at = D, first
    for frame, value in marks:
        frame = min(max(frame, first), last + 1)
        label[at:frame] = code
        code, at = value, max(at, frame)
    label[at:] = code
    return label[frames]


def build(game):
    download(game)
    home = tracking(path(game, FILES[0]))
    away = tracking(path(game, FILES[1]))
    if not np.array_equal(home[1], away[1]):
        raise ValueError("home and away frames differ")
    period, frame = home[0], home[1]
    px = np.concatenate([home[2], away[2]], axis=1)
    py = 1.0 - np.concatenate([home[3], away[3]], axis=1)
    bx, by = home[4], 1.0 - home[5]
    team = np.array([0] * home[2].shape[1] + [1] * away[2].shape[1], dtype=np.int8)
    k = SPEED_FRAMES // 2
    dx = (np.roll(px, -k, axis=0) - np.roll(px, k, axis=0)) * LENGTH
    dy = (np.roll(py, -k, axis=0) - np.roll(py, k, axis=0)) * WIDTH
    ps = (np.hypot(dx, dy) * FPS / (2 * k)).astype(np.float32)
    ps[:k], ps[-k:] = np.nan, np.nan
    events = pd.read_csv(path(game, FILES[2]))
    label = labels(events, frame)
    half = np.where(np.isin(period, (1, 2)), period, 0).astype(np.int8)
    # one empty frame between the halves, as in external/dfl.py
    cut = int(np.flatnonzero(period == 2)[0])
    blank_rows = lambda a: np.concatenate([a[:cut], np.full((1,) + a.shape[1:], np.nan, a.dtype), a[cut:]])
    out = {"px": blank_rows(px), "py": blank_rows(py), "ps": blank_rows(ps),
           "bx": blank_rows(bx), "by": blank_rows(by),
           "label": np.concatenate([label[:cut], [D], label[cut:]]).astype(np.int8),
           "half": np.concatenate([half[:cut], [0], half[cut:]]).astype(np.int8),
           "team": team, "length": np.array(LENGTH), "width": np.array(WIDTH)}
    ps_out = out["ps"]
    ps_out[cut - k:cut + k + 1] = np.nan  # no speed across the half-time join
    return out


def load(game, rebuild=False):
    file = os.path.join(CACHE_DIR, game + ".npz")
    if rebuild or not os.path.isfile(file):
        data = build(game)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(file, **data)
    with np.load(file) as saved:
        return {k: saved[k] for k in saved.files}


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import dfl
    for game in GAMES:
        c = dfl.checks(game, load(game, "--rebuild" in sys.argv))
        print("{}  {:5.1f} min  H {:.0%} A {:.0%} D {:.0%}  label changes {:4d}  players {:.1f}  "
              "nearest==possession {:.0%}  speed(positions)/speed(file) {:.2f}".format(
                  game, c["minutes"], c["H"], c["A"], c["D"], c["changes"], c["players_on_pitch"],
                  c["nearest_agrees_with_possession"], c["speed_ratio"]))
