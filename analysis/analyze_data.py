"""Exploratory analysis of the provided match data. Run it locally; it writes a report you read.

    python analysis/analyze_data.py                  # tactical source, report in ./analysis_out
    python analysis/analyze_data.py --broadcast      # plus a short broadcast summary
    python analysis/analyze_data.py --rebuild        # ignore the parsed-ATD cache

Everything it writes goes to analysis_out/ (gitignored). The terms of use forbid redistributing the
match data and anything that would reconstruct it, and forbid sending it, or extracts of it, to
hosted LLMs -- so keep analysis_out/ out of the public repository, and out of chat windows.

What it answers, in order (each is a section of report.md):

  1. Alignment        does the ATD have one record per video frame?
  2. Presence         how many players, how often is the ball seen, by half and by minute
  3. Ball gaps        how long are the holes, and how many would linear interpolation close
  4. Ball plausibility out-of-pitch positions, implausible jumps
  5. Ball vs players  how close is the nearest player, how contested is the ball
  6. Player motion    speed distribution (a stoppage should look different from play)
  7. Smart Tagging   codes, teams, set pieces, how much of the match it covers
  8. Homography      validity and reprojection error on graded frames
  9. Public window   the 5 minutes with ground truth: every segment and every transition against
                     the signals above, plus a check that the parsing reproduces the official
                     baselines exactly
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
import load_homography as lh  # noqa: E402
import load_smart_tagging as lst  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
LENGTH, WIDTH = games.GAMES[GAME]["pitch"]
N_FRAMES = games.source_frames("tactical", GAME)
PUBLIC_WINDOW = (11185, 18684)  # inclusive, the frames the public ground truth covers
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

GAP_LIMITS = (5, 12, 25, 50, 125, 250)  # frames; 25 = 1 s
JUMP_MS = 40.0  # a ball moving faster than this between two frames is a tracking error
NEAR_M = (1.0, 2.0, 3.0, 5.0)
SMOOTH_WINDOWS = (0, 25, 50, 125)
TRANSITION_CONTEXT = 25  # frames each side of a ground-truth transition
TAG_CONTEXT = 50

# Tags whose Team is, by their definition, the side that has the ball. Whether that is what the
# file really does is checked against the public ground truth in section 9 rather than assumed.
POSSESSION_CODES = ("build up", "progression", "creating chances", "attacking transition",
                    "long ball", "ball in final third", "ball in the box", "players in the box")

COLOURS = {"H": "#1f77b4", "A": "#d62728", "D": "#999999", "X": "#ffffff"}


# -- small helpers -----------------------------------------------------------------------------

class Report(object):
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        self.lines.append(text)

    def h(self, title):
        self.add("")
        self.add("## " + title)
        self.add("")

    def table(self, headers, rows):
        self.add("| " + " | ".join(str(h) for h in headers) + " |")
        self.add("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            self.add("| " + " | ".join(str(c) for c in row) + " |")
        self.add("")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines) + "\n")


def pct(x, digits=1):
    return "{:.{d}f}%".format(100.0 * float(x), d=digits)


def num(x, digits=2):
    return "nan" if x is None or not np.isfinite(x) else "{:.{d}f}".format(float(x), d=digits)


def runs_of(mask):
    """(starts, stops) of the True runs of a boolean array, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def percentiles(values, qs=(50, 90, 99)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return [float(np.percentile(values, q)) if len(values) else float("nan") for q in qs]


def team_code(name):
    """H / A / None for a club name as written in the tagging or the tracking metadata."""
    lowered = (name or "").strip().lower()
    if "barcel" in lowered:
        return 0
    if "atletic" in lowered or "athletic" in lowered:
        return 1
    return None


# -- ATD ---------------------------------------------------------------------------------------

def parse_atd(raw_path, n_frames):
    """Stream the raw ATD into arrays indexed by video frame (frame_id - 1).

    Line format: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,...>"; a missing observation starts with N.
    """
    px = py = ps = None
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bs = np.full(n_frames, np.nan, dtype=np.float32)
    present = np.zeros(n_frames, dtype=bool)
    started = time.time()

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                raise ValueError("frame_id {} outside 1..{}".format(i + 1, n_frames))
            present[i] = True

            groups = [g for g in parts[1].split(";") if g]
            if px is None:
                n_slots = len(groups)
                px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
            if len(groups) > px.shape[1]:
                raise ValueError("frame {} has {} track groups, expected {}".format(
                    i + 1, len(groups), px.shape[1]))
            for slot, group in enumerate(groups):
                if group[0] == "N":
                    continue
                fields = group.split(",")
                px[i, slot] = float(fields[0])
                py[i, slot] = float(fields[1])
                if len(fields) > 2:
                    ps[i, slot] = float(fields[2])

            ball = parts[2].strip()
            if ball and ball[0] != "N":
                fields = ball.split(",")
                bx[i] = float(fields[0])
                by[i] = float(fields[1])
                if len(fields) > 2:
                    bs[i] = float(fields[2])
    print("  parsed {} records in {:.0f}s".format(int(present.sum()), time.time() - started))
    return dict(px=px, py=py, ps=ps, bx=bx, by=by, bs=bs, present=present)


def load_atd(raw_path, n_frames, cache_path, rebuild):
    if not rebuild and os.path.isfile(cache_path):
        stamp = (os.path.getsize(raw_path), int(os.path.getmtime(raw_path)))
        with np.load(cache_path) as cached:
            if tuple(cached["stamp"]) == stamp:
                print("  using cache {}".format(cache_path))
                return {k: cached[k] for k in cached.files if k != "stamp"}
    data = parse_atd(raw_path, n_frames)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    np.savez_compressed(cache_path, stamp=stamp, **data)
    return data


def xml_structure(path):
    """Tag paths with counts and attribute names -- what the metadata file actually declares."""
    seen = {}

    def walk(element, prefix):
        here = prefix + "/" + element.tag
        entry = seen.setdefault(here, {"n": 0, "attrs": set(), "text": set()})
        entry["n"] += 1
        entry["attrs"].update(element.attrib.keys())
        text = (element.text or "").strip()
        if text and len(entry["text"]) < 4:
            entry["text"].add(text[:60])
        for child in element:
            walk(child, here)

    walk(ET.parse(path).getroot(), "")
    return seen


# -- the analysis ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="defaults to ./data or $MTAH_DATA_DIR")
    parser.add_argument("--out", default=os.path.join(ROOT, "analysis_out"))
    parser.add_argument("--rebuild", action="store_true", help="ignore the parsed-ATD cache")
    parser.add_argument("--broadcast", action="store_true", help="add a broadcast summary")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    report = Report()
    report.add("# Data analysis -- BAR-ATH tactical")
    report.add("")
    report.add("Generated locally. Contains figures derived from the match data: do not commit it, "
               "do not redistribute it, do not paste it into a hosted LLM.")

    paths = dict(zip(("atd_meta", "atd_raw", "homography", "smart_tagging"),
                     games.require(["atd_meta", "atd_raw", "homography", "smart_tagging"],
                                   GAME, args.data_dir)))

    print("ATD")
    atd = load_atd(paths["atd_raw"], N_FRAMES, os.path.join(out, "cache", "atd_tactical.npz"),
                   args.rebuild)
    px, py, ps = atd["px"], atd["py"], atd["ps"]
    bx, by, bs, present = atd["bx"], atd["by"], atd["bs"], atd["present"]
    n_slots = px.shape[1]

    team_of_slot = bl.team_of_each_track(paths["atd_meta"])
    if len(team_of_slot) != n_slots:
        raise ValueError("metadata lists {} track slots, the ATD has {}".format(
            len(team_of_slot), n_slots))
    home_slots, away_slots = team_of_slot == bl.HOME, team_of_slot == bl.AWAY
    home_name, away_name = bl.team_names(paths["atd_meta"])

    blocks = games.half_video_blocks(GAME)
    graded = np.zeros(N_FRAMES, dtype=bool)
    half_of = np.zeros(N_FRAMES, dtype=np.int8)
    for half, first, last in blocks:
        graded[first:last + 1] = True
        half_of[first:last + 1] = half
    n_graded = int(graded.sum())
    lo, hi = PUBLIC_WINDOW
    window = slice(lo, hi + 1)

    ball_ok = np.isfinite(bx) & np.isfinite(by)
    player_ok = np.isfinite(px) & np.isfinite(py)
    n_players = player_ok.sum(axis=1)
    n_home = (player_ok & home_slots).sum(axis=1)
    n_away = (player_ok & away_slots).sum(axis=1)

    # 1. alignment -----------------------------------------------------------------------------
    report.h("1. Alignment and structure")
    report.add("Teams from the metadata: H = {}, A = {}. Track slots: {} ({} home, {} away).".format(
        home_name, away_name, n_slots, int(home_slots.sum()), int(away_slots.sum())))
    report.add("")
    report.table(["check", "result"], [
        ["records parsed", int(present.sum())],
        ["expected (one per video frame)", N_FRAMES],
        ["every video frame has a record", bool(present.all())],
        ["graded frames", "{} ({} + {})".format(n_graded, *[b - a + 1 for _, a, b in blocks])],
    ])
    structure = xml_structure(paths["atd_meta"])
    with open(os.path.join(out, "atd_xml_structure.txt"), "w", encoding="utf-8") as handle:
        for tag in sorted(structure):
            e = structure[tag]
            handle.write("{:<70} n={:<5} attrs={} text={}\n".format(
                tag, e["n"], sorted(e["attrs"]), sorted(e["text"])))
    report.add("The tracking metadata declares {} distinct element paths; the full list, with "
               "attribute names, is in `atd_xml_structure.txt`. Read it once to confirm there is "
               "no channel beyond x, y and speed.".format(len(structure)))

    # 2. presence ------------------------------------------------------------------------------
    report.h("2. Presence: players and ball")
    rows = [["whole video", pct(ball_ok.mean())],
            ["graded frames", pct(ball_ok[graded].mean())]]
    for half, first, last in blocks:
        rows.append(["half {}".format(half), pct(ball_ok[first:last + 1].mean())])
    rows.append(["public window", pct(ball_ok[window].mean())])
    report.add("Ball present (both coordinates):")
    report.add("")
    report.table(["scope", "ball present"], rows)

    g = graded
    report.add("Tracked objects per graded frame: mean {:.1f}, home {:.1f}, away {:.1f}.".format(
        n_players[g].mean(), n_home[g].mean(), n_away[g].mean()))
    report.add("")
    report.table(["objects tracked", "share of graded frames"], [
        ["fewer than 18", pct((n_players[g] < 18).mean())],
        ["18 to 22", pct(((n_players[g] >= 18) & (n_players[g] <= 22)).mean())],
        ["more than 22 (duplicates)", pct((n_players[g] > 22).mean())],
    ])

    per_minute = []
    for minute in range(N_FRAMES // (FPS * 60)):
        a, b = minute * FPS * 60, (minute + 1) * FPS * 60
        per_minute.append((minute, graded[a:b].mean(), ball_ok[a:b].mean(), n_players[a:b].mean()))
    pd.DataFrame(per_minute, columns=["video_minute", "share_graded", "ball_present",
                                      "mean_objects"]).to_csv(
        os.path.join(out, "per_minute.csv"), index=False)
    report.add("Per-minute table: `per_minute.csv`.")

    # 3. ball gaps -----------------------------------------------------------------------------
    report.h("3. Ball gaps (graded frames)")
    gap_lengths, interior_gaps = [], []
    for half, first, last in blocks:
        missing = ~ball_ok[first:last + 1]
        starts, stops = runs_of(missing)
        for s, e in zip(starts, stops):
            gap_lengths.append(e - s)
            if s > 0 and e < len(missing):
                interior_gaps.append(e - s)
    gap_lengths = np.array(gap_lengths)
    interior_gaps = np.array(interior_gaps)
    total_missing = gap_lengths.sum()
    p50, p90, p99 = percentiles(gap_lengths)
    report.add("{} gaps covering {} frames. Gap length in frames: median {:.0f}, p90 {:.0f}, "
               "p99 {:.0f}, longest {} ({:.1f} s).".format(
                   len(gap_lengths), total_missing, p50, p90, p99, gap_lengths.max(),
                   gap_lengths.max() / float(FPS)))
    report.add("")
    rows = []
    for limit in GAP_LIMITS:
        closed = interior_gaps[interior_gaps <= limit].sum()
        rows.append(["<= {} ({:.1f} s)".format(limit, limit / float(FPS)),
                     pct(gap_lengths[gap_lengths <= limit].sum() / float(total_missing)),
                     pct((ball_ok[graded].sum() + closed) / float(n_graded))])
    report.table(["gap length", "share of missing frames in gaps this short",
                  "ball coverage if those interior gaps were interpolated"], rows)

    # 4. plausibility --------------------------------------------------------------------------
    report.h("4. Ball plausibility")
    gb = graded & ball_ok
    outside = (bx[gb] < 0) | (bx[gb] > 1) | (by[gb] < 0) | (by[gb] > 1)
    report.add("Graded frames with the ball outside the 0-1 pitch square: {} ({} of frames with a "
               "ball).".format(int(outside.sum()), pct(outside.mean())))
    step = np.hypot(np.diff(bx.astype(np.float64)) * LENGTH,
                    np.diff(by.astype(np.float64)) * WIDTH) * FPS
    pair = graded[1:] & graded[:-1] & ball_ok[1:] & ball_ok[:-1]
    speeds = step[pair]
    p50, p90, p99 = percentiles(speeds)
    p999 = percentiles(speeds, (99.9,))[0]
    report.add("")
    report.add("Ball speed from consecutive-frame positions, m/s: median {}, p90 {}, p99 {}, "
               "p99.9 {}. Frame pairs above {:.0f} m/s: {} ({}).".format(
                   num(p50), num(p90), num(p99), num(p999), JUMP_MS,
                   int((speeds > JUMP_MS).sum()), pct((speeds > JUMP_MS).mean(), 2)))
    if np.isfinite(bs[graded]).any():
        p50, p90, p99 = percentiles(bs[graded])
        report.add("")
        report.add("The ATD's own ball third value on graded frames: median {}, p90 {}, p99 {}. "
                   "Check its units against the position-derived speed above before using it."
                   .format(num(p50), num(p90), num(p99)))

    # 5. ball vs players -----------------------------------------------------------------------
    report.h("5. Ball vs players")
    dx = (px - bx[:, None]) * LENGTH
    dy = (py - by[:, None]) * WIDTH
    dist = np.sqrt(dx * dx + dy * dy)
    dist[~ball_ok] = np.nan
    filled = np.where(np.isfinite(dist), dist, np.inf)
    nearest_slot = filled.argmin(axis=1)
    nearest_dist = filled[np.arange(N_FRAMES), nearest_slot]
    nearest_dist[~np.isfinite(nearest_dist)] = np.nan
    nearest_team = np.where(np.isfinite(nearest_dist), team_of_slot[nearest_slot], bl.NOBODY)

    d_home = np.where(home_slots[None, :], filled, np.inf).min(axis=1)
    d_away = np.where(away_slots[None, :], filled, np.inf).min(axis=1)
    both = np.isfinite(d_home) & np.isfinite(d_away)
    margin = np.full(N_FRAMES, np.nan)
    margin[both] = np.abs(d_home[both] - d_away[both])

    nd = nearest_dist[gb]
    nd = nd[np.isfinite(nd)]
    p50, p90, p99 = percentiles(nd)
    report.add("Distance from the ball to the nearest tracked player, graded frames with a ball: "
               "median {} m, p90 {} m, p99 {} m.".format(num(p50), num(p90), num(p99)))
    report.add("")
    report.table(["nearest player within", "share of frames"],
                 [["{:.0f} m".format(m), pct((nd <= m).mean())] for m in NEAR_M])
    mg = margin[gb]
    mg = mg[np.isfinite(mg)]
    report.add("Gap between the nearest home and nearest away player (a small gap = contested "
               "ball): median {} m; under 1 m on {} of frames, under 2 m on {}.".format(
                   num(np.median(mg)), pct((mg < 1).mean()), pct((mg < 2).mean())))

    # 6. player motion -------------------------------------------------------------------------
    report.h("6. Player motion")
    speed_ok = np.isfinite(ps)
    mean_speed = np.where(speed_ok.sum(axis=1) > 0,
                          np.where(speed_ok, ps, 0.0).sum(axis=1) / np.maximum(speed_ok.sum(axis=1), 1),
                          np.nan)
    p50, p90, p99 = percentiles(ps[graded][speed_ok[graded]])
    report.add("Player speed (the ATD's third value), graded frames: median {}, p90 {}, p99 {} "
               "(units as stated in the ATD; m/s per DATA.md).".format(num(p50), num(p90), num(p99)))
    report.add("")
    q50, q10, q90 = percentiles(mean_speed[graded], (50, 10, 90))
    report.add("Mean speed over all tracked players per frame: p10 {}, median {}, p90 {}."
               .format(num(q10), num(q50), num(q90)))
    out_pitch = player_ok & ((px < 0) | (px > 1) | (py < 0) | (py > 1))
    report.add("")
    report.add("Player observations outside the 0-1 square on graded frames: {}."
               .format(pct(out_pitch[graded].sum() / float(max(player_ok[graded].sum(), 1)), 2)))

    # 7. smart tagging -------------------------------------------------------------------------
    report.h("7. Smart Tagging")
    events = lst.load(GAME, args.data_dir)
    for event in events:
        event["team"] = team_code(event["tags"].get("Team"))
    table = pd.DataFrame([{"code": e["code"], "team": e["tags"].get("Team"),
                           "half": e["tags"].get("Half"), "type": e["tags"].get("Type"),
                           "start": e["start_frame"], "end": e["end_frame"]} for e in events])
    table["duration_s"] = (table["end"] - table["start"]) / float(FPS)
    table["team_na"] = table["team"].map(team_code).isna()
    summary = table.groupby("code").agg(n=("code", "size"), mean_s=("duration_s", "mean"),
                                        median_s=("duration_s", "median"),
                                        n_team_na=("team_na", "sum"))
    summary = summary.sort_values("n", ascending=False)
    report.add("{} instances.".format(len(events)))
    report.add("")
    report.table(["code", "n", "mean duration s", "median duration s", "team not H/A"],
                 [[code, int(r.n), num(r.mean_s, 1), num(r.median_s, 1), int(r.n_team_na)]
                  for code, r in summary.iterrows()])
    table.to_csv(os.path.join(out, "smart_tagging_frames.csv"), index=False)

    setpieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    sp_rows = sorted(((e["start_frame"], e["end_frame"], e["tags"].get("Type"),
                       e["tags"].get("Team"), e["tags"].get("Half")) for e in setpieces))
    pd.DataFrame(sp_rows, columns=["start_frame", "end_frame", "type", "team", "half"]).to_csv(
        os.path.join(out, "setpieces.csv"), index=False)
    kinds = pd.Series([r[2] for r in sp_rows]).value_counts()
    report.add("Set pieces by type (list with frames in `setpieces.csv`): " +
               ", ".join("{} {}".format(k, v) for k, v in kinds.items()) + ".")
    na = [r for r in sp_rows if team_code(r[3]) is None]
    report.add("")
    report.add("Set pieces whose team is not Home/Away: {} of {}.".format(len(na), len(sp_rows)))

    # tag-implied possession over the match
    n_h = np.zeros(N_FRAMES + 1, dtype=np.int32)
    n_a = np.zeros(N_FRAMES + 1, dtype=np.int32)
    for e in events:
        if (e["code"] or "").strip().lower() in POSSESSION_CODES and e["team"] is not None:
            a, b = max(e["start_frame"], 0), min(e["end_frame"], N_FRAMES)
            (n_h if e["team"] == 0 else n_a)[a] += 1
            (n_h if e["team"] == 0 else n_a)[b] -= 1
    n_h, n_a = np.cumsum(n_h)[:N_FRAMES], np.cumsum(n_a)[:N_FRAMES]
    tag_team = np.where((n_h > 0) & (n_a > 0), 2, np.where(n_h > 0, 0, np.where(n_a > 0, 1, -1)))
    report.add("")
    report.table(["graded frames covered by a possession-type phase", "share"], [
        ["only home", pct((tag_team[graded] == 0).mean())],
        ["only away", pct((tag_team[graded] == 1).mean())],
        ["both teams at once (conflict)", pct((tag_team[graded] == 2).mean())],
        ["no phase at all", pct((tag_team[graded] == -1).mean())],
    ])

    # 8. homography ----------------------------------------------------------------------------
    report.h("8. Homography (tactical)")
    _, reproj, valid = lh.load(GAME, args.data_dir)
    usable = reproj[graded & valid]
    p50, p90, p99 = percentiles(usable)
    report.add("Valid on {} of all frames and {} of graded frames. Reprojection error on graded "
               "valid frames: median {}, p90 {}, p99 {}; above 1.0 on {}.".format(
                   pct(valid.mean()), pct(valid[graded].mean()), num(p50, 3), num(p90, 3),
                   num(p99, 3), pct((usable > 1.0).mean())))

    # 9. public window -------------------------------------------------------------------------
    report.h("9. The public ground-truth window")
    gt_codes = pm.label_codes(pm.read_label_csv(GT_PATH))
    runs = pm.valid_runs(gt_codes)

    # parsing check: our arrays must reproduce the official baselines exactly
    rows = []
    for window_size in SMOOTH_WINDOWS:
        labels = bl.smooth(nearest_team.astype(np.int8), window_size)
        result = pm.evaluate(gt_codes, pm.label_codes(labels))
        name = "baseline_smooth{}".format(window_size) if window_size else "baseline_raw"
        committed = os.path.join(RESULTS_DIR, name + ".csv")
        same = "n/a"
        if os.path.isfile(committed):
            reference = pm.read_label_csv(committed)["label"].astype(str).str.strip().values
            same = pct((reference == labels).mean(), 3)
        rows.append([name, num(result["macro_f1"], 4), num(result["spell_f1_mean"], 4),
                     result["n_pred_spells"], same])
    report.add("Scores of the nearest-player baseline rebuilt from the parsed arrays, on the public "
               "window. README.md lists macroF1/spellF1 0.5972/0.0378, 0.6200/0.2868, "
               "0.6207/0.3256, 0.5812/0.4242. The last column is how many of the 199319 labels "
               "equal the committed baseline file -- it should be 100%.")
    report.add("")
    report.table(["variant", "macroF1", "spellF1", "possessions", "== committed file"], rows)

    smooth50 = bl.smooth(nearest_team.astype(np.int8), 50)
    gt_segments = pm.segments(gt_codes, runs)
    label_of = {0: "H", 1: "A", 2: "D"}
    in_window = np.zeros(N_FRAMES, dtype=bool)
    in_window[window] = True

    def tags_in(a, b, only_starting=False):
        found = []
        for e in events:
            hit = (a <= e["start_frame"] < b) if only_starting else (e["start_frame"] < b and e["end_frame"] > a)
            if hit:
                tag = e["code"]
                if e["tags"].get("Type"):
                    tag += "[" + e["tags"]["Type"] + "]"
                found.append("{}/{}".format(tag, {0: "H", 1: "A", None: "-"}[e["team"]]))
        return "; ".join(found)

    seg_rows = []
    for a, b, code in gt_segments:
        sl = slice(a, b)
        near = nearest_team[sl]
        known = near >= 0
        agree = float((near[known] == code).mean()) if code in (0, 1) and known.any() else float("nan")
        seg_rows.append({
            "start": a, "end": b - 1, "label": label_of[code], "seconds": (b - a) / float(FPS),
            "ball_present": float(ball_ok[sl].mean()),
            "ball_outside_pitch": float(((bx[sl] < 0) | (bx[sl] > 1) | (by[sl] < 0) | (by[sl] > 1)).mean()),
            "nearest_dist_m": float(np.nanmean(nearest_dist[sl])) if known.any() else float("nan"),
            "nearest_team_agrees": agree,
            "mean_player_speed": float(np.nanmean(mean_speed[sl])),
            "tags": tags_in(a, b),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(os.path.join(out, "public_window_segments.csv"), index=False)
    report.add("Each ground-truth segment against the signals (full table with the overlapping "
               "tags: `public_window_segments.csv`):")
    report.add("")
    report.table(["frames", "label", "s", "ball seen", "nearest m", "nearest team = label",
                  "mean speed"],
                 [["{}-{}".format(r.start, r.end), r.label, num(r.seconds, 1), pct(r.ball_present, 0),
                   num(r.nearest_dist_m, 1), pct(r.nearest_team_agrees, 0)
                   if np.isfinite(r.nearest_team_agrees) else "-", num(r.mean_player_speed, 2)]
                  for r in seg_df.itertuples()])

    # dead vs alive in the window
    dead = (gt_codes == pm.D) & in_window
    alive = ((gt_codes == pm.H) | (gt_codes == pm.A)) & in_window
    rows = []
    for name, fn in (("ball missing", lambda m: 1 - ball_ok[m].mean()),
                     ("ball outside pitch", lambda m: ((bx[m] < 0) | (bx[m] > 1) | (by[m] < 0) | (by[m] > 1)).mean()),
                     ("median of per-frame mean player speed", lambda m: np.nanmedian(mean_speed[m])),
                     ("median nearest distance (m)", lambda m: np.nanmedian(nearest_dist[m])),
                     ("median ball step speed (m/s)", lambda m: np.nanmedian(step[np.flatnonzero(m[:-1] & m[1:] & ball_ok[:-1] & ball_ok[1:])]))):
        rows.append([name, num(fn(alive)), num(fn(dead))])
    report.add("Dead-ball frames vs alive frames in the window ({} vs {} frames):".format(
        int(dead.sum()), int(alive.sum())))
    report.add("")
    report.table(["signal", "alive (H/A)", "dead (D)"], rows)

    # transitions
    transition_rows = []
    for frame, was, now in pm.transitions(gt_codes, runs):
        before = nearest_team[max(frame - TRANSITION_CONTEXT, 0):frame]
        after = nearest_team[frame:frame + TRANSITION_CONTEXT]

        def mode(values):
            values = values[values >= 0]
            return "-" if not len(values) else "HA"[int(np.bincount(values, minlength=2).argmax())]
        transition_rows.append({
            "frame": frame, "was": label_of[was], "to": label_of[now],
            "ball_seen_before": float(ball_ok[max(frame - TRANSITION_CONTEXT, 0):frame].mean()),
            "ball_seen_after": float(ball_ok[frame:frame + TRANSITION_CONTEXT].mean()),
            "nearest_before": mode(before), "nearest_after": mode(after),
            "tags_starting_within_2s": tags_in(frame - TAG_CONTEXT, frame + TAG_CONTEXT, True),
        })
    tr_df = pd.DataFrame(transition_rows)
    tr_df.to_csv(os.path.join(out, "public_window_transitions.csv"), index=False)
    report.add("Ground-truth transitions with ball coverage and nearest-player team 1 s before and "
               "after (`public_window_transitions.csv` adds the tags that start within 2 s):")
    report.add("")
    report.table(["frame", "change", "ball seen before", "ball seen after", "nearest before",
                  "nearest after"],
                 [[r.frame, "{}>{}".format(r.was, r.to), pct(r.ball_seen_before, 0),
                   pct(r.ball_seen_after, 0), r.nearest_before, r.nearest_after]
                  for r in tr_df.itertuples()])

    # does the tag team mean "has the ball"?
    report.add("Does a tag's Team mean the side in possession? Share of alive ground-truth frames "
               "in the window where the tag is active and its team equals the ground-truth team "
               "(near 100% = same side as possession; near 0% = the other side; ~50% = no "
               "information). Tiny samples -- read the frame counts.")
    report.add("")
    rows = []
    for code in sorted({(e["code"] or "").lower() for e in events}):
        n_equal = n_total = 0
        for e in events:
            if (e["code"] or "").lower() != code or e["team"] is None:
                continue
            a, b = max(e["start_frame"], lo), min(e["end_frame"], hi + 1)
            if b <= a:
                continue
            labels = gt_codes[a:b]
            labels = labels[(labels == pm.H) | (labels == pm.A)]
            n_total += len(labels)
            n_equal += int((labels == e["team"]).sum())
        if n_total:
            rows.append([code, n_total, pct(n_equal / float(n_total), 0)])
    report.table(["code", "frames in window", "tag team == possession"], rows)

    # 10. broadcast ----------------------------------------------------------------------------
    if args.broadcast:
        report.h("10. Broadcast (summary only)")
        raw = games.require(["atd_raw"], GAME, args.data_dir, "broadcast")[0]
        n_b = games.source_frames("broadcast", GAME)
        broadcast = load_atd(raw, n_b, os.path.join(out, "cache", "atd_broadcast.npz"),
                             args.rebuild)
        _, _, valid_b = lh.load(GAME, args.data_dir, "broadcast")
        ball_b = np.isfinite(broadcast["bx"]) & np.isfinite(broadcast["by"])
        report.add("{} records, ball present on {}, homography valid on {}. It is not aligned to "
                   "the tactical timeline.".format(int(broadcast["present"].sum()),
                                                   pct(ball_b.mean()), pct(valid_b.mean())))

    # plots ------------------------------------------------------------------------------------
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(13, 3))
        ax.fill_between([m for m, *_ in per_minute], [100 * c for _, _, c, _ in per_minute],
                        color="#1f77b4", alpha=.8)
        ax.axhline(100 * ball_ok[graded].mean(), color="#d62728", ls="--", label="graded average")
        ax.set_xlabel("minute of video"); ax.set_ylabel("% frames with a ball")
        ax.set_ylim(0, 100); ax.legend(); ax.set_title("Ball coverage per minute")
        fig.savefig(os.path.join(out, "ball_coverage_per_minute.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        axes[0].hist(gap_lengths, bins=np.logspace(0, np.log10(max(gap_lengths.max(), 10)), 40))
        axes[0].set_xscale("log"); axes[0].set_title("ball gap length (frames)")
        axes[1].hist(np.clip(nd, 0, 15), bins=60); axes[1].set_title("nearest player to ball (m)")
        axes[2].hist(np.clip(speeds, 0, 60), bins=60); axes[2].set_yscale("log")
        axes[2].set_title("ball speed between frames (m/s)")
        fig.savefig(os.path.join(out, "distributions.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        # the public window as a timeline
        span = np.arange(lo, hi + 1)
        palette = ListedColormap([COLOURS["H"], COLOURS["A"], COLOURS["D"]])
        fig, axes = plt.subplots(6, 1, figsize=(15, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1, 1, 1, 2, 2]})

        def strip(ax, codes, title):
            ax.imshow(np.asarray(codes)[None, :], aspect="auto", cmap=palette, vmin=0, vmax=2,
                      extent=[lo, hi + 1, 0, 1], interpolation="nearest")
            ax.set_yticks([]); ax.set_ylabel(title, rotation=0, ha="right", va="center")

        strip(axes[0], gt_codes[window], "ground truth")
        strip(axes[1], pm.label_codes(smooth50)[window], "baseline +/-50")
        tag_strip = np.where(tag_team[window] == 2, 2, np.where(tag_team[window] < 0, 2, tag_team[window]))
        strip(axes[2], tag_strip, "tag team")
        axes[3].fill_between(span, ball_ok[window].astype(int), step="mid", color="#444")
        axes[3].set_yticks([]); axes[3].set_ylabel("ball seen", rotation=0, ha="right", va="center")
        axes[4].plot(span, nearest_dist[window], lw=.7)
        axes[4].set_ylim(0, 20); axes[4].set_ylabel("nearest player (m)")
        axes[5].plot(span, mean_speed[window], lw=.7)
        axes[5].set_ylabel("mean player speed"); axes[5].set_xlabel("video frame")
        for a, b, code in gt_segments:
            for ax in axes:
                ax.axvline(a, color="k", lw=.3, alpha=.4)
        fig.savefig(os.path.join(out, "public_window_timeline.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    report.h("Files written")
    report.add("report.md, per_minute.csv, atd_xml_structure.txt, smart_tagging_frames.csv, "
               "setpieces.csv, public_window_segments.csv, public_window_transitions.csv, "
               "and (unless --no-plots) three PNGs. All in `{}`.".format(out))
    report.save(os.path.join(out, "report.md"))
    print("\nreport: {}".format(os.path.join(out, "report.md")))


if __name__ == "__main__":
    main()
