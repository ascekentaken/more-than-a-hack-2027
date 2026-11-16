"""Look at one instant of the match in every file at once.

    python analysis/inspect_frame.py --frame 11185
    python analysis/inspect_frame.py --time 7:27.4            # video clock, as shown by a player
    python analysis/inspect_frame.py --frame 11185 12900 16000 --trail 75
    python analysis/inspect_frame.py --frame 11185 --no-video  # skip decoding the mp4

For each instant it writes, into analysis_out/frames/frame_<n>/:

  info.txt    everything the files say about that frame: the clock in each file's own units, the
              ground-truth label if there is one, the ball, every tracked slot with its team,
              position in metres, speed, distance to the ball and how long that slot has been
              continuously occupied, the homography, and the Smart Tagging phases active then
  pitch.png   the tracking seen from above, turned to match the camera, slots numbered
  video.png   the real video frame with the tracked slots and the ball drawn over it by the
              homography. If a dot is not on the player it claims to be, that is the tracking's
              mistake -- or the video seek's; see --video-offset

The point is to compare what you SEE in the video with what the NUMBERS say, one instant at a time.
Open the video in any player, jump to the time printed at the top of info.txt, and look.

Everything it writes is derived from the gated match data: it stays in analysis_out/ (gitignored).
Do not commit it, do not redistribute it, and do not send the images or the text to a hosted LLM.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
import load_homography as lh  # noqa: E402
import load_smart_tagging as lst  # noqa: E402
import possession_metrics as pm  # noqa: E402
import utils  # noqa: E402
from analyze_data import load_atd, team_code  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
LENGTH, WIDTH = games.GAMES[GAME]["pitch"]
N_FRAMES = games.source_frames("tactical", GAME)
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
LIFE_CAP_FRAMES = FPS * 3600
SEEK_BACK = 40  # frames to decode before the target, so an inexact seek still lands exactly

HOME_RGB, AWAY_RGB, BALL_RGB = (31, 94, 168), (178, 58, 47), (255, 214, 0)
HOME_HEX, AWAY_HEX, BALL_HEX = "#1f5ea8", "#b23a2f", "#ffd600"


def clock(frame):
    """Video clock as a player shows it: m:ss.ff, plus h:mm:ss for players that want hours."""
    seconds = frame / float(FPS)
    minutes, rest = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return "{}:{:05.2f}  ({}:{:02d}:{:05.2f})".format(
        int(seconds // 60), rest, hours, minutes, rest)


def parse_time(text):
    """'7:27', '7:27.4' or '1:02:03' (video clock) -> frame."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60.0 + float(part)
    return int(round(seconds * FPS))


def load_context(data_dir):
    paths = dict(zip(("atd_meta", "atd_raw", "homography", "smart_tagging"),
                     games.require(["atd_meta", "atd_raw", "homography", "smart_tagging"],
                                   GAME, data_dir)))
    cache = os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz")
    atd = load_atd(paths["atd_raw"], N_FRAMES, cache, False)
    matrices, reprojection, valid = lh.load(GAME, data_dir)
    events = lst.load(GAME, data_dir)
    for event in events:
        event["team"] = team_code(event["tags"].get("Team"))
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    return {
        "atd": atd,
        "team_of_slot": bl.team_of_each_track(paths["atd_meta"]),
        "names": bl.team_names(paths["atd_meta"]),
        "H": matrices, "reproj": reprojection, "valid": valid,
        "events": events, "gt": gt, "gt_segments": pm.segments(gt, pm.valid_runs(gt)),
        "data_dir": data_dir,
    }


def life_seconds(occupied, frame, slot):
    """How long this slot has been continuously occupied up to this frame, in seconds."""
    count, i = 0, frame
    while i >= 0 and occupied[i, slot] and count < LIFE_CAP_FRAMES:
        count += 1
        i -= 1
    return count / float(FPS)


def build_text(frame, ctx):
    atd = ctx["atd"]
    px, py, ps = atd["px"], atd["py"], atd["ps"]
    bx, by, bs = atd["bx"], atd["by"], atd["bs"]
    team = ctx["team_of_slot"]
    occupied = np.isfinite(px) & np.isfinite(py)
    ball_ok = np.isfinite(bx) & np.isfinite(by)
    half = games.half_of_video_frame(GAME, frame)

    out = []
    add = out.append
    add("FRAME {}".format(frame))
    add("  video clock (jump here in a player) : {}".format(clock(frame)))
    add("  ATD frame_id                        : {}   (video frame + 1)".format(frame + 1))
    add("  homography row                      : {}".format(frame))
    add("  Smart Tagging time                  : {:.2f} s".format(frame / float(FPS)))
    add("  part of the match                   : {}".format(
        {1: "first half (graded)", 2: "second half (graded)"}.get(half, "not graded (outside the two halves)")))

    gt_code = ctx["gt"][frame]
    add("")
    if gt_code == pm.X:
        add("GROUND TRUTH: none for this frame (only the first 5 minutes of play are public)")
    else:
        label = pm.LABELS[gt_code]
        for a, b, code in ctx["gt_segments"]:
            if a <= frame < b:
                add("GROUND TRUTH: {}  (this segment: frames {}-{}, {:.1f} s long; {:.1f} s into it)".format(
                    label, a, b - 1, (b - a) / float(FPS), (frame - a) / float(FPS)))
                break
        add("  H = {} in possession, A = {} in possession, D = ball dead".format(*ctx["names"]))

    add("")
    if ball_ok[frame]:
        add("BALL: tracked at ({:.1f} m, {:.1f} m)   third value {}".format(
            bx[frame] * LENGTH, by[frame] * WIDTH,
            "{:.2f}".format(bs[frame]) if np.isfinite(bs[frame]) else "n/a"))
        inside = 0 <= bx[frame] <= 1 and 0 <= by[frame] <= 1
        add("  inside the pitch rectangle: {}".format("yes" if inside else "NO (outside the lines)"))
    else:
        before = np.flatnonzero(ball_ok[:frame])
        after = np.flatnonzero(ball_ok[frame:])
        add("BALL: NOT tracked on this frame")
        add("  last seen {} ago; seen again {} later".format(
            "{:.1f} s".format((frame - before[-1]) / float(FPS)) if len(before) else "never",
            "{:.1f} s".format(after[0] / float(FPS)) if len(after) else "never"))

    slots = np.flatnonzero(occupied[frame])
    n_home = int((team[slots] == bl.HOME).sum())
    add("")
    add("TRACKED SLOTS: {}  (home {}, away {}) out of {} columns in the file".format(
        len(slots), n_home, len(slots) - n_home, px.shape[1]))
    add("  A slot is a column of the file, not a person. 'life' = seconds this slot has been")
    add("  continuously occupied; a short life means a track that has just (re)started.")
    add("")
    add("  slot  team   x_m    y_m   speed  to_ball_m  life_s")
    rows = []
    for s in slots:
        dist = (np.hypot((px[frame, s] - bx[frame]) * LENGTH, (py[frame, s] - by[frame]) * WIDTH)
                if ball_ok[frame] else np.nan)
        rows.append((int(team[s]), int(s), px[frame, s] * LENGTH, py[frame, s] * WIDTH,
                     ps[frame, s], dist, life_seconds(occupied, frame, s)))
    for t, s, x, y, v, d, life in sorted(rows):
        add("  {:>4}  {:<5} {:>6.1f} {:>6.1f} {:>6} {:>9} {:>7.1f}".format(
            s + 1, "home" if t == bl.HOME else "away", x, y,
            "{:.2f}".format(v) if np.isfinite(v) else "n/a",
            "{:.1f}".format(d) if np.isfinite(d) else "-", life))
    if ball_ok[frame] and rows:
        ranked = sorted((r for r in rows if np.isfinite(r[5])), key=lambda r: r[5])
        near = ranked[0]
        add("")
        add("  nearest to the ball: slot {} ({}) at {:.1f} m".format(
            near[1] + 1, "home" if near[0] == bl.HOME else "away", near[5]))
        for t, name in ((bl.HOME, "home"), (bl.AWAY, "away")):
            best = [r[5] for r in ranked if r[0] == t]
            if best:
                add("  nearest {:<4}: {:.1f} m".format(name, best[0]))

    add("")
    if ctx["valid"][frame]:
        add("HOMOGRAPHY: valid, reprojection error {:.3f} (lower is better)".format(ctx["reproj"][frame]))
    else:
        add("HOMOGRAPHY: not valid on this frame")

    add("")
    active = [e for e in ctx["events"] if e["start_frame"] <= frame < e["end_frame"]]
    add("SMART TAGGING active now: {}".format(len(active)))
    for e in sorted(active, key=lambda e: e["start_frame"]):
        extra = ", ".join("{}={}".format(k, v) for k, v in e["tags"].items() if k != "Half")
        add("  {:<30} frames {}-{}  ({:.0f} s left)  {}".format(
            e["code"], e["start_frame"], e["end_frame"], (e["end_frame"] - frame) / float(FPS), extra))
    near_tags = [e for e in ctx["events"]
                 if not (e["start_frame"] <= frame < e["end_frame"])
                 and abs(e["start_frame"] - frame) <= 2 * FPS]
    if near_tags:
        add("  starting within 2 s: " + "; ".join(
            "{} ({}{} s)".format(e["code"], "+" if e["start_frame"] > frame else "",
                                 (e["start_frame"] - frame) / float(FPS)) for e in near_tags))
    return "\n".join(out) + "\n"


def orientation(matrix):
    """(flip_x, flip_y) that make a top-down plot look like the camera view."""
    if not np.isfinite(matrix).all():
        return False, False
    x_left, _ = lh.project(matrix, 0.0, 0.5)
    x_right, _ = lh.project(matrix, 1.0, 0.5)
    _, y_low = lh.project(matrix, 0.5, 0.0)
    _, y_high = lh.project(matrix, 0.5, 1.0)
    return bool(x_left > x_right), bool(y_high > y_low)


def save_pitch(frame, ctx, path, trail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atd = ctx["atd"]
    px, py, bx, by = atd["px"], atd["py"], atd["bx"], atd["by"]
    team = ctx["team_of_slot"]
    occupied = np.isfinite(px) & np.isfinite(py)

    fig, ax = plt.subplots(figsize=(12, 7.6))
    utils.draw_pitch(ax, LENGTH, WIDTH)
    first = max(frame - trail, 0)
    for s in np.flatnonzero(occupied[frame]):
        colour = HOME_HEX if team[s] == bl.HOME else AWAY_HEX
        history = np.flatnonzero(occupied[first:frame + 1, s]) + first
        if trail and len(history) > 1:
            ax.plot(px[history, s] * LENGTH, py[history, s] * WIDTH, color=colour, lw=1.2,
                    alpha=0.7, zorder=2)
        ax.scatter(px[frame, s] * LENGTH, py[frame, s] * WIDTH, s=330, color=colour,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.text(px[frame, s] * LENGTH, py[frame, s] * WIDTH, str(s + 1), color="white",
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    ball = np.isfinite(bx[frame]) and np.isfinite(by[frame])
    if ball:
        if trail:
            seen = np.flatnonzero(np.isfinite(bx[first:frame + 1]) & np.isfinite(by[first:frame + 1])) + first
            ax.plot(bx[seen] * LENGTH, by[seen] * WIDTH, color="black", lw=1.0, zorder=4)
        ax.scatter(bx[frame] * LENGTH, by[frame] * WIDTH, s=130, color=BALL_HEX,
                   edgecolors="black", linewidths=1.5, zorder=5)

    gt_code = ctx["gt"][frame]
    ax.set_title("frame {}   video {}   ground truth: {}   ball: {}   blue = {} (H), red = {} (A)".format(
        frame, clock(frame).split("  ")[0], "none" if gt_code == pm.X else pm.LABELS[gt_code],
        "tracked" if ball else "NOT tracked", *ctx["names"]), fontsize=10)
    flip_x, flip_y = orientation(ctx["H"][frame])
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def read_video_frame(video_path, frame, offset):
    """Decode exactly one frame. Seeks a little early and walks forward, which is exact where a
    plain seek to the target is not."""
    import cv2
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("cannot open " + video_path)
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(frame + offset, 0)
    start = max(target - SEEK_BACK, 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    image = None
    for _ in range(target - start + 1):
        ok, image = capture.read()
        if not ok:
            image = None
            break
    capture.release()
    return image, reported


def save_video(frame, ctx, path, offset):
    import cv2
    video_path = games.require(["video"], GAME, ctx["data_dir"])[0]
    image, reported = read_video_frame(video_path, frame, offset)
    if image is None:
        return "could not decode frame {} of the video".format(frame)

    atd = ctx["atd"]
    px, py, bx, by = atd["px"], atd["py"], atd["bx"], atd["by"]
    team = ctx["team_of_slot"]
    occupied = np.isfinite(px) & np.isfinite(py)
    height, width = image.shape[:2]
    note = "video reports {} frames, expected {}; image {}x{}".format(reported, N_FRAMES, width, height)

    if not ctx["valid"][frame]:
        cv2.imwrite(path, image)
        return note + "; homography not valid here, nothing drawn"

    matrix = ctx["H"][frame]
    inside = total = 0
    for s in np.flatnonzero(occupied[frame]):
        u, v = lh.project(matrix, px[frame, s], py[frame, s])
        total += 1
        if not (0 <= u < width and 0 <= v < height):
            continue
        inside += 1
        rgb = HOME_RGB if team[s] == bl.HOME else AWAY_RGB
        centre = (int(round(u)), int(round(v)))
        cv2.circle(image, centre, 15, rgb[::-1], 3)
        cv2.putText(image, str(s + 1), (centre[0] - 10, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 3)
        cv2.putText(image, str(s + 1), (centre[0] - 10, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1)
    if np.isfinite(bx[frame]) and np.isfinite(by[frame]):
        u, v = lh.project(matrix, bx[frame], by[frame])
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(image, (int(round(u)), int(round(v))), 11, BALL_RGB[::-1], -1)
            cv2.circle(image, (int(round(u)), int(round(v))), 11, (0, 0, 0), 2)
    corners = np.array([lh.project(matrix, cx, cy) for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))])
    if np.isfinite(corners).all():
        cv2.polylines(image, [corners.round().astype(np.int32)], True, (255, 255, 255), 1)
    cv2.imwrite(path, image)
    return note + "; slots drawn inside the image: {}/{}".format(inside, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", type=int, nargs="+", help="video frame(s), counted from 0")
    parser.add_argument("--time", nargs="+", help="video clock(s) such as 7:27 or 1:02:03.5")
    parser.add_argument("--trail", type=int, default=50, help="frames of movement trail in pitch.png")
    parser.add_argument("--no-video", action="store_true", help="skip decoding the mp4")
    parser.add_argument("--video-offset", type=int, default=0,
                        help="shift the decoded video frame by this many frames if the picture is "
                             "visibly early or late against the tracking")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "analysis_out", "frames"))
    args = parser.parse_args()

    frames = list(args.frame or []) + [parse_time(t) for t in (args.time or [])]
    if not frames:
        parser.error("give at least one --frame or --time")
    for frame in frames:
        if not 0 <= frame < N_FRAMES:
            parser.error("frame {} is outside 0..{}".format(frame, N_FRAMES - 1))

    ctx = load_context(args.data_dir)
    for frame in frames:
        folder = os.path.join(args.out, "frame_{}".format(frame))
        os.makedirs(folder, exist_ok=True)
        text = build_text(frame, ctx)
        with open(os.path.join(folder, "info.txt"), "w", encoding="utf-8") as handle:
            handle.write(text)
        save_pitch(frame, ctx, os.path.join(folder, "pitch.png"), args.trail)
        message = "video skipped"
        if not args.no_video:
            try:
                message = save_video(frame, ctx, os.path.join(folder, "video.png"), args.video_offset)
            except (IOError, ImportError) as error:
                message = "video not written: {}".format(error)
        print(text)
        print("files in {}\n  video: {}\n".format(folder, message))


if __name__ == "__main__":
    main()
