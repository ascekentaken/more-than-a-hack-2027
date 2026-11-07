"""The five public minutes of video with the ground-truth labels drawn underneath.

    python analysis/gt_video.py              # analysis_out/gt_video/gt_public.mp4 + possessions.txt
    python analysis/gt_video.py --overlay    # also draws the tracked players and ball on the video

Under every video frame a panel shows:

  the label of this frame     H (home in possession), A (away), D (dead ball), big and coloured
  where we are                which possession of the 17, how long it lasts, the next change
  the next / last 10 seconds  a strip of labels centred on the current frame, one tick a second
  the whole window            the five minutes as one strip, with a cursor
  the tracking from above     players and ball as the automatic tracking has them, so you can see
                              what the data shows -- and what it misses -- when the label changes

With imageio-ffmpeg installed the video is re-encoded as H.264, so Windows' own player opens it;
without it, open it with VLC.

possessions.txt lists every labelled stretch and every change with its video clock, to jump to
it in a player (VLC: Ctrl+T).

Allowed by the rules: "inspecting the provided data and the video to better understand what counts
as possession". It shows the GROUND TRUTH only, never a model's output -- checking our output
against the video is not allowed. Everything it writes is derived from the gated data and stays in
analysis_out/ (gitignored): do not commit it, do not share it outside the team, do not upload it.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
import load_homography as lh  # noqa: E402
import possession_metrics as pm  # noqa: E402
import inspect_frame as inspect  # noqa: E402

# Params
FPS = inspect.FPS
OUT_DIR = os.path.join(ROOT, "analysis_out", "gt_video")
VIDEO_W, VIDEO_H = 1280, 720
PANEL_H = 300
MARGIN_FRAMES = 3 * FPS            # a little video before and after the labelled window
ZOOM_FRAMES = 10 * FPS             # the strip shows this much before and after the cursor
MAP_W, MAP_H = 330, 215
FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR
COLOUR = {pm.H: inspect.HOME_RGB[::-1], pm.A: inspect.AWAY_RGB[::-1], pm.D: (150, 150, 150),
          pm.X: (60, 60, 60)}
BALL = inspect.BALL_RGB[::-1]
BACKGROUND, INK, FAINT = (30, 30, 30), (240, 240, 240), (150, 150, 150)


def describe(code, names):
    return {pm.H: "H  {} in possesso".format(names[0]), pm.A: "A  {} in possesso".format(names[1]),
            pm.D: "D  palla morta", pm.X: "fuori dalla verita pubblica"}[code]


def clock(frame):
    seconds = frame / float(FPS)
    return "{}:{:05.2f}".format(int(seconds // 60), seconds % 60)


def text(image, value, origin, scale=0.55, colour=INK, thickness=1):
    cv2.putText(image, value, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def strip(panel, codes, first, last, x0, y0, width, height, cursor, ticks):
    """Labels of frames first..last as coloured blocks, a white cursor at `cursor`."""
    span = last - first
    columns = np.clip(((np.arange(width) / float(width)) * span + first).astype(int), 0, len(codes) - 1)
    for x, f in enumerate(columns):
        colour = COLOUR[codes[f]] if first <= f <= last else COLOUR[pm.X]
        panel[y0:y0 + height, x0 + x] = colour
    for f in ticks:
        x = x0 + int((f - first) / float(span) * width)
        if x0 <= x < x0 + width:
            panel[y0 + height - 6:y0 + height, x] = INK
    x = x0 + int((cursor - first) / float(span) * width)
    cv2.line(panel, (x, y0 - 4), (x, y0 + height + 4), (255, 255, 255), 2)


def minimap(ctx, frame):
    image = np.full((MAP_H, MAP_W, 3), (40, 90, 40), np.uint8)
    pad = 10
    w, h = MAP_W - 2 * pad, MAP_H - 2 * pad - 18
    flip_x, flip_y = inspect.orientation(ctx["H"][frame])

    def to_px(x, y):
        x = 1 - x if flip_x else x
        y = 1 - y if flip_y else y
        return int(pad + x * w), int(pad + 18 + y * h)

    cv2.rectangle(image, to_px(0, 0), to_px(1, 1), (220, 220, 220), 1)
    cv2.line(image, to_px(0.5, 0), to_px(0.5, 1), (220, 220, 220), 1)
    atd, team = ctx["atd"], ctx["team_of_slot"]
    px, py, bx, by = atd["px"][frame], atd["py"][frame], atd["bx"][frame], atd["by"][frame]
    for s in np.flatnonzero(np.isfinite(px) & np.isfinite(py)):
        colour = COLOUR[pm.H] if team[s] == bl.HOME else COLOUR[pm.A] if team[s] == bl.AWAY else FAINT
        cv2.circle(image, to_px(px[s], py[s]), 5, colour, -1, cv2.LINE_AA)
        cv2.circle(image, to_px(px[s], py[s]), 5, (0, 0, 0), 1, cv2.LINE_AA)
    if np.isfinite(bx) and np.isfinite(by):
        cv2.circle(image, to_px(bx, by), 4, BALL, -1, cv2.LINE_AA)
        cv2.circle(image, to_px(bx, by), 4, (0, 0, 0), 1, cv2.LINE_AA)
        text(image, "tracking: palla vista", (pad, 16), 0.45)
    else:
        text(image, "tracking: palla NON vista", (pad, 16), 0.45, (80, 200, 255))
    return image


def overlay(image, ctx, frame):
    """Tracked slots and ball on the video frame, through the homography (1920x1080 pixels)."""
    if not ctx["valid"][frame]:
        return
    matrix = ctx["H"][frame]
    atd, team = ctx["atd"], ctx["team_of_slot"]
    height, width = image.shape[:2]
    for s in np.flatnonzero(np.isfinite(atd["px"][frame]) & np.isfinite(atd["py"][frame])):
        u, v = lh.project(matrix, atd["px"][frame, s], atd["py"][frame, s])
        if 0 <= u < width and 0 <= v < height:
            colour = COLOUR[pm.H] if team[s] == bl.HOME else COLOUR[pm.A]
            cv2.circle(image, (int(u), int(v)), 14, colour, 3, cv2.LINE_AA)
    if np.isfinite(atd["bx"][frame]) and np.isfinite(atd["by"][frame]):
        u, v = lh.project(matrix, atd["bx"][frame], atd["by"][frame])
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(image, (int(u), int(v)), 10, BALL, 3, cv2.LINE_AA)


def segments_list(gt, first, last):
    """[(start, stop, code)] maximal runs of one label inside first..last (stop exclusive)."""
    window = gt[first:last + 1]
    edges = np.flatnonzero(np.diff(window) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(window)]))
    return [(first + a, first + b, int(window[a])) for a, b in zip(starts, stops)]


def write_list(path, segments, names):
    possession = 0
    lines = ["Verita pubblica, finestra {}-{} (video {} - {})".format(
        segments[0][0], segments[-1][1] - 1, clock(segments[0][0]), clock(segments[-1][1] - 1)),
        "H = {} in possesso, A = {} in possesso, D = palla morta".format(*names), "",
        "{:>4}  {:<5} {:>7} {:>7}  {:>10} {:>10}  {:>7}".format("#", "label", "start", "end", "video start",
                                                              "video end", "durata")]
    for start, stop, code in segments:
        tag = ""
        if code in (pm.H, pm.A):
            possession += 1
            tag = "poss {}".format(possession)
        lines.append("{:>4}  {:<5} {:>7} {:>7}  {:>10} {:>10}  {:>6.1f}s  {}".format(
            len(lines) - 3, pm.LABELS[code], start, stop - 1, clock(start), clock(stop - 1),
            (stop - start) / float(FPS), tag))
    lines += ["", "Cambi (primo frame della nuova etichetta):"]
    for (a, _, before), (b, _, after) in zip(segments[:-1], segments[1:]):
        lines.append("  video {:>9}  frame {:>6}   {} -> {}".format(clock(b), b, pm.LABELS[before], pm.LABELS[after]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overlay", action="store_true", help="draw the tracked slots and ball on the video")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    ctx = inspect.load_context(args.data_dir)
    gt = ctx["gt"]
    labelled = np.flatnonzero(gt != pm.X)
    first, last = int(labelled[0]), int(labelled[-1])
    segments = segments_list(gt, first, last)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_list(os.path.join(OUT_DIR, "possessions.txt"), segments, ctx["names"])

    possession_of = np.zeros(len(gt), dtype=int)
    count = 0
    for start, stop, code in segments:
        if code in (pm.H, pm.A):
            count += 1
            possession_of[start:stop] = count
    n_possessions = count
    segment_of = {f: (start, stop, code) for start, stop, code in segments for f in (start,)}
    change_frames = [s for s, _, _ in segments[1:]]

    video = games.require(["video"], games.GAME, args.data_dir)[0]
    capture = cv2.VideoCapture(video)
    begin, end = max(first - MARGIN_FRAMES, 0), min(last + MARGIN_FRAMES, len(gt) - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(begin - inspect.SEEK_BACK, 0))
    for _ in range(begin - max(begin - inspect.SEEK_BACK, 0)):
        capture.read()
    path = os.path.join(OUT_DIR, "gt_public.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VIDEO_W, VIDEO_H + PANEL_H))
    second_ticks = list(range(first - ZOOM_FRAMES * 2, last + ZOOM_FRAMES * 2, FPS))
    half_minute_ticks = list(range(first, last + 1, 30 * FPS))
    current = None

    for frame in range(begin, end + 1):
        ok, image = capture.read()
        if not ok:
            break
        if args.overlay:
            overlay(image, ctx, frame)
        image = cv2.resize(image, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
        panel = np.full((PANEL_H, VIDEO_W, 3), BACKGROUND, np.uint8)
        code = int(gt[frame])
        if frame in segment_of:
            current = segment_of[frame]

        # this frame
        cv2.rectangle(panel, (14, 14), (520, 66), COLOUR[code], -1)
        text(panel, describe(code, ctx["names"]), (26, 50), 0.85, (255, 255, 255), 2)
        text(panel, "video {}   frame {}".format(clock(frame), frame), (540, 34), 0.6)
        if first <= frame <= last and current is not None:
            start, stop, _ = current
            where = "possesso {} di {}".format(possession_of[frame], n_possessions) if possession_of[frame] else "interruzione"
            text(panel, "{}   ({:.1f} s su {:.1f} s)".format(where, (frame - start) / FPS, (stop - start) / FPS),
                 (540, 60), 0.6)
            upcoming = [c for c in change_frames if c > frame]
            if upcoming:
                nxt = upcoming[0]
                text(panel, "prossimo cambio tra {:.1f} s: {} -> {}".format(
                    (nxt - frame) / FPS, pm.LABELS[int(gt[nxt - 1])], pm.LABELS[int(gt[nxt])]), (540, 84), 0.55, FAINT)

        # the next and last 10 seconds
        text(panel, "10 s prima", (14, 110), 0.45, FAINT)
        text(panel, "ora", (14 + 440 - 12, 110), 0.45, FAINT)
        text(panel, "10 s dopo", (14 + 880 - 80, 110), 0.45, FAINT)
        strip(panel, gt, frame - ZOOM_FRAMES, frame + ZOOM_FRAMES, 14, 118, 880, 52, frame, second_ticks)
        for start, stop, c in segments:
            mid = (max(start, frame - ZOOM_FRAMES) + min(stop, frame + ZOOM_FRAMES)) / 2.0
            if stop - start > FPS and frame - ZOOM_FRAMES < mid < frame + ZOOM_FRAMES:
                x = 14 + int((mid - (frame - ZOOM_FRAMES)) / (2.0 * ZOOM_FRAMES) * 880)
                text(panel, pm.LABELS[c], (x - 7, 152), 0.65, (255, 255, 255), 2)

        # the whole window
        text(panel, "tutti i 5 minuti", (14, 200), 0.45, FAINT)
        strip(panel, gt, first, last, 14, 208, 880, 30, frame, half_minute_ticks)
        for f in half_minute_ticks:
            x = 14 + int((f - first) / float(last - first) * 880)
            text(panel, clock(f).split(".")[0], (x - 14, 256), 0.4, FAINT)
        text(panel, "blu H = {}   rosso A = {}   grigio D = palla morta".format(*ctx["names"]),
             (14, 286), 0.45, FAINT)

        # the tracking from above
        panel[PANEL_H - MAP_H - 10:PANEL_H - 10, VIDEO_W - MAP_W - 14:VIDEO_W - 14] = minimap(ctx, frame)

        writer.write(np.vstack([image, panel]))
        if (frame - begin) % 1500 == 0:
            print("  {} / {}".format(frame - begin, end - begin))
    writer.release()
    capture.release()
    path = to_h264(path)
    print("written {}\n        {}".format(path, os.path.join(OUT_DIR, "possessions.txt")))


def to_h264(path):
    """Re-encode OpenCV's mp4v as H.264, which Windows' own player and browsers can play.

    Needs imageio-ffmpeg (pip install imageio-ffmpeg); without it the mp4v file stays, and VLC
    plays that.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        print("  imageio-ffmpeg not installed: keeping mp4v, open it with VLC")
        return path
    import subprocess
    target = path.replace(".mp4", "_h264.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                    "-movflags", "+faststart", target], check=True)
    os.replace(target, path)
    return path


if __name__ == "__main__":
    main()
