"""Where v4 misses a change of possession, on the two German HOLDOUT matches (external data only).

    python analysis/dfl_errors.py        # analysis_out/dfl_errors/missed.mp4, missed.txt, summary on screen
    python analysis/dfl_errors.py --kinds short --per-kind 12 --slow 2 --name short_slow
                                         # only short possessions, half speed, both matches

The challenge rules forbid checking a model's output against the challenge video, and studying our
errors on the five public minutes to fix them would be tuning on the challenge's ground truth. The
German matches have no such limit: their truth is part of the external data. So the errors are
studied there, on the degraded version the model is evaluated on (exp04 simulator, draw 0).

A change in the truth "into" a possession (the frame where H or A starts) counts as taken if the
prediction also starts that same label within TAKEN_FRAMES (1 s) of it. A missed one is:

  short     the true possession that starts there lasts under SHORT_S seconds
  late      the prediction starts it, but between 1 and LATE_S seconds away
  missed    no such start within LATE_S seconds

The clips show, for a sample of each kind, the pitch from above (players as the degraded tracking has
them, the ball the model sees in yellow and the true ball as a grey ring, touches the model detects
as a flash), the truth and the prediction as strips over +-CLIP_S seconds, and the model's
probabilities for the current frame.

Uses the predictions saved by external/exp04.py (variant D = v4, decoder from its CV run), so it
needs that run's checkpoint in external_data/dfl/cache/exp04_holdout.pkl.
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "analysis", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import exp04  # noqa: E402,F401  (exp03 set to the exp04 simulator)
import exp03  # noqa: E402
import exp02  # noqa: E402
import degrade  # noqa: E402
import dfl  # noqa: E402
import possession_metrics as pm  # noqa: E402
import gt_video  # noqa: E402

# Params
FPS = 25
TAKEN_FRAMES = FPS
LATE_S = 5
SHORT_S = 2.0
CLIP_S = 6
PER_KIND = 5
SCENARIO = "matched_speed"
VARIANT = "D"
OUT_DIR = os.path.join(ROOT, "analysis_out", "dfl_errors")
W, H = 1280, 720
PITCH_H = 470
# BGR
COLOUR = {pm.H: (230, 140, 40), pm.A: (60, 60, 220), pm.D: (150, 150, 150), pm.X: (60, 60, 60)}
BALL = (0, 220, 255)
NAMES = {pm.H: "H casa", pm.A: "A ospiti", pm.D: "D palla morta"}
KIND_IT = {"short": "possesso breve (< 2 s) non visto", "late": "cambio preso in ritardo/anticipo (1-5 s)",
           "missed": "cambio non preso (oltre 5 s)"}


def starts(codes):
    """Frames where H or A starts, with the label that starts."""
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return [(int(t), int(codes[t])) for t in change if codes[t] in (pm.H, pm.A)]


def run_length(codes, t):
    e = t
    while e < len(codes) and codes[e] == codes[t]:
        e += 1
    return (e - t) / FPS


def classify(truth, pred):
    pred_starts = {}
    for t, code in starts(pred):
        pred_starts.setdefault(code, []).append(t)
    out = []
    for t, code in starts(truth):
        near = np.array(pred_starts.get(code, []))
        gap = np.min(np.abs(near - t)) if len(near) else np.inf
        if gap <= TAKEN_FRAMES:
            kind = "taken"
        elif run_length(truth, t) < SHORT_S:
            kind = "short"
        elif gap <= LATE_S * FPS:
            kind = "late"
        else:
            kind = "missed"
        out.append({"frame": t, "label": code, "kind": kind, "gap_s": float(gap / FPS) if np.isfinite(gap) else None,
                    "length_s": run_length(truth, t), "pred_here": int(pred[t]), "truth_before": int(truth[t - 1])})
    return out


def draw_pitch(image, data, f, touches, team):
    pad = 30
    w, h = W - 2 * pad, PITCH_H - 2 * pad

    def px(x, y):
        return int(pad + x * w), int(pad + (1 - y) * h)

    image[:PITCH_H] = (40, 95, 40)
    cv2.rectangle(image, px(0, 0), px(1, 1), (220, 220, 220), 1)
    cv2.line(image, px(0.5, 0), px(0.5, 1), (220, 220, 220), 1)
    for s in np.flatnonzero(np.isfinite(data["px"][f])):
        cv2.circle(image, px(data["px"][f, s], data["py"][f, s]), 8, COLOUR[pm.H if team[s] == 0 else pm.A], -1, cv2.LINE_AA)
    if np.isfinite(data["clean_bx"][f]):
        cv2.circle(image, px(data["clean_bx"][f], data["clean_by"][f]), 9, (200, 200, 200), 2, cv2.LINE_AA)
    if np.isfinite(data["bx"][f]):
        cv2.circle(image, px(data["bx"][f], data["by"][f]), 6, BALL, -1, cv2.LINE_AA)
        gt_video.text(image, "palla vista dal modello (gialla); vera = anello grigio", (pad, 20), 0.5)
    else:
        gt_video.text(image, "palla NON vista dal modello; vera = anello grigio", (pad, 20), 0.5, (80, 200, 255))
    for t, side in touches:
        if 0 <= f - t < 12 and np.isfinite(data["bx"][t]):
            cv2.circle(image, px(data["bx"][t], data["by"][t]), 18 + 2 * (f - t), COLOUR[side], 2, cv2.LINE_AA)


def strip(image, codes, first, last, y0, label, cursor):
    x0, width, height = 160, W - 200, 34
    gt_video.text(image, label, (20, y0 + 24), 0.6)
    columns = np.clip((np.arange(width) / width * (last - first) + first).astype(int), 0, len(codes) - 1)
    for x, f in enumerate(columns):
        image[y0:y0 + height, x0 + x] = COLOUR[codes[f]]
    for f in range(first - first % FPS, last, FPS):
        x = x0 + int((f - first) / (last - first) * width)
        if x0 <= x < x0 + width:
            image[y0 + height - 6:y0 + height, x] = (240, 240, 240)
    x = x0 + int((cursor - first) / (last - first) * width)
    cv2.line(image, (x, y0 - 4), (x, y0 + height + 4), (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="+", default=["short", "late", "missed"])
    parser.add_argument("--per-kind", type=int, default=PER_KIND)
    parser.add_argument("--slow", type=int, default=1, help="write every frame this many times")
    parser.add_argument("--name", default="missed")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(exp03.OUT_DIR, "results.json"), encoding="utf-8") as handle:
        decoder = json.load(handle)["results"][VARIANT]["decoder"]
    with open(os.path.join(exp02.CACHE_DIR, "exp04_holdout.pkl"), "rb") as handle:
        saved = pickle.load(handle)

    events, per_scenario = [], {}
    for scenario in exp03.MATCHED:
        counts = {}
        for m, (prob, change, half, truth, _) in zip(exp03.HOLDOUT, saved[scenario][VARIANT]):
            pred = exp03.decode(prob, change, half, decoder)
            for e in classify(truth, pred):
                counts[e["kind"]] = counts.get(e["kind"], 0) + 1
                if scenario == SCENARIO:
                    e.update(match=m)
                    events.append(e)
        per_scenario[scenario] = counts

    lines = ["Cambi di possesso (inizio di un H o di un A) sulle 2 partite tedesche di controllo, v4", ""]
    for scenario, counts in per_scenario.items():
        total = sum(counts.values())
        lines.append("{}: {} inizi di possesso".format(scenario, total))
        for kind in ("taken", "short", "late", "missed"):
            lines.append("  {:<7} {:5d}  {:5.1%}".format(kind, counts.get(kind, 0), counts.get(kind, 0) / total))
    lines.append("")

    # what the model says where it misses, and how much ball it sees there (scenario SCENARIO)
    data_by_match = {}
    for m in exp03.HOLDOUT:
        clean = dfl.load(m)
        degraded = degrade.degrade(clean, SCENARIO, exp03.SEED + exp03.all_matches().index(m), ball="drift", dead="real")
        degraded["clean_bx"], degraded["clean_by"] = clean["bx"], clean["by"]
        table = exp03.item(m, SCENARIO)[0]
        touch = np.flatnonzero(table["touch"].values == 1)
        degraded["touches"] = [(int(t), int(table["last_touch_team"].values[t])) for t in touch
                               if table["last_touch_team"].values[t] in (0, 1)]
        degraded["table"] = table
        data_by_match[m] = degraded
    for kind in ("taken", "short", "late", "missed"):
        chosen = [e for e in events if e["kind"] == kind]
        if not chosen:
            continue
        seen = [np.isfinite(data_by_match[e["match"]]["bx"][max(e["frame"] - FPS, 0):e["frame"] + FPS]).mean() for e in chosen]
        here = [e["pred_here"] for e in chosen]
        lines.append("{:<7} n={:4d}  palla vista entro 1 s: {:4.0%}   il modello li' dice: stessa squadra {:4.0%}, "
                     "altra squadra {:4.0%}, palla morta {:4.0%}   prima c'era palla morta: {:4.0%}".format(
                         kind, len(chosen), np.mean(seen),
                         np.mean([h == e["label"] for h, e in zip(here, chosen)]),
                         np.mean([h not in (e["label"], pm.D) for h, e in zip(here, chosen)]),
                         np.mean([h == pm.D for h in here]), np.mean([e["truth_before"] == pm.D for e in chosen])))
    lines.append("")
    lines.append("Clip (in ordine): partita, minuto, tipo, possesso vero che inizia, durata, scarto della previsione")
    rng = np.random.RandomState(0)
    clips = []
    for kind in args.kinds:
        for m in exp03.HOLDOUT:  # half from each match
            pool = [e for e in events if e["kind"] == kind and e["match"] == m]
            take = min((args.per_kind + 1) // 2 if m == exp03.HOLDOUT[0] else args.per_kind // 2, len(pool))
            for i in sorted(rng.choice(len(pool), take, replace=False)):
                clips.append(pool[i])
    for i, e in enumerate(clips, 1):
        lines.append("{:2d}. {} {:5.1f}' {:<7} {}  dura {:4.1f} s  scarto {}".format(
            i, e["match"], e["frame"] / FPS / 60, e["kind"], NAMES[e["label"]], e["length_s"],
            "{:.1f} s".format(e["gap_s"]) if e["gap_s"] is not None else "-"))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, args.name + ".txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    path = os.path.join(OUT_DIR, args.name + ".mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i, e in enumerate(clips, 1):
        m, t = e["match"], e["frame"]
        data = data_by_match[m]
        prob, change, half, truth, _ = saved[SCENARIO][VARIANT][exp03.HOLDOUT.index(m)]
        pred = exp03.decode(prob, change, half, decoder)
        first, last = max(t - CLIP_S * FPS, 0), min(t + CLIP_S * FPS, len(truth) - 1)
        title = np.zeros((H, W, 3), np.uint8)
        gt_video.text(title, "Clip {} di {}".format(i, len(clips)), (60, 250), 1.4, (240, 240, 240), 2)
        gt_video.text(title, KIND_IT[e["kind"]], (60, 320), 1.0, (80, 200, 255), 2)
        gt_video.text(title, "{}  minuto {:.1f}   inizia {} (dura {:.1f} s)".format(
            m, t / FPS / 60, NAMES[e["label"]], e["length_s"]), (60, 380), 0.8)
        for _ in range(FPS * 2):
            writer.write(title)
        for f in range(first, last + 1):
            image = np.full((H, W, 3), (30, 30, 30), np.uint8)
            draw_pitch(image, data, f, data["touches"], data["team"])
            strip(image, truth, first, last, PITCH_H + 20, "verita", f)
            strip(image, pred, first, last, PITCH_H + 70, "v4", f)
            p = prob[f]
            gt_video.text(image, "probabilita v4:  H {:.2f}   A {:.2f}   D {:.2f}    cambio di squadra qui {:.2f}".format(
                p[0], p[1], p[2], change[f]), (20, PITCH_H + 140), 0.6)
            gt_video.text(image, "{:+.1f} s dal cambio vero   ora: verita {}  /  v4 {}".format(
                (f - t) / FPS, NAMES.get(int(truth[f]), "X"), NAMES.get(int(pred[f]), "X")), (20, PITCH_H + 175), 0.6)
            gt_video.text(image, "clip {}/{}  {}{}".format(i, len(clips), KIND_IT[e["kind"]],
                                                         "   (rallentato x{})".format(args.slow) if args.slow > 1 else ""),
                          (20, PITCH_H + 210), 0.55, (80, 200, 255))
            for _ in range(args.slow):
                writer.write(image)
    writer.release()
    path = gt_video.to_h264(path)
    print("\nwritten", path)


if __name__ == "__main__":
    main()
