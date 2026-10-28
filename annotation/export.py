"""Turn the hand annotations into ground-truth files the official scorer reads, and check how much
two annotators agree.

    python annotation/export.py                      # every annotator, first one alphabetically wins
    python annotation/export.py --primary fosco      # prefer fosco's version of a clip when there are two

Writes, in annotation/labels/out/:

  labels_tune.csv   frame,label over all 199319 frames: the annotated tune clips, X everywhere else
  labels_test.csv   the same for the test clips -- keep it for the final check only
  report.txt        coverage, possessions per set, and agreement between annotators

Score a submission against them exactly as against the public window:

    python evaluation/run_evaluation.py --gt annotation/labels/out/labels_test.csv
    python evaluation/run_evaluation.py --gt annotation/labels/out/labels_test.csv --bootstrap 1000

Agreement is the spellF1 of one annotator scored as if the other were the truth. It is the ceiling
of what any model can be shown to reach on these labels: if two people agree at 0.80, a model at
0.80 against one of them is as good as a person.
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import common  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames("tactical", GAME)
SETS = ("tune", "test")


def write_labels(path, labels):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def agreement(clip, first, second):
    """spellF1 / macroF1 of `second` scored against `first`, inside the clip only."""
    a = common.full_length(common.clip_labels(first, clip["start"], clip["end"]), clip["start"], N_FRAMES)
    b = common.full_length(common.clip_labels(second, clip["start"], clip["end"]), clip["start"], N_FRAMES)
    result = pm.evaluate(pm.label_codes(a), pm.label_codes(b))
    return result["spell_f1_mean"], result["macro_f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--primary", default=None, help="annotator whose version wins when a clip has several")
    parser.add_argument("--include-unfinished", action="store_true",
                        help="also export clips nobody has marked as completed")
    args = parser.parse_args()

    plan = common.load_plan()
    people = common.annotators()
    if args.primary:
        people = [args.primary] + [p for p in people if p != args.primary]
    lines = ["annotators: {}".format(", ".join(people) or "none")]

    out = {name: np.full(N_FRAMES, "X", dtype="<U1") for name in SETS}
    used = {name: [] for name in SETS}
    for clip in plan["clips"]:
        versions = []
        for person in people:
            document = common.load_annotation(person, clip)
            if document["changes"] and (document["done"] or args.include_unfinished):
                versions.append((person, document))
        if len(versions) >= 2:
            for (p1, d1), (p2, d2) in itertools.combinations(versions, 2):
                spell, macro = agreement(clip, d1["changes"], d2["changes"])
                lines.append("agreement {:<6} {} vs {}: spellF1 {:.3f}  macroF1 {:.3f}".format(
                    clip["id"], p1, p2, spell, macro))
        if clip["set"] not in SETS:
            continue  # the calibration clip is the public window: it already has real labels
        if not versions:
            lines.append("missing  {:<6} ({}) -- nobody has finished it".format(clip["id"], clip["set"]))
            continue
        person, document = versions[0]
        labels = common.clip_labels(document["changes"], clip["start"], clip["end"])
        unlabelled = int((labels == "X").sum())
        if unlabelled:
            lines.append("warning  {:<6} {} frames before the first mark stay X".format(clip["id"], unlabelled))
        out[clip["set"]][clip["start"]:clip["end"] + 1] = labels
        used[clip["set"]].append((clip["id"], person))

    os.makedirs(common.OUT_DIR, exist_ok=True)
    for name in SETS:
        path = os.path.join(common.OUT_DIR, "labels_{}.csv".format(name))
        write_labels(path, out[name])
        codes = pm.label_codes(out[name])
        graded = int((codes != pm.X).sum())
        spells = pm.possession_spells(codes, pm.valid_runs(codes)) if graded else []
        lines.append("{:<5} {:>5.1f} min from {} clip(s) [{}]  possessions {}  dead ball {:.0f}%  -> {}".format(
            name, graded / (60.0 * FPS), len(used[name]),
            ", ".join("{}:{}".format(c, p) for c, p in used[name]), len(spells),
            100.0 * (codes == pm.D).sum() / graded if graded else 0.0, os.path.relpath(path, ROOT)))

    report = "\n".join(lines)
    with open(os.path.join(common.OUT_DIR, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
