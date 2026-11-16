"""Experiment exp04c: an ensemble of v4 models (variant D of exp04), against the single v4.

    python external/exp04c.py            # writes experiments/exp04_dead/ensemble.md and ensemble.json
    python external/exp04c.py --final    # retrains the ensemble on all 7 matches -> model/params_v4e.json

The single v4 is one frame model and one change model, trained on one random draw of each degraded
scenario. Here K pairs are trained: pair k on draw k of the degradation (other ball gaps, player
drop-outs, stoppages) with LightGBM seed SEED + k. The frame probabilities are averaged, the change
probabilities are averaged, and the decoder prices are searched again (averaged probabilities are
smoother). Same matches, scenarios, simulator (new ball, realistic dead ball) and protocol as exp04;
evaluation always on draw 0, so the numbers compare directly with experiments/exp04_dead/results.md.

Decision rule, fixed before the run (as exp02c): the ensemble replaces the single v4 only if its
leave-one-match-out spellF1 on the TUNE matches (mean of the two matched scenarios) is at least
MIN_GAIN higher, and its worst case over the four degraded scenarios is not more than MIN_GAIN lower.
The HOLDOUT matches are only reported.
"""

import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp04  # noqa: E402,F401  (sets exp03 to the exp04 simulator and paths)
import exp03  # noqa: E402
import exp02  # noqa: E402
import dfl  # noqa: E402

# Params
K = 5
MIN_GAIN = 0.005
VARIANT = "D"
OUT_DIR = exp03.OUT_DIR
PARAMS_PATH = os.path.join(ROOT, "model", "params_v4e.json")


def train(matches, names):
    return [(exp03.fit(matches, names, "frame", k), exp03.fit(matches, names, "change", k)) for k in range(K)]


def cases(models, names, matches, scenario):
    out = []
    for m in matches:
        table, truth, runs = exp03.item(m, scenario)
        swapped = exp03.item(m, scenario, True)[0]
        frame = np.mean([exp03.predict_frame(f, names, table, swapped) for f, _ in models], axis=0)
        change = np.mean([exp03.predict_change(c, names, table, swapped) for _, c in models], axis=0)
        out.append((frame, change, table["half"].values, truth, runs))
    return out


def checkpoint(name, compute):
    import pickle
    path = os.path.join(exp02.CACHE_DIR, "exp04c_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def main():
    started = time.time()
    names = exp03.names_for("v3")
    print("tables for draws 1-{}".format(K - 1))
    for k in range(1, K):
        for m in sorted(dfl.MATCHES):
            for s in exp03.DEGRADED:
                for swap in (False, True):
                    exp03.item(m, s, swap, k)
    print("  ready in {:.0f} min".format((time.time() - started) / 60))

    print("\nleave-one-match-out on TUNE")
    oof = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        def fold():
            models = train([m for m in exp03.TUNE if m != held], names)
            return {s: cases(models, names, [held], s) for s in exp03.DEGRADED}
        done = checkpoint("fold_" + held, fold)
        for s in exp03.DEGRADED:
            oof[s] += done[s]
        print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60))
    decoder = exp03.search(oof[exp03.MATCHED[0]] + oof[exp03.MATCHED[1]], True)
    cv = {s: exp03.summary(oof[s], decoder) for s in exp03.DEGRADED}

    print("\nHOLDOUT")

    def holdout_cases():
        models = train(list(exp03.TUNE), names)
        return {s: cases(models, names, exp03.HOLDOUT, s) for s in exp03.DEGRADED}

    done = checkpoint("holdout", holdout_cases)
    holdout = {s: exp03.summary(done[s], decoder) for s in exp03.DEGRADED}

    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        single = json.load(handle)["results"][VARIANT]
    result = {"decoder": decoder, "cv": cv, "holdout": holdout,
              "cv_matched": float(np.mean([cv[s]["spellF1"] for s in exp03.MATCHED])),
              "cv_worst": float(min(cv[s]["spellF1"] for s in exp03.DEGRADED)),
              "holdout_matched": float(np.mean([holdout[s]["spellF1"] for s in exp03.MATCHED]))}
    adopt = (result["cv_matched"] >= single["cv_matched"] + MIN_GAIN
             and result["cv_worst"] >= single["cv_worst"] - MIN_GAIN)
    out = {"ensemble": result, "single": {k: single[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder")},
           "adopt": adopt, "rule": "cv_matched >= single + {0} and cv_worst >= single - {0}".format(MIN_GAIN),
           "k": K, "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    with open(os.path.join(OUT_DIR, "ensemble.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    lines = ["# exp04c -- ensemble of {} v4 models (generated by external/exp04c.py)".format(K), "",
             "Rule: {} -> **{}**".format(out["rule"], "adopt the ensemble" if adopt else "keep the single v4"), "",
             "| | CV matched | CV worst | holdout matched | decoder |", "|---|---|---|---|---|",
             "| single v4 (exp04 D) | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                 single["cv_matched"], single["cv_worst"], single["holdout_matched"], single["decoder"]),
             "| ensemble | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                 result["cv_matched"], result["cv_worst"], result["holdout_matched"], decoder), "",
             "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout @0.7 | holdout transF1 | holdout pred/true |",
             "|---|---|---|---|---|---|---|---|"]
    for s in exp03.DEGRADED:
        c, h = cv[s], holdout[s]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["spell@0.7"], h["transF1"], h["pred/true"]))
    with open(os.path.join(OUT_DIR, "ensemble.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


def final():
    """All 7 matches, K pairs, the decoder of the CV run: files for model/v3.py --params model/params_v4e.json."""
    with open(os.path.join(OUT_DIR, "ensemble.json"), encoding="utf-8") as handle:
        result = json.load(handle)["ensemble"]
    names = exp03.names_for("v3")
    frames, changes = [], []
    for k in range(K):
        for target, files in (("frame", frames), ("change", changes)):
            path = os.path.join(ROOT, "model", "v4e_{}_{}.txt".format(target, k))
            if not os.path.isfile(path):
                exp03.fit(sorted(dfl.MATCHES), names, target, k).save_model(path + ".tmp")
                os.replace(path + ".tmp", path)
            files.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
            print("written", files[-1])
    with open(exp03.PARAMS_PATH, encoding="utf-8") as handle:
        params = json.load(handle)
    params.update({"variant": "D: + touch signals, varying price; ensemble of {}".format(K),
                   "frame_model": None, "change_model": None, "frame_models": frames, "change_models": changes,
                   "decoder": result["decoder"], "lgb_seeds": [exp02.SEED + k for k in range(K)],
                   "trained_on": params["trained_on"] + ", degradation draws 0-{}".format(K - 1),
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
