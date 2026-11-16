"""Experiment exp04w: v4's frame model trained with every possession weighing alike, against short misses.

    python external/exp04w.py            # writes experiments/exp04_dead/weights.md and weights.json
    python external/exp04w.py --final    # retrains the chosen weighting on all 7 matches -> model/params_v4w.json

Why. On the German HOLDOUT matches v4 misses most possessions shorter than 2 s, and the frame model
is the main culprit: in 60% of those it never gives the right team more than 0.5
(analysis/dfl_errors.py). It learns frame by frame, so a 1-second interception is 25 frames among
140,000; spellF1 counts it as much as a one-minute possession.

What changes: each training frame gets the weight (1 / L) ** ALPHA, L the length in frames of the
true stretch (H, A or D) it belongs to, at least MIN_LEN, then normalised to mean 1. ALPHA = 1 makes
every stretch weigh the same in total, as in spellF1; ALPHA = 0.5 is halfway. Everything else is v4
(exp04 variant D): same tables, simulator, LightGBM parameters and seeds; the change model is not
retrained -- its predictions are read from exp04's checkpoints, so it is exactly v4's.

Variants: D (exp04's, ALPHA = 0, from results.json), W05 (ALPHA = 0.5), W10 (ALPHA = 1). Decoder
prices searched again for each, as in exp04.

Decision rule, fixed before the run: the W variant with the higher leave-one-match-out spellF1 on the
TUNE matches (mean of the two matched scenarios) replaces D only if that is at least MIN_GAIN above
D's and its worst case over the four degraded scenarios is not more than MIN_GAIN below D's. The DFL
HOLDOUT matches are only reported.
"""

import datetime
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp04  # noqa: E402,F401  (sets exp03 to the exp04 simulator and paths)
import exp03  # noqa: E402
import exp02  # noqa: E402
import lightgbm as lgb  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
ALPHAS = {"W05": 0.5, "W10": 1.0}
MIN_LEN = 5
MIN_GAIN = 0.005
OUT_DIR = exp03.OUT_DIR
PARAMS_PATH = os.path.join(ROOT, "model", "params_v4w.json")


def stretch_length(truth):
    """Length in frames of the run of equal labels each frame is in."""
    change = np.ones(len(truth), dtype=bool)
    change[1:] = truth[1:] != truth[:-1]
    start = np.flatnonzero(change)
    lengths = np.diff(np.append(start, len(truth)))
    return np.repeat(lengths, lengths)


def rows(matches, names, alpha):
    xs, ys, ws = [], [], []
    for m in matches:
        for s in exp03.DEGRADED:
            for swap in (False, True):
                table, truth, _ = exp03.item(m, s, swap)
                keep = np.flatnonzero(truth != pm.X)[::exp02.STRIDE]
                xs.append(table.iloc[keep][names].astype(np.float32))
                ys.append(truth[keep])
                ws.append(np.maximum(stretch_length(truth)[keep], MIN_LEN).astype(np.float64) ** -alpha)
    w = np.concatenate(ws)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), w / w.mean()


def fit(matches, names, alpha):
    x, y, w = rows(matches, names, alpha)
    return lgb.train(exp02.LGB_PARAMS, lgb.Dataset(x, y, weight=w), num_boost_round=exp02.N_TREES)


def load(name):
    with open(os.path.join(exp02.CACHE_DIR, "exp04_{}.pkl".format(name)), "rb") as handle:
        return pickle.load(handle)


def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "exp04w_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def reweighted(model, names, matches, scenario, v4_cases):
    """v4's cases with the frame probabilities of `model`; the change probabilities stay v4's."""
    out = []
    for m, (_, change, half, truth, runs) in zip(matches, v4_cases):
        table, swapped = exp03.item(m, scenario)[0], exp03.item(m, scenario, True)[0]
        out.append((exp03.predict_frame(model, names, table, swapped), change, half, truth, runs))
    return out


def block(by_scenario, decoder):
    s = {sc: exp03.summary(by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def main():
    started = time.time()
    names = exp03.names_for("v3")
    print("leave-one-match-out on TUNE")
    oof = {v: {s: [] for s in exp03.DEGRADED} for v in ALPHAS}
    for held in exp03.TUNE:
        v4 = load("fold_" + held)
        for v, alpha in ALPHAS.items():
            def fold():
                model = fit([m for m in exp03.TUNE if m != held], names, alpha)
                return {s: reweighted(model, names, [held], s, v4[s]["D"]) for s in exp03.DEGRADED}
            done = checkpoint("{}_fold_{}".format(v, held), fold)
            for s in exp03.DEGRADED:
                oof[v][s] += done[s]
        print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60))

    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        single = json.load(handle)["results"]["D"]
    v4_holdout = load("holdout")
    results = {}
    for v, alpha in ALPHAS.items():
        print("  decoder search", v)
        decoder = exp03.search(oof[v][exp03.MATCHED[0]] + oof[v][exp03.MATCHED[1]], True)
        cv, cv_matched, cv_worst = block(oof[v], decoder)
        done = checkpoint(v + "_holdout", lambda: (lambda model: {
            s: reweighted(model, names, exp03.HOLDOUT, s, v4_holdout[s]["D"]) for s in exp03.DEGRADED})(
                fit(list(exp03.TUNE), names, alpha)))
        holdout, holdout_matched, _ = block(done, decoder)
        results[v] = {"alpha": alpha, "decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched,
                      "cv_worst": cv_worst, "holdout_matched": holdout_matched}
    best = max(results, key=lambda v: results[v]["cv_matched"])
    adopt = (results[best]["cv_matched"] >= single["cv_matched"] + MIN_GAIN
             and results[best]["cv_worst"] >= single["cv_worst"] - MIN_GAIN)
    out = {"best": best, "adopt": adopt, "results": results,
           "single": {k: single[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder", "cv", "holdout")},
           "rule": "best W: cv_matched >= D + {0} and cv_worst >= D - {0}".format(MIN_GAIN),
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    with open(os.path.join(OUT_DIR, "weights.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {} |"
    lines = ["# exp04w -- every possession weighing alike (generated by external/exp04w.py)", "",
             "Rule: {} -> **{}**".format(out["rule"], "adopt " + best if adopt else "keep D"), "",
             "| | CV matched | CV worst | holdout matched | holdout pred/true (matched_speed) | decoder |",
             "|---|---|---|---|---|---|",
             row.format("D (exp04, no weights)", single["cv_matched"], single["cv_worst"], single["holdout_matched"],
                        single["holdout"]["matched_speed"]["pred/true"], single["decoder"])]
    for v, r in results.items():
        lines.append(row.format("{} (alpha {})".format(v, r["alpha"]), r["cv_matched"], r["cv_worst"],
                                r["holdout_matched"], r["holdout"]["matched_speed"]["pred/true"], r["decoder"]))
    for v, r in results.items():
        lines += ["", "## " + v, "",
                  "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout transF1 | holdout pred/true |",
                  "|---|---|---|---|---|---|---|"]
        for s in exp03.DEGRADED:
            c, h = r["cv"][s], r["holdout"][s]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["transF1"], h["pred/true"]))
    with open(os.path.join(OUT_DIR, "weights.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


def final():
    with open(os.path.join(OUT_DIR, "weights.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    r = out["results"][out["best"]]
    names = exp03.names_for("v3")
    path = os.path.join(ROOT, "model", "v4w_frame.txt")
    fit(exp03.all_matches()[:7], names, r["alpha"]).save_model(path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(exp03.PARAMS_PATH, encoding="utf-8") as handle:
        params = json.load(handle)
    params.update({"variant": "D: + touch signals, varying price; frames weighted (1/L)^{}".format(r["alpha"]),
                   "frame_model": os.path.relpath(path, ROOT).replace(os.sep, "/"), "decoder": r["decoder"],
                   "frame_weight_alpha": r["alpha"], "frame_weight_min_len": MIN_LEN,
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(path, ROOT), "and", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
