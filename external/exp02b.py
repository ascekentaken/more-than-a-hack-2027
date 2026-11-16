"""Experiment exp02b: three variants of v2, chosen on cross-validation only.

    python external/exp02b.py                   # writes experiments/exp02_v2/variants.md and .json
    python external/exp02b.py --final mixed     # retrains that variant on all 7 matches for model/v2.py

exp02 left three open questions. Each is a variant here, on the same data and protocol as exp02:

  base        exp02's v2, with a wider decoder grid (exp02 stopped at the edge: enter_dead = 60)
  mixed       trained also on the old scenarios (independent, stoppage), so that it does not rely
              entirely on "missing ball means dead ball"
  relative    player speeds as a ratio to the match's median (features_v2.relative_speeds)
  mixed+rel   both

Selection rule, fixed before the run: the variant with the highest leave-one-match-out spellF1 on
the TUNE matches, averaged over the two matched scenarios; within 0.01 of the best, the one with
the higher worst-case score over the four degraded scenarios wins. The HOLDOUT matches are only
reported, never used to choose.
"""

import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402
import possession_metrics as pm  # noqa: E402
import degrade  # noqa: E402
import dfl  # noqa: E402
import exp02  # noqa: E402
import features_v2  # noqa: E402

# Params
DEGRADED = degrade.MATCHED + degrade.SCENARIOS
VARIANTS = {
    "base": {"train": degrade.MATCHED, "relative": False},
    "mixed": {"train": DEGRADED, "relative": False},
    "relative": {"train": degrade.MATCHED, "relative": True},
    "mixed+rel": {"train": DEGRADED, "relative": True},
}
WIDE = [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60, 90, 130, 200, 300]
exp02.DECODER_GRID.update({"switch_team": WIDE, "enter_dead": WIDE, "exit_dead": WIDE})
TIE = 0.01


def table_of(match_id, scenario, swap, relative):
    table, truth, runs = exp02.item(match_id, scenario, swap)
    return (features_v2.relative_speeds(table) if relative else table), truth, runs


def fit(matches, variant):
    xs, ys = [], []
    for m in matches:
        for s in variant["train"]:
            for swap in (False, True):
                table, truth, _ = table_of(m, s, swap, variant["relative"])
                keep = np.flatnonzero(truth != pm.X)[::exp02.STRIDE]
                xs.append(table.iloc[keep][features_v2.feature_names(table)])
                ys.append(truth[keep])
    x, y = pd.concat(xs, ignore_index=True), np.concatenate(ys)
    return lgb.train(exp02.LGB_PARAMS, lgb.Dataset(x, y), num_boost_round=exp02.N_TREES)


def cases(model, matches, scenario, relative):
    out = []
    for m in matches:
        table, truth, runs = table_of(m, scenario, False, relative)
        swapped = table_of(m, scenario, True, relative)[0]
        out.append((exp02.probabilities(model, table, swapped), table["half"].values, truth, runs))
    return out


def summary(case_list, decoder):
    return exp02.mean_metrics([exp02.metrics(exp02.decode(p, h, decoder), g, r) for p, h, g, r in case_list])


def final(name):
    """Retrain the chosen variant on all seven matches and write the files model/v2.py reads."""
    with open(os.path.join(exp02.OUT_DIR, "variants.json"), encoding="utf-8") as handle:
        result = json.load(handle)["results"][name]
    variant = result["variant"]
    model = fit(sorted(dfl.MATCHES), variant)
    model.save_model(exp02.MODEL_PATH)
    names = features_v2.feature_names(table_of(exp02.TUNE[0], degrade.MATCHED[0], False, variant["relative"])[0])
    with open(exp02.PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump({"variant": name, "decoder": result["decoder"], "relative_speeds": variant["relative"],
                   "features": names, "lgb_params": exp02.LGB_PARAMS, "n_trees": exp02.N_TREES,
                   "trained_on": "open DFL / IDSSE matches {} (CC BY 4.0), scenarios {}, both sides".format(
                       ", ".join(sorted(dfl.MATCHES)), ", ".join(variant["train"])),
                   "decoder_tuned_on": "out-of-fold predictions on {}, matched scenarios".format(", ".join(exp02.TUNE)),
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()}, handle, indent=1)
    print("written {} and {} ({})".format(exp02.MODEL_PATH, exp02.PARAMS_PATH, name))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--final":
        return final(sys.argv[2])
    started = time.time()
    results = {}
    for name, variant in VARIANTS.items():
        print("\n== {} ==".format(name))
        oof = {s: [] for s in DEGRADED}
        for held in exp02.TUNE:
            model = fit([m for m in exp02.TUNE if m != held], variant)
            for s in DEGRADED:
                oof[s] += cases(model, [held], s, variant["relative"])
            print("  fold {} ({:.0f} s so far)".format(held, time.time() - started))
        decoder, _ = exp02.decoder_search(oof[degrade.MATCHED[0]] + oof[degrade.MATCHED[1]])
        cv = {s: summary(oof[s], decoder) for s in DEGRADED}
        model = fit(list(exp02.TUNE), variant)
        holdout = {s: summary(cases(model, exp02.HOLDOUT, s, variant["relative"]), decoder) for s in DEGRADED}
        results[name] = {"variant": variant, "decoder": decoder, "cv": cv, "holdout": holdout,
                         "cv_matched": float(np.mean([cv[s]["spellF1"] for s in degrade.MATCHED])),
                         "cv_worst": float(min(cv[s]["spellF1"] for s in DEGRADED)),
                         "holdout_matched": float(np.mean([holdout[s]["spellF1"] for s in degrade.MATCHED]))}
        print("  CV matched {:.3f}  CV worst {:.3f}  holdout matched {:.3f}".format(
            results[name]["cv_matched"], results[name]["cv_worst"], results[name]["holdout_matched"]))

    best = max(r["cv_matched"] for r in results.values())
    close = [n for n, r in results.items() if r["cv_matched"] >= best - TIE]
    chosen = max(close, key=lambda n: results[n]["cv_worst"])
    out = {"chosen": chosen, "results": results, "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    with open(os.path.join(exp02.OUT_DIR, "variants.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=lambda v: list(v) if isinstance(v, tuple) else float(v))
    lines = ["# exp02b -- variants (generated by external/exp02b.py)", "",
             "Chosen on cross-validation (TUNE only): **{}**".format(chosen), "",
             "| variant | CV matched | CV worst | holdout matched | decoder |", "|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(n, r["cv_matched"], r["cv_worst"],
                                                                    r["holdout_matched"], r["decoder"]))
    for n, r in results.items():
        lines += ["", "## {}".format(n), "", "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout @0.7 | holdout transF1 | holdout pred/true |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in DEGRADED:
            c, h = r["cv"][s], r["holdout"][s]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["spell@0.7"], h["transF1"], h["pred/true"]))
    with open(os.path.join(exp02.OUT_DIR, "variants.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\nchosen: {}  ({:.0f} min)".format(chosen, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
