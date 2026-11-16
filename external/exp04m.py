"""Experiment exp04m: v4 (variant D of exp04) trained with two more external matches, from Metrica Sports.

    python external/exp04m.py            # writes experiments/exp04_dead/metrica.md and metrica.json
    python external/exp04m.py --final    # retrains on the 7 DFL + 2 Metrica matches -> model/params_v4m.json

Why. v4 learns from seven matches of one league and one data provider. external/metrica.py adds two
full matches from another provider (labels rebuilt from its event log), degraded by the same
simulator (new ball, realistic dead ball, gaps per player-speed bin).

What is compared, on exactly the evaluation of exp04 (leave-one-DFL-match-out on TUNE, draw 0):

  D    exp04's variant D as it is (results.json): trained on the other four TUNE matches
  DM   the same, with MET1 and MET2 added to the training data of every fold; decoder searched again

Decision rule, fixed before the run: DM replaces D only if its leave-one-match-out spellF1 on the
TUNE matches (mean of the two matched scenarios) is at least MIN_GAIN higher and its worst case over
the four degraded scenarios is not more than MIN_GAIN lower. The DFL HOLDOUT matches are only reported.

Also reported, not used to decide: how a model trained on DFL only does on the Metrica matches (does
v4 carry over to another data source?), and how one trained on DFL plus the other Metrica match does.
"""

import datetime
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp04  # noqa: E402,F401  (sets exp03 to the exp04 simulator and paths)
import exp03  # noqa: E402
import exp02  # noqa: E402
import metrica  # noqa: E402

# Params
EXTRA = sorted(metrica.GAMES)
MIN_GAIN = 0.005
VARIANT = "D"
OUT_DIR = exp03.OUT_DIR
PARAMS_PATH = os.path.join(ROOT, "model", "params_v4m.json")


def train(matches, names):
    return exp03.fit(matches, names, "frame"), exp03.fit(matches, names, "change")


def cases(models, names, matches, scenario):
    frame_model, change_model = models
    out = []
    for m in matches:
        table, truth, runs = exp03.item(m, scenario)
        swapped = exp03.item(m, scenario, True)[0]
        out.append((exp03.predict_frame(frame_model, names, table, swapped),
                    exp03.predict_change(change_model, names, table, swapped), table["half"].values, truth, runs))
    return out


def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "exp04m_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def block(cases_by_scenario, decoder):
    s = {sc: exp03.summary(cases_by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def main():
    started = time.time()
    names = exp03.names_for("v3")
    print("tables for", ", ".join(EXTRA))
    for m in EXTRA:
        for s in exp03.DEGRADED:
            for swap in (False, True):
                exp03.item(m, s, swap)
    print("  ready in {:.0f} min".format((time.time() - started) / 60))

    print("\nleave-one-match-out on TUNE, Metrica added to every fold")
    oof = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        def fold():
            models = train([m for m in exp03.TUNE if m != held] + EXTRA, names)
            return {s: cases(models, names, [held], s) for s in exp03.DEGRADED}
        done = checkpoint("fold_" + held, fold)
        for s in exp03.DEGRADED:
            oof[s] += done[s]
        print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60))
    decoder = exp03.search(oof[exp03.MATCHED[0]] + oof[exp03.MATCHED[1]], True)
    cv, cv_matched, cv_worst = block(oof, decoder)

    print("\nHOLDOUT")
    done = checkpoint("holdout", lambda: (lambda models: {s: cases(models, names, exp03.HOLDOUT, s) for s in exp03.DEGRADED})(
        train(list(exp03.TUNE) + EXTRA, names)))
    holdout, holdout_matched, _ = block(done, decoder)

    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        single = json.load(handle)["results"][VARIANT]

    print("\nMetrica, reported only")
    on_dfl = checkpoint("metrica_dfl_only", lambda: (lambda models: {s: cases(models, names, EXTRA, s) for s in exp03.DEGRADED})(
        train(list(exp03.TUNE), names)))
    other = {s: [] for s in exp03.DEGRADED}
    for held in EXTRA:
        done = checkpoint("metrica_" + held, lambda: (lambda models: {s: cases(models, names, [held], s) for s in exp03.DEGRADED})(
            train(list(exp03.TUNE) + [m for m in EXTRA if m != held], names)))
        for s in exp03.DEGRADED:
            other[s] += done[s]
    met_d, met_d_matched, _ = block(on_dfl, single["decoder"])
    met_dm, met_dm_matched, _ = block(other, decoder)

    result = {"decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched, "cv_worst": cv_worst,
              "holdout_matched": holdout_matched}
    adopt = cv_matched >= single["cv_matched"] + MIN_GAIN and cv_worst >= single["cv_worst"] - MIN_GAIN
    out = {"with_metrica": result, "single": {k: single[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder")},
           "adopt": adopt, "rule": "cv_matched >= D + {0} and cv_worst >= D - {0}".format(MIN_GAIN),
           "metrica_report": {"trained_dfl_only": met_d, "trained_dfl_only_matched": met_d_matched,
                              "trained_dfl_plus_other_metrica": met_dm, "trained_dfl_plus_other_metrica_matched": met_dm_matched},
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    with open(os.path.join(OUT_DIR, "metrica.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.3f} | {:.3f} | {:.3f} | {} |"
    lines = ["# exp04m -- v4 with two Metrica matches added (generated by external/exp04m.py)", "",
             "Rule: {} -> **{}**".format(out["rule"], "adopt DM" if adopt else "keep D"), "",
             "| | CV matched | CV worst | holdout matched | decoder |", "|---|---|---|---|---|",
             row.format("D  (exp04, DFL only)", single["cv_matched"], single["cv_worst"], single["holdout_matched"], single["decoder"]),
             row.format("DM (DFL + Metrica)", cv_matched, cv_worst, holdout_matched, decoder), "",
             "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout transF1 | holdout pred/true |",
             "|---|---|---|---|---|---|---|"]
    for s in exp03.DEGRADED:
        c, h = cv[s], holdout[s]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["transF1"], h["pred/true"]))
    lines += ["", "## Metrica matches (reported only)", "",
              "| scenario | trained on DFL only: spellF1 | macroF1 | + other Metrica match: spellF1 | macroF1 |",
              "|---|---|---|---|---|"]
    for s in exp03.DEGRADED:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            s, met_d[s]["spellF1"], met_d[s]["macroF1"], met_dm[s]["spellF1"], met_dm[s]["macroF1"]))
    with open(os.path.join(OUT_DIR, "metrica.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


def final():
    with open(os.path.join(OUT_DIR, "metrica.json"), encoding="utf-8") as handle:
        result = json.load(handle)["with_metrica"]
    names = exp03.names_for("v3")
    matches = exp03.all_matches()
    files = {}
    for target in ("frame", "change"):
        path = os.path.join(ROOT, "model", "v4m_{}.txt".format(target))
        exp03.fit(matches, names, target).save_model(path + ".tmp")
        os.replace(path + ".tmp", path)
        files[target] = os.path.relpath(path, ROOT).replace(os.sep, "/")
        print("written", files[target])
    with open(exp03.PARAMS_PATH, encoding="utf-8") as handle:
        params = json.load(handle)
    params.update({"variant": "D: + touch signals, varying price; DFL + Metrica", "frame_model": files["frame"],
                   "change_model": files["change"], "decoder": result["decoder"],
                   "trained_on": params["trained_on"] + "; plus Metrica Sports sample games 1 and 2 "
                                 "(github.com/metrica-sports/sample-data), labels rebuilt from their event logs",
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
