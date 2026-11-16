"""Experiment exp08: v7's stacker with a longer view of the raw signals, for stoppages.

    python external/exp08.py            # writes experiments/exp08_long/results.md and results.json
    python external/exp08.py --final    # retrains the adopted stacker on all 7 DFL matches -> model/params_v8.json
    python external/exp08.py --final --force   # the same for L although not adopted (public 5-min look only)

Why. A stoppage shows in how LONG the players have been slow and the ball still, and play restarts
with a kick, not when a player reaches the ball. v7's stacker sees v4w's answers over +-2 s and the
raw signals at t only. exp06.long_context() adds, from the raw signals already in the tables:
frames the players have been / will stay slow (mean speed < 2 m/s), frames since / until the ball
is kicked (> 5 m/s), the ball still (< 1.5 m/s, seen), the ball still with a player within 1.5 m,
the ball unseen; means of player speed, ball seen and ball speed over the 5 and 10 s before and after.

Everything else is v7 (exp07 R6): first stage v4w frame + v4 change, the same nested folds and
first-stage answers (exp06's checkpoints), stacker weights (1/L)^0.5, decoder searched again; all
judged on DFL truth rewritten to controlled possession (external/exp07.py, 50 frames).

Variants:
  L   long-context stacker, learnt on the DFL truth as it is (as v7's stacker)
  LC  the same, learnt on the rewritten truth

Decision rule, fixed before the run: the better of L and LC by CV spellF1 (mean of the two matched
scenarios, rewritten truth) replaces v7 (exp07 R6) only if at least MIN_GAIN above it and its worst
case over the four scenarios is not more than MIN_GAIN below. HOLDOUT only reported.
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

import exp07  # noqa: E402  (also sets exp03 to the exp04 simulator; patches exp03.item -- undone below)
import exp03  # noqa: E402
import exp02  # noqa: E402
import exp05  # noqa: E402
import exp06  # noqa: E402

exp03.item = exp07._item          # raw truth by default; LC switches to the rewritten one to learn
exp06.LONG = True

# Params
MIN_GAIN = 0.005
PREFIX = "exp08"   # checkpoint prefix (exp10 reuses this module)
STACK_ALPHA = 0.5
VARIANTS = {"L": False, "LC": True}   # learn on rewritten truth?
OUT_DIR = os.path.join(ROOT, "experiments", "exp08_long")
BASE_PARAMS = os.path.join(ROOT, "model", "params_v7.json")
PARAMS_PATH = os.path.join(ROOT, "model", "params_v8.json")


def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "{}_{}.pkl".format(PREFIX, name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def fit_stacker(answers, names, rewritten):
    exp03.item = exp07.item if rewritten else exp07._item
    try:
        return exp06.stack_fit(answers, names, STACK_ALPHA)
    finally:
        exp03.item = exp07._item


def inner(held):
    """exp06's first-stage answers for the TUNE matches other than `held` (trained without both)."""
    out = {}
    for m in exp03.TUNE:
        if m != held:
            pair = sorted((held, m))
            done = exp05.load("exp06_pair_{}_{}".format(*pair))
            out.update({k: v for k, v in done.items() if k[0] == m})
    return out


def judged(model, names, matches, scenario, base):
    """v4w's cases restacked by `model`, truth rewritten to controlled possession."""
    return [(p, c, h, exp07.controlled(g), r) for p, c, h, g, r in exp06.restacked(model, names, matches, scenario, base)]


def block(by_scenario, decoder):
    s = {sc: exp03.summary(by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def main():
    started = time.time()
    names = exp03.names_for("v3")
    with open(os.path.join(exp07.OUT_DIR, "results.json"), encoding="utf-8") as handle:
        r6 = json.load(handle)["results"]["R6"]
    results = {}
    for v, rewritten in VARIANTS.items():
        print("\nvariant", v, flush=True)
        cv_cases = {s: [] for s in exp03.DEGRADED}
        for held in exp03.TUNE:
            base = exp05.load("exp04w_W05_fold_" + held)
            done = checkpoint("{}_fold_{}".format(v, held), lambda: (lambda model: {
                s: judged(model, names, [held], s, base[s]) for s in exp03.DEGRADED})(
                    fit_stacker(inner(held), names, rewritten)))
            for s in exp03.DEGRADED:
                cv_cases[s] += done[s]
            print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60), flush=True)
        print("  decoder search", v, flush=True)
        decoder = exp03.search(cv_cases[exp03.MATCHED[0]] + cv_cases[exp03.MATCHED[1]], True)
        cv, cv_matched, cv_worst = block(cv_cases, decoder)
        base = exp05.load("exp04w_W05_holdout")
        done = checkpoint(v + "_holdout", lambda: (lambda model: {
            s: judged(model, names, exp03.HOLDOUT, s, base[s]) for s in exp03.DEGRADED})(
                fit_stacker(exp06.loo(exp03.TUNE), names, rewritten)))
        holdout, holdout_matched, _ = block(done, decoder)
        results[v] = {"decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched,
                      "cv_worst": cv_worst, "holdout_matched": holdout_matched}
        print("  {} CV matched {:.4f} worst {:.4f} holdout {:.4f}".format(v, cv_matched, cv_worst, holdout_matched), flush=True)
    best = max(results, key=lambda v: results[v]["cv_matched"])
    adopt = (results[best]["cv_matched"] >= r6["cv_matched"] + MIN_GAIN
             and results[best]["cv_worst"] >= r6["cv_worst"] - MIN_GAIN)
    out = {"best": best, "adopt": adopt, "results": results,
           "v7": {k: r6[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder", "cv", "holdout")},
           "rule": "best of L/LC: cv_matched >= v7 + {0} and cv_worst >= v7 - {0} (controlled truth)".format(MIN_GAIN),
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {} |"
    lines = ["# exp08 -- v7's stacker with a longer view of the raw signals (generated by external/exp08.py)", "",
             "Rule: {} -> **{}**".format(out["rule"], "adopt " + best if adopt else "keep v7"), "",
             "| | CV matched | CV worst | holdout matched | holdout pred/true (matched_speed) | decoder |",
             "|---|---|---|---|---|---|",
             row.format("v7 (exp07 R6)", r6["cv_matched"], r6["cv_worst"], r6["holdout_matched"],
                        r6["holdout"]["matched_speed"]["pred/true"], r6["decoder"])]
    for v, r in results.items():
        lines.append(row.format(v, r["cv_matched"], r["cv_worst"], r["holdout_matched"],
                                r["holdout"]["matched_speed"]["pred/true"], r["decoder"]))
    for v, r in results.items():
        lines += ["", "## " + v, "",
                  "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout transF1 | holdout pred/true |",
                  "|---|---|---|---|---|---|---|"]
        for s in exp03.DEGRADED:
            c, h = r["cv"][s], r["holdout"][s]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["transF1"], h["pred/true"]))
    with open(os.path.join(OUT_DIR, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]), flush=True)


def final():
    """v7's first stage; the long-context stacker learns on leave-one-out answers for all 7 DFL matches (exp06's)."""
    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    if not out["adopt"] and "--force" not in sys.argv:
        print("exp08 kept v7; nothing to write")
        return
    v, r = out["best"], out["results"][out["best"]]
    answers = {}
    for m in exp03.all_matches()[:7]:
        answers.update(exp05.load("exp06_final_loo_" + m))
    names = exp03.names_for("v3")
    path = os.path.join(ROOT, "model", "v8_stack.txt")
    fit_stacker(answers, names, VARIANTS[v]).save_model(path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(BASE_PARAMS, encoding="utf-8") as handle:
        params = json.load(handle)
    params.update({"variant": params["variant"] + "; exp08 " + v + " (long context)",
                   "stack_model": os.path.relpath(path, ROOT).replace(os.sep, "/"), "stack_long": True,
                   "decoder": r["decoder"],
                   "decoder_tuned_on": "exp08: leave-one-out stacker answers on the TUNE matches, judged on controlled possession",
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
