"""Experiment exp11: average only the two stackers with the new signals (exp10 X + XC).

    python external/exp11.py            # writes experiments/exp11_pair/results.json
    python external/exp11.py --final    # trains X and XC on all 7 DFL matches -> model/params_v11.json

Follow-up to exp10, decided after seeing that X and XC alone beat the six-stacker mean E6
(so it is not blind: see experiments/exp11_pair/README.md). Judged as always on DFL truth rewritten
to controlled possession; the reference is E6.
"""

import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp09  # noqa: E402
import exp07  # noqa: E402
import exp06  # noqa: E402
import exp05  # noqa: E402
import exp03  # noqa: E402

MIN_GAIN = 0.005
SOURCES = ("exp10_X", "exp10_XC")
OUT_DIR = os.path.join(ROOT, "experiments", "exp11_pair")
E6_PATH = os.path.join(ROOT, "experiments", "exp10_extra", "blend.json")


def main():
    cv_cases = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        done = exp09.blend(SOURCES, "fold_" + held)
        for s in exp03.DEGRADED:
            cv_cases[s] += done[s]
    decoder = exp03.search(cv_cases[exp03.MATCHED[0]] + cv_cases[exp03.MATCHED[1]], True)
    cv, cv_matched, cv_worst = exp09.block(cv_cases, decoder)
    holdout, holdout_matched, _ = exp09.block(exp09.blend(SOURCES, "holdout"), decoder)
    with open(E6_PATH, encoding="utf-8") as handle:
        e6 = json.load(handle)["E6"]
    adopt = cv_matched >= e6["cv_matched"] + MIN_GAIN and cv_worst >= e6["cv_worst"] - MIN_GAIN
    out = {"adopt": adopt, "EX": {"sources": SOURCES, "decoder": decoder, "cv_matched": cv_matched, "cv_worst": cv_worst,
                                  "holdout_matched": holdout_matched, "cv": cv, "holdout": holdout},
           "E6": {k: e6[k] for k in ("cv_matched", "cv_worst", "holdout_matched")}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    print("EX CV matched {:.4f} worst {:.4f} holdout {:.4f} -> {}".format(
        cv_matched, cv_worst, holdout_matched, "adopt EX" if adopt else "keep E6"))
    print("E6 CV matched {:.4f} worst {:.4f} holdout {:.4f}".format(e6["cv_matched"], e6["cv_worst"], e6["holdout_matched"]))
    print("decoder", decoder)


def final():
    """X and XC on v4w's leave-one-out answers for all 7 DFL matches (exp06's checkpoints)."""
    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    if not out["adopt"]:
        print("nothing to write")
        return
    names = exp03.names_for("v3")
    answers = {}
    for m in exp03.all_matches()[:7]:
        answers.update(exp05.load("exp06_final_loo_" + m))
    with open(os.path.join(ROOT, "model", "params_v7.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    v4w = {"frame_model": params["frame_model"], "change_model": params["change_model"],
           "stack_long": True, "stack_extra": True}
    exp06.LONG = exp06.EXTRA = True
    blend = []
    for tag, item in (("x", exp07._item), ("xc", exp07.item)):
        exp03.item = item
        blend.append(dict(v4w, stack_model=exp09.save(exp06.stack_fit(answers, names, exp07.STACK_ALPHA),
                                                      "v11_{}_stack.txt".format(tag))))
        print("stacker", tag, "written", flush=True)
    for key in ("stack_model", "stack_alpha"):
        params.pop(key, None)
    params.update({
        "variant": "v4w first stage; exp11 EX: mean of two stackers with long context and extra signals (exp10 X, XC)",
        "blend": blend, "decoder": out["EX"]["decoder"],
        "decoder_tuned_on": "exp11: mean of X and XC leave-one-out answers on the TUNE matches, judged on controlled possession",
        "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(os.path.join(ROOT, "model", "params_v11.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written model/params_v11.json", flush=True)


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
