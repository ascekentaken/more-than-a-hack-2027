"""Experiment exp10: stackers with new signals (exp06.extra_context), added to v9's average.

    python external/exp10.py     # writes experiments/exp10_extra/results.md, results.json, blend.json

X / XC are exp08's L / LC with the extra signals; E6 = exp09's E4 members + X + XC.
Rule fixed in experiments/exp10_extra/README.md.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp08  # noqa: E402
import exp09  # noqa: E402
import exp06  # noqa: E402
import exp03  # noqa: E402

MIN_GAIN = 0.005
OUT_DIR = os.path.join(ROOT, "experiments", "exp10_extra")
exp06.EXTRA = True
exp08.PREFIX, exp08.VARIANTS, exp08.OUT_DIR = "exp10", {"X": False, "XC": True}, OUT_DIR


def main():
    exp08.main()
    sources = tuple(exp09.VARIANTS["E4"]) + ("exp10_X", "exp10_XC")
    cv_cases = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        done = exp09.blend(sources, "fold_" + held)
        for s in exp03.DEGRADED:
            cv_cases[s] += done[s]
    decoder = exp03.search(cv_cases[exp03.MATCHED[0]] + cv_cases[exp03.MATCHED[1]], True)
    _, cv_matched, cv_worst = exp09.block(cv_cases, decoder)
    _, holdout_matched, _ = exp09.block(exp09.blend(sources, "holdout"), decoder)
    with open(os.path.join(exp09.OUT_DIR, "results.json"), encoding="utf-8") as handle:
        e4 = json.load(handle)["results"]["E4"]
    adopt = cv_matched >= e4["cv_matched"] + MIN_GAIN and cv_worst >= e4["cv_worst"] - MIN_GAIN
    out = {"adopt": adopt, "E6": {"sources": sources, "decoder": decoder, "cv_matched": cv_matched,
                                  "cv_worst": cv_worst, "holdout_matched": holdout_matched},
           "v9": {k: e4[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder")}}
    with open(os.path.join(OUT_DIR, "blend.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    print("E6 CV matched {:.4f} worst {:.4f} holdout {:.4f} -> {}".format(
        cv_matched, cv_worst, holdout_matched, "adopt E6" if adopt else "keep v9"), flush=True)
    print("v9 CV matched {:.4f} worst {:.4f} holdout {:.4f}".format(e4["cv_matched"], e4["cv_worst"], e4["holdout_matched"]))


if __name__ == "__main__":
    main()
