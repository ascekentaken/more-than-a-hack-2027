"""Experiment exp12: v11's stackers plus team-continuity signals (exp06.team_context).

    python external/exp12.py     # writes experiments/exp12_team/results.md, results.json, blend.json

Y / YC are exp10's X / XC with the extra signals; EY = mean of Y and YC, compared with v11 (exp11 EX).
Rule fixed in experiments/exp12_team/README.md.
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
OUT_DIR = os.path.join(ROOT, "experiments", "exp12_team")
SOURCES = ("exp12_Y", "exp12_YC")
exp06.EXTRA = exp06.EXTRA2 = True
exp08.PREFIX, exp08.VARIANTS, exp08.OUT_DIR = "exp12", {"Y": False, "YC": True}, OUT_DIR


def main():
    exp08.main()
    cv_cases = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        done = exp09.blend(SOURCES, "fold_" + held)
        for s in exp03.DEGRADED:
            cv_cases[s] += done[s]
    decoder = exp03.search(cv_cases[exp03.MATCHED[0]] + cv_cases[exp03.MATCHED[1]], True)
    _, cv_matched, cv_worst = exp09.block(cv_cases, decoder)
    _, holdout_matched, _ = exp09.block(exp09.blend(SOURCES, "holdout"), decoder)
    with open(os.path.join(ROOT, "experiments", "exp11_pair", "results.json"), encoding="utf-8") as handle:
        ex = json.load(handle)["EX"]
    adopt = (cv_matched >= ex["cv_matched"] + MIN_GAIN and cv_worst >= ex["cv_worst"] - MIN_GAIN
             and holdout_matched >= ex["holdout_matched"])
    out = {"adopt": adopt, "EY": {"sources": SOURCES, "decoder": decoder, "cv_matched": cv_matched,
                                  "cv_worst": cv_worst, "holdout_matched": holdout_matched},
           "v11": {k: ex[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder")}}
    with open(os.path.join(OUT_DIR, "blend.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    print("EY CV matched {:.4f} worst {:.4f} holdout {:.4f} -> {}".format(
        cv_matched, cv_worst, holdout_matched, "adopt EY" if adopt else "keep v11"), flush=True)
    print("v11 CV matched {:.4f} worst {:.4f} holdout {:.4f}".format(ex["cv_matched"], ex["cv_worst"], ex["holdout_matched"]))


if __name__ == "__main__":
    main()
