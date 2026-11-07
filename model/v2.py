"""Possession model v2 (experiment exp02): a frame classifier learned on external data, then a decoder.

    python model/v2.py         # writes evaluation/results/v2_raw.csv for the challenge match

How it decides:

  1. For every frame, LightGBM gives the probability of H, A and D from the signals of
     model/features_v2.py (who was near the ball last and next, player speed before and after,
     where the ball is, when it went missing...). The match is read from both sides -- once as it
     is, once with home and away exchanged -- and the two answers are averaged, so the model has no
     preference for either team.
  2. A Viterbi decoder turns those probabilities into a sequence, paying one price to change team,
     one to enter dead ball and one to leave it.

Where the numbers come from. The classifier (model/v2_lgb.txt) and the decoder prices
(model/params_v2.json) are written by external/exp02.py from the seven open DFL matches only,
degraded to look like the challenge tracking. Nothing is fitted on the challenge match's ground
truth, its video or hand labels. As in v1, frames outside the halves are X and the second half is
dead ball until its kickoff (FAQ.md).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import lightgbm as lgb  # noqa: E402
import possession_metrics as pm  # noqa: E402
import exp02  # noqa: E402
import features_v2  # noqa: E402
import v1  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def predict(table, table_swap, model, params):
    prob = exp02.probabilities(model, table, table_swap)
    before_kickoff = (table["half"].values == 2) & (table["frame"].values < v1.SECOND_HALF_KICKOFF)
    prob[before_kickoff] = [1e-4, 1e-4, 1.0]
    return np.array(pm.LABELS)[exp02.decode(prob, table["half"].values, params["decoder"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "v2_raw.csv"))
    args = parser.parse_args()

    with open(exp02.PARAMS_PATH, encoding="utf-8") as handle:
        params = json.load(handle)
    model = lgb.Booster(model_file=exp02.MODEL_PATH)
    table, table_swap = features_v2.challenge(), features_v2.challenge(swap=True)
    if params.get("relative_speeds"):
        table, table_swap = features_v2.relative_speeds(table), features_v2.relative_speeds(table_swap)
    if features_v2.feature_names(table) != params["features"]:
        raise SystemExit("feature list differs from the trained model: rerun external/exp02.py")
    labels = predict(table, table_swap, model, params)
    v1.write(args.out, labels)
    problem = pm.validate_submission(pm.read_label_csv(args.out), v1.N_FRAMES)
    print("written {}  ({})".format(os.path.relpath(args.out, ROOT), problem or "valid submission"))
    graded = labels != "X"
    print("graded frames: H {:.0%}  A {:.0%}  D {:.0%}   possessions {} (mean {:.1f} s)".format(
        *[(labels[graded] == k).mean() for k in ("H", "A", "D")], *v1.possessions(labels)))
    print("\nscore it:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
