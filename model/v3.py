"""Possession model v3 family (experiments exp03-exp07); the submitted file is v7 (model/params_v7.json).

    python model/v3.py --params model/params_v7.json --out evaluation/results/noesi.csv

How v7 decides:

  1. For every frame, a LightGBM model gives the probability of H, A and D from the 76 signals of
     model/features_v3.py: who was near the ball last and next, player speed, where the ball is,
     when it went missing, the touches (the ball turns or changes speed sharply next to a player)
     and how long one player keeps the ball. It was trained with every frame weighted
     (1 / length of its true run) ** 0.5, so short possessions count (exp04w). The match is read
     from both sides and the two answers are averaged, so the model has no preference for either team.
  2. A second LightGBM model gives, for every frame, the probability that possession passes directly
     from one team to the other right there.
  3. A third LightGBM model, the stacker (exp06, external/exp06.py), reads the first two models'
     answers over +-2 s (shifted values, window means, before/after differences) plus the 76
     signals, and gives new H / A / D probabilities.
  4. A Viterbi decoder turns them into one sequence per half. Changing team costs
     max(0, a - b * logit(p change)) -- in v7 b = 0, a flat high price, so possession changes only
     where the new team keeps the ball -- and entering and leaving dead ball have fixed prices.
     The prices were chosen on DFL labels rewritten to the challenge's "controlled possession"
     (exp07, external/exp07.py).

Other params files (params_v3/v4/v4w/v6.json) run the earlier members of the family.

Where the numbers come from. Every model and price is written by external/exp03.py ... exp07.py
from the seven open DFL matches only, degraded to look like the challenge tracking -- including a
ball that moves like the challenge's and a realistic dead ball, fitted on the tracking input only.
Nothing is fitted on the challenge match's ground truth, its video or hand labels. As in v1, frames
outside the halves are X and the second half is dead ball until its kickoff (FAQ.md).
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
import exp03  # noqa: E402
import features_v3  # noqa: E402
import v1  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def predict(table, table_swap, frame_models, change_models, params, stack_model=None):
    """Several models (an ensemble, exp04c): their probabilities are averaged. With a stacker
    (exp06), the frame probabilities are replaced by the stacker's, which reads them over +-2 s."""
    names = params["features"]
    prob = np.mean([exp03.predict_frame(m, names, table, table_swap) for m in frame_models], axis=0)
    change = (np.mean([exp03.predict_change(m, names, table, table_swap) for m in change_models], axis=0)
              if change_models else None)
    if stack_model is not None:
        import exp06
        exp06.LONG = bool(params.get("stack_long", False))  # exp08's longer context
        prob = exp06.stack_predict(stack_model, prob, change, table["half"].values, table, table_swap, names)
    if params.get("blend"):  # exp09: the mean of several stackers, each on its own first stage
        import exp06
        probs = []
        for member in params["blend"]:
            load = lambda key: lgb.Booster(model_file=os.path.join(ROOT, member[key]))
            first = exp03.predict_frame(load("frame_model"), names, table, table_swap)
            first_change = exp03.predict_change(load("change_model"), names, table, table_swap)
            exp06.LONG = bool(member.get("stack_long", False))
            exp06.EXTRA = bool(member.get("stack_extra", False))
            exp06.EXTRA2 = bool(member.get("stack_extra2", False))
            probs.append(exp06.stack_predict(load("stack_model"), first, first_change, table["half"].values,
                                             table, table_swap, names))
        prob = np.mean(probs, axis=0)
    before_kickoff = (table["half"].values == 2) & (table["frame"].values < v1.SECOND_HALF_KICKOFF)
    prob[before_kickoff] = [1e-4, 1e-4, 1.0]
    return np.array(pm.LABELS)[exp03.decode(prob, change, table["half"].values, params["decoder"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "noesi.csv"))
    parser.add_argument("--params", default=os.path.join(ROOT, "model", "params_v11.json"),
                        help="model/params_v11.json (the submission, exp11); params_v7.json (exp07); params_v3/v4/v4w/v6.json run "
                             "the earlier models of exp03/exp04/exp04w/exp06")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as handle:
        params = json.load(handle)
    frame_files = params.get("frame_models") or [params["frame_model"]]
    change_files = params.get("change_models") or ([params["change_model"]] if params["change_model"] else [])
    frame_models = [lgb.Booster(model_file=os.path.join(ROOT, f)) for f in frame_files]
    change_models = [lgb.Booster(model_file=os.path.join(ROOT, f)) for f in change_files]
    stack_model = lgb.Booster(model_file=os.path.join(ROOT, params["stack_model"])) if params.get("stack_model") else None
    table, table_swap = features_v3.challenge(), features_v3.challenge(swap=True)
    expected = features_v3.feature_names(table)
    if params["features_kind"] == "v2":
        import features_v2
        v2_names = set(features_v2.feature_names(features_v2.challenge()))
        expected = [n for n in expected if n in v2_names]
    if expected != params["features"]:
        raise SystemExit("feature list differs from the trained model: rerun external/exp03.py --final")
    labels = predict(table, table_swap, frame_models, change_models, params, stack_model)
    v1.write(args.out, labels)
    problem = pm.validate_submission(pm.read_label_csv(args.out), v1.N_FRAMES)
    print("written {}  ({})".format(os.path.relpath(args.out, ROOT), problem or "valid submission"))
    graded = labels != "X"
    print("graded frames: H {:.0%}  A {:.0%}  D {:.0%}   possessions {} (mean {:.1f} s)".format(
        *[(labels[graded] == k).mean() for k in ("H", "A", "D")], *v1.possessions(labels)))
    print("\nscore it:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
