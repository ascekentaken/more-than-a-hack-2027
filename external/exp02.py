"""Experiment exp02: a learned frame model (v2) and a simulator matched to the challenge tracking.

    python external/exp02.py                 # the whole experiment, writes experiments/exp02_v2/
    python external/exp02.py --skip-v1-retune

Everything here runs on the open DFL matches only (external data). The challenge match enters in
one way: external/challenge_profile.py measures its tracking INPUT (when the ball goes missing,
how long the gaps last, how noisy the ball is) so that external/degrade.py's "matched" scenarios
look like it. Its ground truth, its video and hand labels are never read.

Protocol, fixed before any result was seen:

  1. TUNE matches (5): leave-one-match-out. For each fold LightGBM is trained on 4 matches (both
     matched scenarios, each also seen from the other side so home and away are interchangeable)
     and predicts the fifth. These out-of-fold probabilities pick the decoder's prices.
  2. HOLDOUT matches (2), never used for any choice: a model trained on all 5 TUNE matches and the
     prices of step 1. Compared with the official baseline, v1 as submitted (params_v1.json) and
     v1 re-tuned on the matched scenarios -- so the gain of the new simulator and the gain of the
     new model can be told apart.
  3. Decision rule: the method with the higher mean holdout spellF1 over the two matched scenarios
     is the one to submit. The old scenarios and the clean data are reported as robustness checks.
  4. The final v2 is retrained on all 7 matches with the same settings (model/v2.py uses it).
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import lightgbm as lgb  # noqa: E402
import possession_metrics as pm  # noqa: E402
import baseline as bl  # noqa: E402
import degrade  # noqa: E402
import dfl  # noqa: E402
import features_v2  # noqa: E402
import tune  # noqa: E402
import v1  # noqa: E402

# Params
TUNE, HOLDOUT, SEED = tune.TUNE, tune.HOLDOUT, tune.SEED
TRAIN_SCENARIOS = degrade.MATCHED
EVAL_SCENARIOS = degrade.MATCHED + degrade.SCENARIOS + ("clean",)
STRIDE = 3  # train on every third frame: neighbours are near copies
LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
              "min_child_samples": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": SEED, "num_threads": 0}
N_TREES = 400
DECODER_START = {"switch_team": 10.0, "enter_dead": 10.0, "exit_dead": 10.0, "dead_bias": 0.0}
DECODER_GRID = {
    "switch_team": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60],
    "enter_dead": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60],
    "exit_dead": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40, 60],
    "dead_bias": [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5],
}
DECODER_ROUNDS = 4
OUT_DIR = os.path.join(ROOT, "experiments", "exp02_v2")
MODEL_PATH = os.path.join(ROOT, "model", "v2_lgb.txt")
PARAMS_PATH = os.path.join(ROOT, "model", "params_v2.json")
CACHE_DIR = dfl.CACHE_DIR

try:
    from numba import njit
except ImportError:
    def njit(function=None, **_):
        return function if function is not None else (lambda f: f)


# -- data --------------------------------------------------------------------------------------

def item(match_id, scenario, swap=False):
    """(v2 feature table, truth codes, runs) of an external match; swap = seen from the other side."""
    path = os.path.join(CACHE_DIR, "v2_{}_{}{}.npz".format(match_id, scenario, "_swap" if swap else ""))
    if not os.path.isfile(path):
        data = dfl.load(match_id)
        if scenario != "clean":
            data = degrade.degrade(data, scenario, SEED + sorted(dfl.MATCHES).index(match_id))
        team = 1 - data["team"] if swap else data["team"]
        columns = features_v2.compute(data["px"], data["py"], data["ps"], data["bx"], data["by"],
                                      team, data["half"], float(data["length"]), float(data["width"]))
        label = data["label"].copy()
        if swap:
            label = np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)
        columns["truth"] = np.where(data["half"] > 0, label, pm.X).astype(np.int8)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        table = pd.DataFrame({k: saved[k] for k in saved.files if k != "truth"})
        truth = saved["truth"]
    runs = [(a, b + 1) for a, b in v1.blocks(table["half"].values)]
    return table, truth, runs


def training_rows(matches, scenarios):
    xs, ys = [], []
    for m in matches:
        for s in scenarios:
            for swap in (False, True):
                table, truth, _ = item(m, s, swap)
                keep = np.flatnonzero(truth != pm.X)[::STRIDE]
                xs.append(table.iloc[keep][features_v2.feature_names(table)])
                ys.append(truth[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(matches, scenarios=TRAIN_SCENARIOS, n_trees=N_TREES):
    x, y = training_rows(matches, scenarios)
    return lgb.train(LGB_PARAMS, lgb.Dataset(x, y), num_boost_round=n_trees)


def probabilities(model, table, table_swap=None):
    """(n, 3) H / A / D probabilities; with the swapped table, averaged over both sides."""
    names = features_v2.feature_names(table)
    p = model.predict(table[names])
    if table_swap is not None:
        q = model.predict(table_swap[names])
        p = (p + q[:, [1, 0, 2]]) / 2
    return np.clip(p, 1e-4, 1.0)


# -- decoder -----------------------------------------------------------------------------------

@njit(cache=True)
def viterbi3(log_e, cost):
    """Most likely H/A/D sequence under a 3x3 price matrix (row = from, column = to)."""
    n = log_e.shape[0]
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_e[0].copy()
    new = np.empty(3)
    for t in range(1, n):
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                value = score[i] - cost[i, j]
                if value > best:
                    best, arg = value, i
            new[j] = best + log_e[t, j]
            back[t, j] = arg
        score[:] = new
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(prob, half, d):
    log_e = np.log(prob)
    log_e[:, 2] += d["dead_bias"]
    cost = np.array([[0.0, d["switch_team"], d["enter_dead"]],
                     [d["switch_team"], 0.0, d["enter_dead"]],
                     [d["exit_dead"], d["exit_dead"], 0.0]])
    codes = np.full(len(prob), pm.X, dtype=np.int8)
    for first, last in v1.blocks(half):
        codes[first:last + 1] = viterbi3(np.ascontiguousarray(log_e[first:last + 1]), cost)
    return codes


# -- scoring -----------------------------------------------------------------------------------

def metrics(codes, truth, runs):
    spells = pm.spell_scores(truth, codes, runs)
    trans = pm.transition_scores(pm.transitions(truth, runs), pm.transitions(codes, runs), 25)
    return {"spellF1": spells["mean_f1"],
            "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(truth, codes))),
            "transF1": trans["f1"], "n_pred": spells["n_pred"], "n_gt": spells["n_gt"],
            **{"spell@{}".format(k): v["f1"] for k, v in spells["per_iou"].items()}}


def mean_metrics(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("n_pred", "n_gt")}
    out["pred/true"] = sum(r["n_pred"] for r in rows) / max(sum(r["n_gt"] for r in rows), 1)
    return out


def decoder_search(cases, start=DECODER_START):
    """Coordinate search of the decoder prices on (prob, half, truth, runs) cases."""
    def score(d):
        return np.mean([metrics(decode(p, h, d), g, r)["spellF1"] for p, h, g, r in cases])
    best = dict(start)
    best_score = score(best)
    print("  decoder start spellF1 {:.4f}".format(best_score))
    for round_number in range(1, DECODER_ROUNDS + 1):
        changed = False
        for name, values in DECODER_GRID.items():
            for value in values:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: float(value)})
                s = score(trial)
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            print("  round {} {:<12} -> {:<6} spellF1 {:.4f}".format(round_number, name, best[name], best_score))
        if not changed:
            break
    return best, best_score


def baseline_codes(table, window):
    labels = bl.smooth(table["nearest_team"].values.astype(np.int8), window)
    codes = pm.label_codes(labels)
    codes[table["half"].values == 0] = pm.X
    return codes


# -- the experiment ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-v1-retune", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log = {"started": datetime.datetime.now().replace(microsecond=0).isoformat(),
           "ball_noise_m": degrade.ball_noise_m(), "lgb_params": LGB_PARAMS, "n_trees": N_TREES,
           "stride": STRIDE, "train_scenarios": TRAIN_SCENARIOS}
    t0 = time.time()

    print("features for every match and scenario (cached after the first run)")
    for m in sorted(dfl.MATCHES):
        for s in EVAL_SCENARIOS:
            for swap in ((False, True) if s in TRAIN_SCENARIOS or m in HOLDOUT else (False,)):
                item(m, s, swap)
    print("  ready in {:.0f} s".format(time.time() - t0))

    # 1. leave-one-match-out on TUNE
    print("\n1. leave-one-match-out on the TUNE matches")
    oof = {}
    importance = None
    for held in TUNE:
        started = time.time()
        model = fit([m for m in TUNE if m != held])
        gain = model.feature_importance("gain")
        importance = gain if importance is None else importance + gain
        for s in TRAIN_SCENARIOS:
            table, truth, runs = item(held, s)
            oof[(held, s)] = (probabilities(model, table, item(held, s, True)[0]), table["half"].values, truth, runs)
        print("  fold {} done in {:.0f} s".format(held, time.time() - started))
    names = features_v2.feature_names(item(TUNE[0], TRAIN_SCENARIOS[0])[0])
    log["feature_importance"] = sorted(zip(names, (importance / importance.sum()).round(4).tolist()),
                                       key=lambda kv: -kv[1])
    frame_acc = np.mean([(p.argmax(1)[g != pm.X] == g[g != pm.X]).mean() for p, _, g, _ in oof.values()])
    print("  out-of-fold frame accuracy (argmax, no decoder) {:.3f}".format(frame_acc))
    decoder, cv_score = decoder_search(list(oof.values()))
    log["decoder"], log["cv_spellF1"], log["cv_frame_accuracy"] = decoder, cv_score, float(frame_acc)
    cv_rows = {s: mean_metrics([metrics(decode(p, h, decoder), g, r) for (m, sc), (p, h, g, r) in oof.items() if sc == s])
               for s in TRAIN_SCENARIOS}
    log["cv_by_scenario"] = cv_rows

    # v1 re-tuned on the matched scenarios (same search as external/tune.py, other scenarios)
    v1_matched = None
    if not args.skip_v1_retune:
        print("\nv1 re-tuned on the matched scenarios")
        v1_items = [tune.item(m, s) for m in TUNE for s in TRAIN_SCENARIOS]
        v1_matched, v1_score = tune.search(v1_items, v1.load_params())
        log["v1_matched_params"], log["v1_matched_tune_spellF1"] = v1_matched, v1_score

    # 2. HOLDOUT
    print("\n2. HOLDOUT matches")
    model = fit(list(TUNE))
    methods = {
        "baseline smooth50": lambda t, ts: baseline_codes(t, 50),
        "baseline smooth125": lambda t, ts: baseline_codes(t, 125),
        "v1 submitted": lambda t, ts: v1.decode(t, v1.load_params()),
        "v2": lambda t, ts: decode(probabilities(model, t, ts), t["half"].values, decoder),
    }
    if v1_matched is not None:
        methods["v1 re-tuned (matched)"] = lambda t, ts: v1.decode(t, v1_matched)
    holdout = {}
    for s in EVAL_SCENARIOS:
        holdout[s] = {}
        cases = [(item(m, s), item(m, s, True)[0]) for m in HOLDOUT]
        for name, method in methods.items():
            holdout[s][name] = mean_metrics([metrics(method(t, ts), g, r) for (t, g, r), ts in cases])
    log["holdout"] = holdout
    choice = {name: np.mean([holdout[s][name]["spellF1"] for s in TRAIN_SCENARIOS]) for name in methods}
    log["decision_score"] = choice
    log["decision"] = max(choice, key=choice.get)

    # 4. final model on all seven matches
    print("\n4. final v2 on all 7 matches")
    final = fit(sorted(dfl.MATCHES))
    final.save_model(MODEL_PATH)
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump({"decoder": decoder, "features": names, "lgb_params": LGB_PARAMS, "n_trees": N_TREES,
                   "trained_on": "open DFL / IDSSE matches {} (CC BY 4.0), scenarios {}, both sides".format(
                       ", ".join(sorted(dfl.MATCHES)), ", ".join(TRAIN_SCENARIOS)),
                   "decoder_tuned_on": "out-of-fold predictions on {}".format(", ".join(TUNE)),
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()}, handle, indent=1)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=1, default=float)
    write_tables(log)
    print("\nwritten {}, {}, {}".format(*(os.path.relpath(p, ROOT) for p in (
        os.path.join(OUT_DIR, "results.md"), MODEL_PATH, PARAMS_PATH))))


def write_tables(log):
    lines = ["# exp02 -- results (generated by external/exp02.py)", "",
             "Decision rule: higher mean HOLDOUT spellF1 over {} -> **{}**".format(
                 " and ".join(TRAIN_SCENARIOS), log["decision"]), ""]
    for name, value in sorted(log["decision_score"].items(), key=lambda kv: -kv[1]):
        lines.append("- {}: {:.3f}".format(name, value))
    lines += ["", "## HOLDOUT ({})".format(", ".join(HOLDOUT)), ""]
    for s, rows in log["holdout"].items():
        lines += ["### {}".format(s), "",
                  "| method | spellF1 | @0.3 | @0.5 | @0.7 | macroF1 | transF1 | pred/true |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                name, r["spellF1"], r["spell@0.3"], r["spell@0.5"], r["spell@0.7"], r["macroF1"],
                r["transF1"], r["pred/true"]))
        lines.append("")
    lines += ["## Cross-validation on TUNE (out of fold, v2)", "",
              "decoder: {}".format(log["decoder"]), "",
              "| scenario | spellF1 | macroF1 | transF1 | pred/true |", "|---|---|---|---|---|"]
    for s, r in log["cv_by_scenario"].items():
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(s, r["spellF1"], r["macroF1"], r["transF1"], r["pred/true"]))
    lines += ["", "## Most useful signals (share of LightGBM gain)", ""]
    lines += ["- {}: {:.1%}".format(n, v) for n, v in log["feature_importance"][:20]]
    with open(os.path.join(OUT_DIR, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
