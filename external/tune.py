"""Tune model v1 on the open DFL matches, degraded to look like the challenge tracking.

    python external/tune.py            # search, then write model/params_v1.json
    python external/tune.py --report   # only score the current params, no search

The competition rules (organisers' clarification, Nov 2026): anything trained or tuned must be
tuned on external data, never on the challenge match, its ground truth, its video or hand labels.
So this script never opens the challenge data. It:

  1. reads the seven DFL matches (external/dfl.py) and degrades each one twice, once per scenario in
     external/degrade.py, with fixed seeds;
  2. turns each into the same feature table the challenge match gets (model/features.compute);
  3. searches the parameters of model/v1.py one at a time, several rounds, maximising spellF1 --
     the challenge's ranking metric, computed with the challenge's own scorer code -- averaged over
     five TUNE matches and both scenarios;
  4. reports the two HOLDOUT matches, never used by the search, to show whether the result
     generalises, next to the official baseline method and the untuned starting values.
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

import baseline as bl  # noqa: E402
import possession_metrics as pm  # noqa: E402
import dfl  # noqa: E402
import degrade  # noqa: E402
import features  # noqa: E402
import v1  # noqa: E402

# Params
TUNE = ("J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ")
HOLDOUT = ("J03WN1", "J03WR9")
SEED = 2027
ROUNDS = 8
GRID = {
    "switch_team": [3, 5, 7, 10, 14, 20, 30, 50, 80],
    "switch_dead": [10, 25, 35, 50, 70, 100, 200],
    "dead_speed_mid": [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "dead_speed_slope": [1.0, 1.5, 2.5, 3.5, 5.0, 8.0],
    "dead_ball_missing": [0.0, 0.35, 0.7, 1.2, 2.0, 3.0],
    "dead_ball_outside": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
    "dead_ball_fast": [0.0, -0.75, -1.5, -3.0, -5.0],
    "team_scale_m": [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5],
    "team_reach_m": [0.5, 1.0, 1.5, 2.5, 4.0],
    "team_decay_m": [0.5, 1.0, 2.5, 5.0, 10.0],
}
CACHE_DIR = dfl.CACHE_DIR


def item(match_id, scenario):
    """(feature table, ground-truth codes, runs) for one match, degraded or clean."""
    path = os.path.join(CACHE_DIR, "features_{}_{}.npz".format(match_id, scenario))
    if not os.path.isfile(path):
        data = dfl.load(match_id)
        if scenario != "clean":
            data = degrade.degrade(data, scenario, SEED + sorted(dfl.MATCHES).index(match_id))
        columns = features.compute(data["px"], data["py"], data["ps"], data["bx"], data["by"],
                                   data["team"], data["half"], float(data["length"]), float(data["width"]))
        columns["truth"] = np.where(data["half"] > 0, data["label"], pm.X).astype(np.int8)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        table = pd.DataFrame({k: saved[k] for k in saved.files if k != "truth"})
        truth = saved["truth"]
    runs = [(a, b + 1) for a, b in v1.blocks(table["half"].values)]
    return table, truth, runs


def evaluate(codes, truth, runs):
    spells = pm.spell_scores(truth, codes, runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(truth, codes)))
    return spells["mean_f1"], macro, spells["n_pred"], spells["n_gt"]


def score(params, items):
    """Mean spellF1 and macroF1 over items, plus the predicted / true possession ratio."""
    rows = [evaluate(v1.decode(table, params), truth, runs) for table, truth, runs in items]
    rows = np.array(rows, dtype=float)
    return rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].sum() / rows[:, 3].sum()


def baseline_codes(table):
    """The official baseline method (nearest player, majority over +/-50 frames) on a table."""
    labels = bl.smooth(table["nearest_team"].values.astype(np.int8), 50)
    codes = pm.label_codes(labels)
    codes[table["half"].values == 0] = pm.X
    return codes


def search(items, start):
    best = dict(start)
    best_score = score(best, items)[0]
    print("start: spellF1 {:.4f}".format(best_score))
    for round_number in range(1, ROUNDS + 1):
        changed = False
        for name, values in GRID.items():
            started = time.time()
            for value in values:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: value})
                s = score(trial, items)[0]
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            print("  round {} {:<18} -> {:<6} spellF1 {:.4f}  ({:.0f} s)".format(
                round_number, name, best[name], best_score, time.time() - started))
        if not changed:
            break
    return best, best_score


def report(params, label):
    print("\n{}".format(label))
    print("{:<9} {:<12} {:<10} {:>8} {:>8} {:>10}".format("set", "scenario", "method", "spellF1", "macroF1", "pred/true"))
    for set_name, matches in (("tune", TUNE), ("holdout", HOLDOUT)):
        for scenario in ("clean",) + degrade.SCENARIOS:
            items = [item(m, scenario) for m in matches]
            rows = []
            for method, function in (("baseline", lambda t: baseline_codes(t)),
                                     ("v1 start", lambda t: v1.decode(t, v1.DEFAULT_PARAMS)),
                                     ("v1 tuned", lambda t: v1.decode(t, params))):
                r = np.array([evaluate(function(t), g, r) for t, g, r in items], dtype=float)
                rows.append((method, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].sum() / r[:, 3].sum()))
            for method, s, m, ratio in rows:
                print("{:<9} {:<12} {:<10} {:>8.3f} {:>8.3f} {:>10.2f}".format(set_name, scenario, method, s, m, ratio))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="score the current params, no search")
    args = parser.parse_args()

    started = time.time()
    tune_items = [item(m, s) for m in TUNE for s in degrade.SCENARIOS]
    print("{} tuning items ready in {:.0f} s".format(len(tune_items), time.time() - started))

    if args.report:
        report(v1.load_params(), "current parameters")
        return

    best, best_score = search(tune_items, v1.DEFAULT_PARAMS)
    payload = {
        "params": best,
        "tuned_on": "open DFL / IDSSE matches {} (CC BY 4.0), degraded per external/degrade.py, "
                    "scenarios {}".format(", ".join(TUNE), ", ".join(degrade.SCENARIOS)),
        "objective": "mean spellF1 (evaluation/possession_metrics.py) over those matches and scenarios",
        "tune_spellF1": round(best_score, 4),
        "holdout_not_used_in_search": list(HOLDOUT),
        "written": datetime.datetime.now().replace(microsecond=0).isoformat(),
    }
    with open(v1.PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print("\nwritten {}".format(os.path.relpath(v1.PARAMS_PATH, ROOT)))
    report(best, "tuned parameters")


if __name__ == "__main__":
    main()
