"""Experiment exp03: touch signals and a time-varying price for changing team, against short possessions.

    python external/exp03.py          # writes experiments/exp03_touch/results.md and results.json

Why. 30% of the DFL possessions last under 2 s and every possession counts the same in spellF1;
v2 merges many of them into their neighbours (it predicts 65-77% of the true possessions).

What changes, all on external data only:

  ball model   the external ball is made to move like the challenge ball (external/ball_model.py:
               a slowly drifting error, then smoothing), fitted on the challenge tracking INPUT.
               With exp02's independent noise the DFL ball made 31-44 touches a minute against
               9.3 on the challenge; with this one 7-11.
  touch        model/features_v3.py: touches (sharp turn or speed change of the ball next to a
               player), per-team time since / until a touch, touch counts, control runs.
  change cost  a second LightGBM model gives, per frame, the probability that possession passes
               directly from one team to the other within CHANGE_RADIUS frames; the decoder's price
               for changing team becomes max(0, a - b * logit(p)). b = 0 is v2's fixed price.

Variants, all trained like exp02b "mixed" (four degraded scenarios, both sides), all with the new
ball:

  A  v2 signals, fixed price          B  v2 + touch signals, fixed price
  C  v2 signals, varying price        D  v2 + touch signals, varying price
  S  the submitted v2 as it is: trained with exp02's ball, decoder as submitted

Protocol and rule, fixed before the run (as in exp02b): leave-one-match-out on the five TUNE
matches picks each variant's decoder and the winner; the winner must beat every other candidate's
CV spellF1 (mean of the two matched scenarios) by at least TIE, else among those within TIE the one
with the best worst case over the four degraded scenarios wins. The two HOLDOUT matches are only
reported.
"""

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
import degrade  # noqa: E402
import dfl  # noqa: E402
import exp02  # noqa: E402
import exp02b  # noqa: E402
import features_v2  # noqa: E402
import features_v3  # noqa: E402

try:
    from numba import njit
except ImportError:
    def njit(function=None, **_):
        return function if function is not None else (lambda f: f)

# Params
TUNE, HOLDOUT, SEED = exp02.TUNE, exp02.HOLDOUT, exp02.SEED
DEGRADED = exp02b.DEGRADED
MATCHED = degrade.MATCHED
BALL = "drift"
CHANGE_RADIUS = 6
TIE = 0.01
CHANGE_PARAMS = dict(exp02.LGB_PARAMS, objective="binary")
CHANGE_PARAMS.pop("num_class")
COST_GRID = {
    "team_a": [2, 3, 4, 6, 8, 10, 14, 20, 28, 40],
    "team_b": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "enter_dead": exp02b.WIDE,
    "exit_dead": exp02b.WIDE,
    "dead_bias": [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5],
}
FIXED = ("team_a", "enter_dead", "exit_dead", "dead_bias")
ROUNDS = 4
OUT_DIR = os.path.join(ROOT, "experiments", "exp03_touch")


# -- data --------------------------------------------------------------------------------------

def item(match_id, scenario, swap=False):
    """(v3 table, truth, runs) with the new ball model; same seeds as exp02, so same gaps."""
    path = os.path.join(exp02.CACHE_DIR, "v3_{}_{}{}.npz".format(match_id, scenario, "_swap" if swap else ""))
    if not os.path.isfile(path):
        data = dfl.load(match_id)
        if scenario != "clean":
            data = degrade.degrade(data, scenario, SEED + sorted(dfl.MATCHES).index(match_id), ball=BALL)
        team = 1 - data["team"] if swap else data["team"]
        columns = features_v3.compute(data["px"], data["py"], data["ps"], data["bx"], data["by"],
                                      team, data["half"], float(data["length"]), float(data["width"]))
        label = data["label"].copy()
        if swap:
            label = np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)
        columns["truth"] = np.where(data["half"] > 0, label, pm.X).astype(np.int8)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        table = pd.DataFrame({k: saved[k] for k in saved.files if k != "truth"})
        truth = saved["truth"]
    runs = [(a, b + 1) for a, b in exp02.v1.blocks(table["half"].values)]
    return table, truth, runs


def change_target(truth):
    """1 within CHANGE_RADIUS frames of a direct H <-> A change in the truth."""
    flip = np.zeros(len(truth), dtype=bool)
    a, b = truth[:-1], truth[1:]
    at = np.flatnonzero(((a == pm.H) & (b == pm.A)) | ((a == pm.A) & (b == pm.H))) + 1
    for t in at:
        flip[max(t - CHANGE_RADIUS, 0):t + CHANGE_RADIUS + 1] = True
    return flip.astype(np.int8)


def names_for(kind):
    table = item(TUNE[0], MATCHED[0])[0]
    v3 = features_v3.feature_names(table)
    if kind == "v3":
        return v3
    v2 = set(features_v2.feature_names(exp02.item(TUNE[0], MATCHED[0])[0]))
    return [n for n in v3 if n in v2]


def rows(matches, names, target):
    xs, ys = [], []
    for m in matches:
        for s in DEGRADED:
            for swap in (False, True):
                table, truth, _ = item(m, s, swap)
                keep = np.flatnonzero(truth != pm.X)[::exp02.STRIDE]
                xs.append(table.iloc[keep][names].astype(np.float32))
                ys.append(truth[keep] if target == "frame" else change_target(truth)[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(matches, names, target):
    x, y = rows(matches, names, target)
    params = exp02.LGB_PARAMS if target == "frame" else CHANGE_PARAMS
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=exp02.N_TREES)


def predict_frame(model, names, table, table_swap):
    p = model.predict(table[names].astype(np.float32))
    q = model.predict(table_swap[names].astype(np.float32))
    return np.clip((p + q[:, [1, 0, 2]]) / 2, 1e-4, 1.0)


def predict_change(model, names, table, table_swap):
    p = (model.predict(table[names].astype(np.float32)) + model.predict(table_swap[names].astype(np.float32))) / 2
    return np.clip(p, 1e-4, 1 - 1e-4)


# -- decoder with a price that varies in time --------------------------------------------------

@njit(cache=True)
def viterbi_tv(log_e, team_cost, enter_dead, exit_dead):
    """Most likely H/A/D sequence; changing team between t-1 and t costs team_cost[t]."""
    n = log_e.shape[0]
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_e[0].copy()
    new = np.empty(3)
    cost = np.zeros((3, 3))
    for t in range(1, n):
        cost[0, 1] = team_cost[t]
        cost[1, 0] = team_cost[t]
        cost[0, 2] = enter_dead
        cost[1, 2] = enter_dead
        cost[2, 0] = exit_dead
        cost[2, 1] = exit_dead
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


def decode(prob, change, half, d):
    log_e = np.log(prob)
    log_e[:, 2] += d["dead_bias"]
    if change is None or d.get("team_b", 0.0) == 0.0:
        team_cost = np.full(len(prob), float(d["team_a"]))
    else:
        team_cost = np.maximum(0.0, d["team_a"] - d["team_b"] * np.log(change / (1 - change)))
    codes = np.full(len(prob), pm.X, dtype=np.int8)
    for first, last in exp02.v1.blocks(half):
        codes[first:last + 1] = viterbi_tv(np.ascontiguousarray(log_e[first:last + 1]),
                                           np.ascontiguousarray(team_cost[first:last + 1]),
                                           float(d["enter_dead"]), float(d["exit_dead"]))
    return codes


def summary(cases, d):
    return exp02.mean_metrics([exp02.metrics(decode(p, c, h, d), g, r) for p, c, h, g, r in cases])


def search(cases, varying):
    grid = {k: v for k, v in COST_GRID.items() if varying or k in FIXED}
    best = {"team_a": 6.0, "team_b": 0.0, "enter_dead": 90.0, "exit_dead": 10.0, "dead_bias": 1.0}
    best_score = summary(cases, best)["spellF1"]
    print("    decoder start {:.4f}".format(best_score))
    for round_number in range(1, ROUNDS + 1):
        changed = False
        for name, values in grid.items():
            for value in values:
                if value == best[name]:
                    continue
                trial = dict(best, **{name: float(value)})
                s = summary(cases, trial)["spellF1"]
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            print("    round {} {:<11} -> {:<6} spellF1 {:.4f}".format(round_number, name, best[name], best_score))
        if not changed:
            break
    return best


# -- the experiment ----------------------------------------------------------------------------

VARIANTS = {"A": ("v2", False), "B": ("v3", False), "C": ("v2", True), "D": ("v3", True)}


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("tables (new ball) for every match, scenario and side")
    for m in sorted(dfl.MATCHES):
        for s in DEGRADED:
            for swap in (False, True):
                item(m, s, swap)
    print("  ready in {:.0f} min".format((time.time() - started) / 60))
    names = {"v2": names_for("v2"), "v3": names_for("v3")}

    def cases_for(models, matches, scenario):
        """{variant: [(prob, change, half, truth, runs)]} for the matches of one scenario."""
        out = {v: [] for v in list(VARIANTS) + ["S"]}
        for m in matches:
            table, truth, runs = item(m, scenario)
            swapped = item(m, scenario, True)[0]
            half = table["half"].values
            frame = {k: predict_frame(models["frame_" + k], names[k], table, swapped) for k in ("v2", "v3")}
            change = {k: predict_change(models["change_" + k], names[k], table, swapped) for k in ("v2", "v3")}
            for v, (kind, varying) in VARIANTS.items():
                out[v].append((frame[kind], change[kind] if varying else None, half, truth, runs))
            out["S"].append((predict_frame(models["old"], names["v2"], table, swapped), None, half, truth, runs))
        return out

    def train_all(matches):
        models = {}
        for kind in ("v2", "v3"):
            models["frame_" + kind] = fit(matches, names[kind], "frame")
            models["change_" + kind] = fit(matches, names[kind], "change")
        models["old"] = exp02b.fit(matches, exp02b.VARIANTS["mixed"])  # exp02's ball
        return models

    print("\nleave-one-match-out on TUNE")
    oof = {s: {v: [] for v in list(VARIANTS) + ["S"]} for s in DEGRADED}
    for held in TUNE:
        models = train_all([m for m in TUNE if m != held])
        for s in DEGRADED:
            for v, cases in cases_for(models, [held], s).items():
                oof[s][v] += cases
        print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60))

    with open(os.path.join(exp02.OUT_DIR, "variants.json"), encoding="utf-8") as handle:
        submitted = json.load(handle)["results"]["mixed"]["decoder"]
    submitted = {"team_a": submitted["switch_team"], "team_b": 0.0, "enter_dead": submitted["enter_dead"],
                 "exit_dead": submitted["exit_dead"], "dead_bias": submitted["dead_bias"]}
    decoders = {"S": submitted}
    for v, (_, varying) in VARIANTS.items():
        print("  decoder search {}".format(v))
        decoders[v] = search(oof[MATCHED[0]][v] + oof[MATCHED[1]][v], varying)
    cv = {v: {s: summary(oof[s][v], decoders[v]) for s in DEGRADED} for v in decoders}

    print("\nHOLDOUT")
    models = train_all(list(TUNE))
    holdout = {v: {} for v in decoders}
    for s in DEGRADED:
        for v, cases in cases_for(models, HOLDOUT, s).items():
            holdout[v][s] = summary(cases, decoders[v])

    results = {}
    for v in decoders:
        results[v] = {"decoder": decoders[v], "cv": cv[v], "holdout": holdout[v],
                      "cv_matched": float(np.mean([cv[v][s]["spellF1"] for s in MATCHED])),
                      "cv_worst": float(min(cv[v][s]["spellF1"] for s in DEGRADED)),
                      "holdout_matched": float(np.mean([holdout[v][s]["spellF1"] for s in MATCHED]))}
    best = max(r["cv_matched"] for r in results.values())
    close = [v for v, r in results.items() if r["cv_matched"] >= best - TIE]
    chosen = max(close, key=lambda v: results[v]["cv_worst"])
    out = {"chosen": chosen, "results": results, "ball_model": __import__("ball_model").params(),
           "change_radius": CHANGE_RADIUS, "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    write_tables(out)
    print("\nchosen: {}  ({:.0f} min)".format(chosen, (time.time() - started) / 60))


LABELS = {"A": "A  v2 signals, fixed price", "B": "B  + touch signals, fixed price",
          "C": "C  v2 signals, varying price", "D": "D  + touch signals, varying price",
          "S": "S  submitted v2 (exp02 ball)"}


def write_tables(out):
    lines = ["# exp03 -- results (generated by external/exp03.py)", "",
             "Chosen on cross-validation (TUNE only): **{}**".format(LABELS[out["chosen"]]), "",
             "All variants evaluated on the new ball model {}.".format(out["ball_model"]), "",
             "| variant | CV matched | CV worst | holdout matched | decoder |", "|---|---|---|---|---|"]
    for v, r in out["results"].items():
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(LABELS[v], r["cv_matched"], r["cv_worst"],
                                                                    r["holdout_matched"], r["decoder"]))
    for v, r in out["results"].items():
        lines += ["", "## {}".format(LABELS[v]), "",
                  "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout @0.7 | holdout transF1 | holdout pred/true |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in DEGRADED:
            c, h = r["cv"][s], r["holdout"][s]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["spell@0.7"], h["transF1"], h["pred/true"]))
    with open(os.path.join(OUT_DIR, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
