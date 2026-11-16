"""Experiment exp05: four changes to v4w (exp04w, W05), each tried alone against it.

    python external/exp05.py            # writes experiments/exp05_more/results.md and results.json
    python external/exp05.py --final    # retrains the adopted variant on all 7 DFL matches -> model/params_v5.json

Base: v4w = exp04 variant D with the frame model trained with weights (1 / L) ** 0.5 (exp04w, W05).
Its leave-one-match-out cases are read from exp04w's checkpoints. Each variant changes one thing and
reuses the rest of v4w; decoder prices searched again for each, as in exp04w.

  WC  the change model too is trained with the weights (1 / L) ** 0.5 (frame model: v4w's)
  WA  the frame model is trained on three draws of the degradation simulator (draws 0, 1, 2) instead
      of one -- more varied degraded data in a single model, not an ensemble (change model: v4w's)
  WB  a larger frame model: num_leaves 127, min_child_samples 100, 800 trees (change model: v4w's)
  WM  the two Metrica matches added to the training data of every fold, frame model weighted as v4w;
      the change model is exp04m's (DFL + Metrica, from its checkpoints)

Decision rule, fixed before the run: the variant with the highest leave-one-match-out spellF1 on the
TUNE matches (mean of the two matched scenarios) replaces v4w only if that is at least MIN_GAIN above
v4w's and its worst case over the four degraded scenarios is not more than MIN_GAIN below v4w's.
The DFL HOLDOUT matches are only reported. Variants that have not finished by the decision time are
left out (the README says which).
"""

import datetime
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp04  # noqa: E402,F401  (sets exp03 to the exp04 simulator and paths)
import exp03  # noqa: E402
import exp02  # noqa: E402
import exp04w  # noqa: E402
import metrica  # noqa: E402
import lightgbm as lgb  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
ALPHA = 0.5
MIN_LEN = exp04w.MIN_LEN
MIN_GAIN = 0.005
DRAWS = (0, 1, 2)
BIG = {"num_leaves": 127, "min_child_samples": 100}
BIG_TREES = 800
EXTRA = sorted(metrica.GAMES)
VARIANTS = ("WC", "WA", "WB", "WM")
OUT_DIR = os.path.join(ROOT, "experiments", "exp05_more")
BASE_PARAMS = os.path.join(ROOT, "model", "params_v4w.json")
PARAMS_PATH = os.path.join(ROOT, "model", "params_v5.json")


def rows(matches, names, target, alpha, draws=(0,)):
    xs, ys, ws = [], [], []
    for m in matches:
        for rep in draws:
            for s in exp03.DEGRADED:
                for swap in (False, True):
                    table, truth, _ = exp03.item(m, s, swap, rep)
                    keep = np.flatnonzero(truth != pm.X)[::exp02.STRIDE]
                    xs.append(table.iloc[keep][names].astype(np.float32))
                    ys.append(truth[keep] if target == "frame" else exp03.change_target(truth)[keep])
                    ws.append(np.maximum(exp04w.stretch_length(truth)[keep], MIN_LEN).astype(np.float64) ** -alpha)
    w = np.concatenate(ws)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), w / w.mean()


def fit(matches, names, target, alpha=ALPHA, draws=(0,), big=False):
    params = dict(exp02.LGB_PARAMS if target == "frame" else exp03.CHANGE_PARAMS, seed=exp03.SEED)
    if big:
        params.update(BIG)
    x, y, w = rows(matches, names, target, alpha, draws)
    return lgb.train(params, lgb.Dataset(x, y, weight=w), num_boost_round=BIG_TREES if big else exp02.N_TREES)


def models_for(variant, train, names):
    """(frame model or None, change model or None) for `variant` trained on `train`; None = v4w's."""
    if variant == "WC":
        return None, fit(train, names, "change")
    if variant == "WA":
        return fit(train, names, "frame", draws=DRAWS), None
    if variant == "WB":
        return fit(train, names, "frame", big=True), None
    if variant == "WM":
        return fit(train + EXTRA, names, "frame"), "exp04m"
    raise ValueError(variant)


def load(name):
    with open(os.path.join(exp02.CACHE_DIR, name + ".pkl"), "rb") as handle:
        return pickle.load(handle)


def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "exp05_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def swap_in(models, names, matches, scenario, base, m04):
    """v4w's cases for `matches` with the parts given in `models` replaced."""
    frame_model, change_model = models
    out = []
    for i, (m, (frame, change, half, truth, runs)) in enumerate(zip(matches, base)):
        table, swapped = exp03.item(m, scenario)[0], exp03.item(m, scenario, True)[0]
        if frame_model is not None:
            frame = exp03.predict_frame(frame_model, names, table, swapped)
        if change_model == "exp04m":
            change = m04[i][1]
        elif change_model is not None:
            change = exp03.predict_change(change_model, names, table, swapped)
        out.append((frame, change, half, truth, runs))
    return out


def block(by_scenario, decoder):
    s = {sc: exp03.summary(by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def run_variant(v, names, started):
    oof = {s: [] for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        base, m04 = load("exp04w_W05_fold_" + held), load("exp04m_fold_" + held)
        train = [m for m in exp03.TUNE if m != held]
        done = checkpoint("{}_fold_{}".format(v, held), lambda: (lambda models: {
            s: swap_in(models, names, [held], s, base[s], m04[s]) for s in exp03.DEGRADED})(models_for(v, train, names)))
        for s in exp03.DEGRADED:
            oof[s] += done[s]
        print("  {} fold {} ({:.0f} min so far)".format(v, held, (time.time() - started) / 60), flush=True)
    print("  decoder search", v, flush=True)
    decoder = exp03.search(oof[exp03.MATCHED[0]] + oof[exp03.MATCHED[1]], True)
    cv, cv_matched, cv_worst = block(oof, decoder)
    base, m04 = load("exp04w_W05_holdout"), load("exp04m_holdout")
    done = checkpoint(v + "_holdout", lambda: (lambda models: {
        s: swap_in(models, names, exp03.HOLDOUT, s, base[s], m04[s]) for s in exp03.DEGRADED})(
            models_for(v, list(exp03.TUNE), names)))
    holdout, holdout_matched, _ = block(done, decoder)
    return {"decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched, "cv_worst": cv_worst,
            "holdout_matched": holdout_matched}


def write(results, single, started):
    done = [v for v in VARIANTS if v in results]
    best = max(done, key=lambda v: results[v]["cv_matched"]) if done else None
    adopt = bool(best) and (results[best]["cv_matched"] >= single["cv_matched"] + MIN_GAIN
                            and results[best]["cv_worst"] >= single["cv_worst"] - MIN_GAIN)
    out = {"best": best, "adopt": adopt, "finished": done, "results": results, "single": single,
           "rule": "best variant: cv_matched >= v4w + {0} and cv_worst >= v4w - {0}".format(MIN_GAIN),
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {} |"
    lines = ["# exp05 -- four changes to v4w (generated by external/exp05.py)", "",
             "Finished: {}. Rule: {} -> **{}**".format(", ".join(done) or "none", out["rule"],
                                                       "adopt " + best if adopt else "keep v4w"), "",
             "| | CV matched | CV worst | holdout matched | holdout pred/true (matched_speed) | decoder |",
             "|---|---|---|---|---|---|",
             row.format("v4w (exp04w W05)", single["cv_matched"], single["cv_worst"], single["holdout_matched"],
                        single["holdout"]["matched_speed"]["pred/true"], single["decoder"])]
    for v in done:
        r = results[v]
        lines.append(row.format(v, r["cv_matched"], r["cv_worst"], r["holdout_matched"],
                                r["holdout"]["matched_speed"]["pred/true"], r["decoder"]))
    for v in done:
        r = results[v]
        lines += ["", "## " + v, "",
                  "| scenario | CV spellF1 | CV macroF1 | holdout spellF1 | holdout macroF1 | holdout transF1 | holdout pred/true |",
                  "|---|---|---|---|---|---|---|"]
        for s in exp03.DEGRADED:
            c, h = r["cv"][s], r["holdout"][s]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                s, c["spellF1"], c["macroF1"], h["spellF1"], h["macroF1"], h["transF1"], h["pred/true"]))
    with open(os.path.join(OUT_DIR, "results.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[:7 + len(done)]), flush=True)


def main():
    started = time.time()
    names = exp03.names_for("v3")
    with open(os.path.join(exp03.OUT_DIR, "weights.json"), encoding="utf-8") as handle:
        w05 = json.load(handle)["results"]["W05"]
    single = {k: w05[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder", "cv", "holdout")}
    only = [a for a in sys.argv[1:] if a in VARIANTS]
    results = {}
    path = os.path.join(OUT_DIR, "results.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            results = json.load(handle)["results"]
    for v in only or VARIANTS:
        if v in results:
            continue
        print("\nvariant", v, flush=True)
        results[v] = run_variant(v, names, started)
        write(results, single, started)


def final():
    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    if not out["adopt"]:
        print("exp05 kept v4w; nothing to write")
        return
    v, r = out["best"], out["results"][out["best"]]
    names = exp03.names_for("v3")
    frame_model, change_model = models_for(v, exp03.all_matches()[:7], names)
    with open(BASE_PARAMS, encoding="utf-8") as handle:
        params = json.load(handle)
    if frame_model is not None:
        path = os.path.join(ROOT, "model", "v5_frame.txt")
        frame_model.save_model(path + ".tmp")
        os.replace(path + ".tmp", path)
        params["frame_model"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
    if change_model == "exp04m":
        path = os.path.join(ROOT, "model", "v5_change.txt")
        exp03.fit(exp03.all_matches(), names, "change").save_model(path + ".tmp")
        os.replace(path + ".tmp", path)
        params["change_model"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
    elif change_model is not None:
        path = os.path.join(ROOT, "model", "v5_change.txt")
        change_model.save_model(path + ".tmp")
        os.replace(path + ".tmp", path)
        params["change_model"] = os.path.relpath(path, ROOT).replace(os.sep, "/")
    params.update({"variant": params["variant"] + "; exp05 " + v, "exp05_variant": v, "decoder": r["decoder"],
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    if v == "WM":
        params["trained_on"] += ("; plus Metrica Sports sample games 1 and 2 (github.com/metrica-sports/sample-data), "
                                 "labels rebuilt from their event logs")
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
