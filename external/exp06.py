"""Experiment exp06: a second model that reads v4w's answers over +-2 s (stacking with time context).

    python external/exp06.py            # writes experiments/exp06_context/results.md and results.json
    python external/exp06.py --final    # retrains the adopted variant on all 7 DFL matches -> model/params_v6.json

Why. v4w's frame model looks at one frame at a time (with some signals that sum up the recent past).
A 1-second interception is a pattern in time -- the ball near a home player, then near an away
player -- and in 60% of the short possessions v4w misses, the frame model never gives the right team
more than 0.5 (analysis/dfl_errors.py). The decoder then sees too weak a signal to pay for two changes.

What it adds: a second LightGBM model (the stacker) whose inputs, for frame t, are
  - v4w's frame probabilities (log) and change probability (logit) at t and at t +- 1, 3, 6, 12, 25, 50 frames
  - their means over centred windows of 25 and 75 frames, over the 50 frames before and the 50
    after, and the after-minus-before difference for H and A; the max change probability over +-12, +-25
  - the 76 v4w signals at t
and whose target is H / A / D at t. Its output replaces v4w's frame probabilities in the decoder;
the change probabilities stay v4w's. Read from both sides and averaged, as the frame model.

No leak: the stacker must learn on first-stage answers for matches the first stage has not seen.
In the fold that holds out match h, for every other TUNE match m the first stage (v4w frame + v4
change) is trained on TUNE minus {h, m} and predicts m; the stacker learns on those four matches and
is evaluated on h with v4w's own answers for h (trained on TUNE minus h, exp04w's checkpoints).
For HOLDOUT, the stacker learns on the five TUNE matches' leave-one-out answers.

Variants:
  T    stacker, every frame weighs alike
  TW   stacker with the weights (1 / L) ** 0.5 of v4w
Decoder prices searched again for each.

Decision rule, fixed before the run: the variant with the higher leave-one-match-out spellF1 on the
TUNE matches (mean of the two matched scenarios) replaces v4w only if that is at least MIN_GAIN above
v4w's and its worst case over the four degraded scenarios is not more than MIN_GAIN below v4w's.
The DFL HOLDOUT matches are only reported.
"""

import datetime
import itertools
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
import exp05  # noqa: E402
import lightgbm as lgb  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
OFFSETS = (-50, -25, -12, -6, -3, -1, 1, 3, 6, 12, 25, 50)
CENTRED = (25, 75)
SIDE = 50
MAX_WINDOWS = (25, 51)
ALPHAS = {"T": 0.0, "TW": 0.5}
MIN_GAIN = 0.005
OUT_DIR = os.path.join(ROOT, "experiments", "exp06_context")
BASE_PARAMS = os.path.join(ROOT, "model", "params_v4w.json")
PARAMS_PATH = os.path.join(ROOT, "model", "params_v6.json")


# -- the stacker's inputs ----------------------------------------------------------------------

def shifted(a, offset, half):
    """a[t + offset], NaN past the ends or in another half."""
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float32)
    same = np.zeros(n, dtype=bool)
    if offset > 0:
        out[:n - offset] = a[offset:]
        same[:n - offset] = half[offset:] == half[:n - offset]
    else:
        out[-offset:] = a[:n + offset]
        same[-offset:] = half[:n + offset] == half[-offset:]
    out[~same] = np.nan
    return out


def rolling(a, half, window, how):
    """Rolling mean (or max) inside each half: how in 'centred', 'before', 'after', 'max'."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        s = pd.Series(a[first:last + 1])
        if how == "centred":
            r = s.rolling(window, center=True, min_periods=1).mean()
        elif how == "max":
            r = s.rolling(window, center=True, min_periods=1).max()
        elif how == "before":
            r = s.rolling(window, min_periods=1).mean()
        else:
            r = s[::-1].rolling(window, min_periods=1).mean()[::-1]
        out[first:last + 1] = r.values
    return out


def stack_inputs(prob, change, half, table, names):
    cols = {}
    logp = np.log(prob).astype(np.float32)
    logit_c = np.log(change / (1 - change)).astype(np.float32)
    for k, label in enumerate("HAD"):
        cols["s_p" + label] = logp[:, k]
        for o in OFFSETS:
            cols["s_p{}_{}".format(label, o)] = shifted(logp[:, k], o, half)
        for w in CENTRED:
            cols["s_p{}_c{}".format(label, w)] = rolling(prob[:, k], half, w, "centred")
        cols["s_p{}_before".format(label)] = rolling(prob[:, k], half, SIDE, "before")
        cols["s_p{}_after".format(label)] = rolling(prob[:, k], half, SIDE, "after")
    for label in "HA":
        cols["s_p{}_jump".format(label)] = cols["s_p{}_after".format(label)] - cols["s_p{}_before".format(label)]
    cols["s_c"] = logit_c
    for o in OFFSETS:
        cols["s_c_{}".format(o)] = shifted(logit_c, o, half)
    for w in MAX_WINDOWS:
        cols["s_c_max{}".format(w)] = rolling(change, half, w, "max")
    if LONG:
        cols.update(long_context(table, half))
    if EXTRA:
        cols.update(extra_context(table, half))
    if EXTRA2:
        cols.update(team_context(table, half))
    out = pd.DataFrame(cols)
    return pd.concat([out, table[names].astype(np.float32).reset_index(drop=True)], axis=1)


# -- longer context on the raw signals (exp08; off unless LONG is set) ---------------------------

LONG = False
EXTRA2 = False  # exp12: how long the same team has been the last to touch / control / be close
EXTRA = False   # exp10: pass chains, team movement, ball jumps over gaps, still ball at the lines
LONG_WINDOWS = (125, 250)       # 5 s and 10 s
SLOW_PLAYERS = 2.0              # m/s: mean player speed below this counts as slow
FAST_BALL = 5.0                 # m/s: a ball faster than this has been kicked
STILL_BALL = 1.5                # m/s
NEAR_BALL = 1.5                 # m


def run_length(flag, half, forward):
    """Frames since flag became true (or until it stops, forward=True), 0 where flag is false, per half."""
    out = np.zeros(len(flag), dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        f = flag[first:last + 1]
        if forward:
            f = f[::-1]
        count, r = 0, np.empty(len(f), dtype=np.float32)
        for i, x in enumerate(f):
            count = count + 1 if x else 0
            r[i] = count
        out[first:last + 1] = r[::-1] if forward else r
    return out


def frames_to(flag, half, forward):
    """Frames since flag was last true (forward=False) or until it is next true (forward=True), per half."""
    return run_length(~flag, half, forward)


def long_context(table, half):
    speed = table["speed_mean"].values.astype(np.float32)
    ball_speed = table["ball_speed"].values.astype(np.float32)
    seen = table["ball_seen"].values.astype(np.float32)
    d_min = table["d_min"].values.astype(np.float32)
    slow = np.nan_to_num(speed, nan=0.0) < SLOW_PLAYERS
    kicked = np.nan_to_num(ball_speed, nan=0.0) > FAST_BALL
    still = (seen > 0) & (np.nan_to_num(ball_speed, nan=99.0) < STILL_BALL)
    guarded = still & (np.nan_to_num(d_min, nan=99.0) < NEAR_BALL)
    cols = {"l_slow_for": run_length(slow, half, False), "l_slow_until": run_length(slow, half, True),
            "l_since_kick": frames_to(kicked, half, False), "l_to_kick": frames_to(kicked, half, True),
            "l_still_for": run_length(still, half, False), "l_still_until": run_length(still, half, True),
            "l_guarded_for": run_length(guarded, half, False), "l_guarded_until": run_length(guarded, half, True),
            "l_unseen_for": run_length(seen <= 0, half, False), "l_unseen_until": run_length(seen <= 0, half, True)}
    for w in LONG_WINDOWS:
        for name, a in (("speed", speed), ("seen", seen), ("ball_speed", ball_speed)):
            cols["l_{}_before{}".format(name, w)] = rolling(a, half, w, "before")
            cols["l_{}_after{}".format(name, w)] = rolling(a, half, w, "after")
    return cols


def delta(a, half, k):
    """a(t+k) - a(t-k) inside each half (edges held)."""
    out = np.zeros(len(a), dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        x = np.nan_to_num(pd.Series(a[first:last + 1].astype(np.float32)).ffill().bfill().values)
        n = len(x)
        i = np.arange(n)
        out[first:last + 1] = x[np.minimum(i + k, n - 1)] - x[np.maximum(i - k, 0)]
    return out


def chain(touch, team, half, forward):
    """Touches in a row by the same team up to t (or from t on, forward=True), capped at 10."""
    out = np.zeros(len(touch), dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        t, k = touch[first:last + 1], team[first:last + 1]
        if forward:
            t, k = t[::-1], k[::-1]
        r, count, current = np.empty(len(t), dtype=np.float32), 0, None
        for i in range(len(t)):
            if t[i]:
                count = count + 1 if k[i] == current else 1
                current = k[i]
            r[i] = min(count, 10)
        out[first:last + 1] = r[::-1] if forward else r
    return out


def extra_context(table, half):
    """exp10. All from columns already in the table, so the home/away swap carries over."""
    get = lambda name: table[name].values.astype(np.float32)
    touch = np.nan_to_num(get("touch")) > 0
    seen = get("ball_seen") > 0
    ball_speed = np.nan_to_num(get("ball_speed"), nan=99.0)
    still = seen & (ball_speed < STILL_BALL)
    at_line = still & ((np.nan_to_num(get("ball_to_touch"), nan=99.0) < 1.5)
                       | (np.nan_to_num(get("ball_to_goalline"), nan=99.0) < 1.5))
    cols = {"x_chain_past": chain(touch, table["last_touch_team"].values, half, False),
            "x_chain_next": chain(touch, table["next_touch_team"].values, half, True),
            "x_line_for": run_length(at_line, half, False), "x_line_until": run_length(at_line, half, True)}
    for k in (25, 75):
        for side in ("home", "away"):
            cols["x_cvel_{}_{}".format(side, k)] = delta(get("centroid_fwd_" + side), half, k)
        cols["x_ball_adv_{}".format(k)] = delta(np.where(seen, get("ball_fwd_home"), np.nan), half, k)
    for side in ("home", "away"):
        for axis in ("x", "y"):
            cols["x_dspread_{}_{}".format(axis, side)] = delta(get("spread_{}_{}".format(axis, side)), half, 50)
    # how far the ball moved across a tracking gap, and the speed that implies
    fwd, wide = np.where(seen, get("ball_fwd_home"), np.nan), np.where(seen, get("ball_wide"), np.nan)
    jump = np.zeros(len(fwd), dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        f, w = pd.Series(fwd[first:last + 1]), pd.Series(wide[first:last + 1])
        jump[first:last + 1] = np.nan_to_num(np.hypot(f.bfill().values - f.ffill().values,
                                                      w.bfill().values - w.ffill().values))
    cols["x_gap_jump"] = jump
    cols["x_gap_speed"] = jump / np.maximum(np.nan_to_num(get("gap_len")), 1.0)
    return cols


def same_for(values, half, forward):
    """Frames the value has been the same up to t (or will stay the same from t, forward=True), per half."""
    out = np.zeros(len(values), dtype=np.float32)
    for first, last in exp02.v1.blocks(half):
        v = values[first:last + 1]
        if forward:
            v = v[::-1]
        start = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
        r = (np.arange(len(v)) - np.repeat(start, np.diff(np.r_[start, len(v)]))).astype(np.float32)
        out[first:last + 1] = r[::-1] if forward else r
    return out


def team_context(table, half):
    """exp12. Per team signal (last touch, control, last close): how long it has pointed at the same
    team, and each team's share of the 5 and 10 s before and after."""
    cols = {}
    for key in ("last_touch_team", "next_touch_team", "control_team", "last_close_team", "next_close_team"):
        v = table[key].values
        cols["y_{}_for".format(key)] = np.minimum(same_for(v, half, False), 1500)
        cols["y_{}_until".format(key)] = np.minimum(same_for(v, half, True), 1500)
    for key in ("last_touch_team", "control_team", "last_close_team"):
        v = table[key].values
        for side, code in (("home", 1), ("away", -1)):
            flag = (v == code).astype(np.float32)
            for w in LONG_WINDOWS:
                cols["y_{}_{}_before{}".format(key, side, w)] = rolling(flag, half, w, "before")
                cols["y_{}_{}_after{}".format(key, side, w)] = rolling(flag, half, w, "after")
    return cols


def stack_predict(model, prob, change, half, table, table_swap, names):
    p = model.predict(stack_inputs(prob, change, half, table, names))
    q = model.predict(stack_inputs(prob[:, [1, 0, 2]], change, half, table_swap, names))
    return np.clip((p + q[:, [1, 0, 2]]) / 2, 1e-4, 1.0)


def stack_fit(first_stage, names, alpha):
    """first_stage: {(match, scenario): (prob, change)} answers for matches the first stage did not see."""
    xs, ys, ws = [], [], []
    for (m, s), (prob, change) in sorted(first_stage.items()):
        for swap in (False, True):
            table, truth, _ = exp03.item(m, s, swap)
            p = prob[:, [1, 0, 2]] if swap else prob
            keep = np.flatnonzero(truth != pm.X)[::exp02.STRIDE]
            xs.append(stack_inputs(p, change, table["half"].values, table, names).iloc[keep])
            ys.append(truth[keep])
            ws.append(np.maximum(exp04w.stretch_length(truth)[keep], exp04w.MIN_LEN).astype(np.float64) ** -alpha)
    w = np.concatenate(ws)
    return lgb.train(exp02.LGB_PARAMS, lgb.Dataset(pd.concat(xs, ignore_index=True), np.concatenate(ys), weight=w / w.mean()),
                     num_boost_round=exp02.N_TREES)


# -- first-stage answers for unseen matches ----------------------------------------------------

def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "exp06_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def first_stage(train, targets, names):
    """v4w frame model + v4 change model trained on `train`; their answers for `targets`, every scenario."""
    frame_model = exp05.fit(train, names, "frame")           # (1/L)^0.5 weights, as v4w
    change_model = exp03.fit(train, names, "change")         # as v4
    out = {}
    for m in targets:
        for s in exp03.DEGRADED:
            table, swapped = exp03.item(m, s)[0], exp03.item(m, s, True)[0]
            out[(m, s)] = (exp03.predict_frame(frame_model, names, table, swapped).astype(np.float32),
                           exp03.predict_change(change_model, names, table, swapped).astype(np.float32))
    return out


def inner(held, names, started):
    """First-stage answers for each TUNE match other than `held`, trained without it and without `held`."""
    out = {}
    for m in exp03.TUNE:
        if m == held:
            continue
        pair = sorted((held, m))
        done = checkpoint("pair_{}_{}".format(*pair), lambda: first_stage(
            [x for x in exp03.TUNE if x not in pair], pair, names))
        out.update({k: v for k, v in done.items() if k[0] == m})
        print("    pair {} {} ({:.0f} min so far)".format(pair[0], pair[1], (time.time() - started) / 60), flush=True)
    return out


def loo(matches):
    """{(match, scenario): (prob, change)} from exp04w W05 leave-one-out checkpoints."""
    out = {}
    for m in matches:
        cases = exp05.load("exp04w_W05_fold_" + m)
        for s in exp03.DEGRADED:
            prob, change = cases[s][0][0], cases[s][0][1]
            out[(m, s)] = (prob, change)
    return out


def restacked(model, names, matches, scenario, base):
    out = []
    for m, (prob, change, half, truth, runs) in zip(matches, base):
        table, swapped = exp03.item(m, scenario)[0], exp03.item(m, scenario, True)[0]
        out.append((stack_predict(model, prob, change, half, table, swapped, names), change, half, truth, runs))
    return out


def block(by_scenario, decoder):
    s = {sc: exp03.summary(by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def main():
    started = time.time()
    names = exp03.names_for("v3")
    with open(os.path.join(exp03.OUT_DIR, "weights.json"), encoding="utf-8") as handle:
        w05 = json.load(handle)["results"]["W05"]
    single = {k: w05[k] for k in ("cv_matched", "cv_worst", "holdout_matched", "decoder", "cv", "holdout")}

    print("leave-one-match-out on TUNE, first stage nested", flush=True)
    oof = {v: {s: [] for s in exp03.DEGRADED} for v in ALPHAS}
    for held in exp03.TUNE:
        base = exp05.load("exp04w_W05_fold_" + held)
        for v, alpha in ALPHAS.items():
            done = checkpoint("{}_fold_{}".format(v, held), lambda: (lambda model: {
                s: restacked(model, names, [held], s, base[s]) for s in exp03.DEGRADED})(
                    stack_fit(inner(held, names, started), names, alpha)))
            for s in exp03.DEGRADED:
                oof[v][s] += done[s]
        print("  fold {} ({:.0f} min so far)".format(held, (time.time() - started) / 60), flush=True)

    base = exp05.load("exp04w_W05_holdout")
    results = {}
    for v, alpha in ALPHAS.items():
        print("  decoder search", v, flush=True)
        decoder = exp03.search(oof[v][exp03.MATCHED[0]] + oof[v][exp03.MATCHED[1]], True)
        cv, cv_matched, cv_worst = block(oof[v], decoder)
        done = checkpoint(v + "_holdout", lambda: (lambda model: {
            s: restacked(model, names, exp03.HOLDOUT, s, base[s]) for s in exp03.DEGRADED})(
                stack_fit(loo(exp03.TUNE), names, alpha)))
        holdout, holdout_matched, _ = block(done, decoder)
        results[v] = {"alpha": alpha, "decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched,
                      "cv_worst": cv_worst, "holdout_matched": holdout_matched}
    best = max(results, key=lambda v: results[v]["cv_matched"])
    adopt = (results[best]["cv_matched"] >= single["cv_matched"] + MIN_GAIN
             and results[best]["cv_worst"] >= single["cv_worst"] - MIN_GAIN)
    out = {"best": best, "adopt": adopt, "results": results, "single": single,
           "rule": "best T: cv_matched >= v4w + {0} and cv_worst >= v4w - {0}".format(MIN_GAIN),
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {} |"
    lines = ["# exp06 -- stacker with +-2 s of context on top of v4w (generated by external/exp06.py)", "",
             "Rule: {} -> **{}**".format(out["rule"], "adopt " + best if adopt else "keep v4w"), "",
             "| | CV matched | CV worst | holdout matched | holdout pred/true (matched_speed) | decoder |",
             "|---|---|---|---|---|---|",
             row.format("v4w (exp04w W05)", single["cv_matched"], single["cv_worst"], single["holdout_matched"],
                        single["holdout"]["matched_speed"]["pred/true"], single["decoder"])]
    for v, r in results.items():
        lines.append(row.format("{} (alpha {})".format(v, r["alpha"]), r["cv_matched"], r["cv_worst"],
                                r["holdout_matched"], r["holdout"]["matched_speed"]["pred/true"], r["decoder"]))
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
    """First stage = v4w's models; the stacker learns on leave-one-out first-stage answers for all 7 DFL matches."""
    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    if not out["adopt"]:
        print("exp06 kept v4w; nothing to write")
        return
    r = out["results"][out["best"]]
    names = exp03.names_for("v3")
    matches = exp03.all_matches()[:7]
    answers = {}
    for m in matches:
        answers.update(checkpoint("final_loo_" + m, lambda: first_stage([x for x in matches if x != m], [m], names)))
        print("  leave-one-out", m, flush=True)
    path = os.path.join(ROOT, "model", "v6_stack.txt")
    stack_fit(answers, names, r["alpha"]).save_model(path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(BASE_PARAMS, encoding="utf-8") as handle:
        params = json.load(handle)
    params.update({"variant": params["variant"] + "; exp06 stacker " + out["best"],
                   "stack_model": os.path.relpath(path, ROOT).replace(os.sep, "/"), "stack_alpha": r["alpha"],
                   "decoder": r["decoder"], "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT))


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
