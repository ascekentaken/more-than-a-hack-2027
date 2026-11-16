"""Experiment exp07: learn and judge on DFL labels rewritten to the challenge's "controlled possession".

    python external/exp07.py            # writes experiments/exp07_control/results.md and results.json
    python external/exp07.py --final    # writes model/params_v7.json for the adopted variant

Why. CHALLENGE.md: possession changes only when the other team gains *controlled* possession; "a
simple touch of the ball without control", "a deflection of a pass that does not intercept it" and
"an aerial duel won without subsequent control" do not count, and while the ball is contested or
loose possession stays with the last team. A suggested proxy: "a player keeps the ball for a few
seconds". The DFL labels we train and judge on count every touch: median possession 3.7 s, 30%
shorter than 2 s. So everything since exp04w has been pushed towards catching possessions the
challenge does not count.

The rewrite (controlled(), applied to every DFL truth, for training and for judging): going through
each half left to right, a run of one team shorter than MIN_CONTROL frames that comes straight after
a run of the other team becomes the other team's (it never got control; the ball stays with the last
team). A run right after a dead ball is kept whatever its length (a restart has a clear possessor).
MIN_CONTROL = 50 frames (2 s), the low end of "a few seconds", fixed from the text before the run.

Variants, all judged on the rewritten truth (leave-one-TUNE-match-out, draw 0), decoder searched again:
  R0  v4w as it is (exp04w W05 answers)                                   <- the reference
  R6  v6 as it is (exp06 TW answers)
  RW  v4w's recipe (frame model weighted (1/L)^0.5, change model) trained on the rewritten truth
  RS  RW plus exp06's stacker (TW), trained on the rewritten truth, nested as in exp06

Decision rule, fixed before the run: the variant among R6, RW, RS with the highest CV spellF1 (mean of
the two matched scenarios) replaces R0 only if that is at least MIN_GAIN above R0's and its worst
case over the four scenarios is not more than MIN_GAIN below R0's. HOLDOUT only reported.
"""

import datetime
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp04  # noqa: E402,F401  (sets exp03 to the exp04 simulator and paths)
import exp03  # noqa: E402
import exp02  # noqa: E402
import exp05  # noqa: E402
import exp06  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
MIN_CONTROL = 50
MIN_GAIN = 0.005
VARIANTS = ("R0", "R6", "RW", "RS")
STACK_ALPHA = 0.5
OUT_DIR = os.path.join(ROOT, "experiments", "exp07_control")
PARAMS_PATH = os.path.join(ROOT, "model", "params_v7.json")


def controlled(truth, min_control=MIN_CONTROL):
    """Short runs of one team straight after the other team's run go to the other team."""
    out = truth.copy()
    change = np.ones(len(out), dtype=bool)
    change[1:] = out[1:] != out[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(out))
    current = None
    for a, b in zip(starts, ends):
        code = truth[a]
        if code in (pm.H, pm.A):
            if current is not None and code != current and b - a < min_control:
                out[a:b] = current
            else:
                current = code
        else:
            current = None
    return out


_item = exp03.item


def item(match_id, scenario, swap=False, rep=0):
    table, truth, runs = _item(match_id, scenario, swap, rep)
    return table, controlled(truth), runs


exp03.item = item  # every training target and every judged truth from here on is rewritten


def checkpoint(name, compute):
    path = os.path.join(exp02.CACHE_DIR, "exp07_{}.pkl".format(name))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = compute()
    with open(path + ".tmp", "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return value


def rejudged(cases):
    return [(p, c, h, controlled(g), r) for p, c, h, g, r in cases]


def answers(models, names, matches, scenario):
    frame_model, change_model = models
    out = []
    for m in matches:
        table, truth, runs = exp03.item(m, scenario)
        swapped = exp03.item(m, scenario, True)[0]
        out.append((exp03.predict_frame(frame_model, names, table, swapped),
                    exp03.predict_change(change_model, names, table, swapped), table["half"].values, truth, runs))
    return out


def train(matches, names):
    return exp05.fit(matches, names, "frame"), exp03.fit(matches, names, "change")


def rw_fold(held, names):
    return checkpoint("RW_fold_" + held, lambda: (lambda models: {
        s: answers(models, names, [held], s) for s in exp03.DEGRADED})(
            train([m for m in exp03.TUNE if m != held], names)))


def rw_holdout(names):
    return checkpoint("RW_holdout", lambda: (lambda models: {
        s: answers(models, names, exp03.HOLDOUT, s) for s in exp03.DEGRADED})(train(list(exp03.TUNE), names)))


def inner(held, names, started):
    out = {}
    for m in exp03.TUNE:
        if m == held:
            continue
        pair = sorted((held, m))
        done = checkpoint("pair_{}_{}".format(*pair), lambda: exp06.first_stage(
            [x for x in exp03.TUNE if x not in pair], pair, names))
        out.update({k: v for k, v in done.items() if k[0] == m})
        print("    pair {} {} ({:.0f} min so far)".format(pair[0], pair[1], (time.time() - started) / 60), flush=True)
    return out


def restacked(model, names, matches, scenario, base):
    out = []
    for m, (prob, change, half, truth, runs) in zip(matches, base):
        table, swapped = exp03.item(m, scenario)[0], exp03.item(m, scenario, True)[0]
        out.append((exp06.stack_predict(model, prob, change, half, table, swapped, names), change, half, truth, runs))
    return out


def cases_for(v, names, started):
    """({scenario: CV cases}, {scenario: HOLDOUT cases}) for variant v, truth rewritten."""
    cv = {s: [] for s in exp03.DEGRADED}
    if v in ("R0", "R6"):
        tag = "exp04w_W05" if v == "R0" else "exp06_TW"
        for held in exp03.TUNE:
            done = exp05.load("{}_fold_{}".format(tag, held))
            for s in exp03.DEGRADED:
                cv[s] += rejudged(done[s])
        done = exp05.load(tag + "_holdout")
        return cv, {s: rejudged(done[s]) for s in exp03.DEGRADED}
    for held in exp03.TUNE:
        base = rw_fold(held, names)
        if v == "RS":
            base = checkpoint("RS_fold_" + held, lambda: (lambda model: {
                s: restacked(model, names, [held], s, base[s]) for s in exp03.DEGRADED})(
                    exp06.stack_fit(inner(held, names, started), names, STACK_ALPHA)))
        for s in exp03.DEGRADED:
            cv[s] += base[s]
        print("  {} fold {} ({:.0f} min so far)".format(v, held, (time.time() - started) / 60), flush=True)
    base = rw_holdout(names)
    if v == "RS":
        def loo():
            out = {}
            for m in exp03.TUNE:
                case = rw_fold(m, names)
                out.update({(m, s): case[s][0][:2] for s in exp03.DEGRADED})
            return out
        base = checkpoint("RS_holdout", lambda: (lambda model: {
            s: restacked(model, names, exp03.HOLDOUT, s, base[s]) for s in exp03.DEGRADED})(
                exp06.stack_fit(loo(), names, STACK_ALPHA)))
    return cv, base


def block(by_scenario, decoder):
    s = {sc: exp03.summary(by_scenario[sc], decoder) for sc in exp03.DEGRADED}
    return s, float(np.mean([s[sc]["spellF1"] for sc in exp03.MATCHED])), float(min(s[sc]["spellF1"] for sc in exp03.DEGRADED))


def write(results, started):
    done = [v for v in VARIANTS if v in results]
    ref = results.get("R0")
    others = [v for v in done if v != "R0"]
    best = max(others, key=lambda v: results[v]["cv_matched"]) if others else None
    adopt = bool(ref and best) and (results[best]["cv_matched"] >= ref["cv_matched"] + MIN_GAIN
                                    and results[best]["cv_worst"] >= ref["cv_worst"] - MIN_GAIN)
    out = {"best": best, "adopt": adopt, "finished": done, "results": results, "min_control": MIN_CONTROL,
           "rule": "best of R6/RW/RS: cv_matched >= R0 + {0} and cv_worst >= R0 - {0}".format(MIN_GAIN),
           "minutes": round((time.time() - started) / 60, 1),
           "written": datetime.datetime.now().replace(microsecond=0).isoformat()}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, default=float)
    row = "| {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {} |"
    lines = ["# exp07 -- judged on 'controlled possession' (runs < {} frames after the other team merged); "
             "generated by external/exp07.py".format(MIN_CONTROL), "",
             "Finished: {}. Rule: {} -> **{}**".format(", ".join(done), out["rule"],
                                                       ("adopt " + best) if adopt else "keep R0 (v4w)"), "",
             "| | CV matched | CV worst | holdout matched | holdout pred/true (matched_speed) | decoder |",
             "|---|---|---|---|---|---|"]
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
    print("\n".join(lines[:6 + len(done)]), flush=True)


def main():
    started = time.time()
    names = exp03.names_for("v3")
    results = {}
    path = os.path.join(OUT_DIR, "results.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            results = json.load(handle)["results"]
    for v in VARIANTS:
        if v in results:
            continue
        print("\nvariant", v, flush=True)
        cv_cases, holdout_cases = cases_for(v, names, started)
        print("  decoder search", v, flush=True)
        decoder = exp03.search(cv_cases[exp03.MATCHED[0]] + cv_cases[exp03.MATCHED[1]], True)
        cv, cv_matched, cv_worst = block(cv_cases, decoder)
        holdout, holdout_matched, _ = block(holdout_cases, decoder)
        results[v] = {"decoder": decoder, "cv": cv, "holdout": holdout, "cv_matched": cv_matched,
                      "cv_worst": cv_worst, "holdout_matched": holdout_matched}
        write(results, started)


def final():
    with open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8") as handle:
        out = json.load(handle)
    v = out["best"] if out["adopt"] else "R0"
    r = out["results"][v]
    names = exp03.names_for("v3")
    base = {"R0": "params_v4w.json", "R6": "params_v6.json"}.get(v, "params_v4w.json")
    with open(os.path.join(ROOT, "model", base), encoding="utf-8") as handle:
        params = json.load(handle)
    if v in ("RW", "RS"):
        matches = exp03.all_matches()[:7]
        frame_model, change_model = train(matches, names)
        files = {}
        for key, model in (("frame_model", frame_model), ("change_model", change_model)):
            path = os.path.join(ROOT, "model", "v7_{}.txt".format(key.split("_")[0]))
            model.save_model(path + ".tmp")
            os.replace(path + ".tmp", path)
            params[key] = os.path.relpath(path, ROOT).replace(os.sep, "/")
        params.pop("stack_model", None)
        if v == "RS":
            got = {}
            for m in matches:
                got.update(checkpoint("final_loo_" + m, lambda: exp06.first_stage([x for x in matches if x != m], [m], names)))
                print("  leave-one-out", m, flush=True)
            path = os.path.join(ROOT, "model", "v7_stack.txt")
            exp06.stack_fit(got, names, STACK_ALPHA).save_model(path + ".tmp")
            os.replace(path + ".tmp", path)
            params.update({"stack_model": os.path.relpath(path, ROOT).replace(os.sep, "/"), "stack_alpha": STACK_ALPHA})
    params.update({"variant": params["variant"] + "; exp07 {} (controlled possession, {} frames)".format(v, MIN_CONTROL),
                   "decoder": r["decoder"], "controlled_min_frames": MIN_CONTROL,
                   "written": datetime.datetime.now().replace(microsecond=0).isoformat()})
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=1)
    print("written", os.path.relpath(PARAMS_PATH, ROOT), "for", v)


if __name__ == "__main__":
    final() if "--final" in sys.argv else main()
