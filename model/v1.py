"""Possession model v1: hand-written signals and a three-state decoder, tuned on external data only.

    python model/v1.py         # writes evaluation/results/v1_raw.csv for the challenge match

How it decides, for each frame:

  1. Alive or dead? A probability that the ball is dead, from how slowly the players move (2-second
     average), whether the tracking sees the ball, whether the ball is outside the lines and
     whether it is moving fast.
  2. Which team? When the ball is seen and someone is close to it, the team whose nearest player is
     closer gets the evidence; the further away both teams are, the weaker it is. With no ball, or
     a ball far from everyone (a long pass in flight), there is no team evidence at all.
  3. One sequence, not frame-by-frame guesses. A Viterbi decoder over H, A, D picks the most likely
     sequence for each half, paying a fixed price for every change of team and another for
     entering or leaving dead ball. It reads the whole half at once, so a gap in the ball is decided
     by what happens on both sides of it.

Where the numbers come from. Every parameter is in model/params_v1.json, written by
external/tune.py from the open DFL matches (external data, CC BY 4.0), made to look like the
challenge tracking first. Nothing is fitted on the challenge match, its ground truth, its video or
any hand labels -- the competition rules forbid all four. Two published facts are used as they
are: frames outside the two halves are X, and the second half is dead ball until its kickoff
(FAQ.md).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
GAME = games.GAME
N_FRAMES = games.source_frames("tactical", GAME)
SECOND_HALF_KICKOFF = 113742
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PARAMS_PATH = os.path.join(HERE, "params_v1.json")

# Starting values before any tuning, chosen by reasoning about football, not measured on the
# challenge data. external/tune.py starts its search here and writes the result to PARAMS_PATH.
DEFAULT_PARAMS = {
    "dead_speed_mid": 1.5,      # m/s of mean player speed (2 s) where slowness alone says 50/50
    "dead_speed_slope": 2.5,    # log-odds per m/s
    "dead_ball_missing": 0.7,   # extra log-odds of dead ball when the tracking has no ball
    "dead_ball_outside": 3.0,   # extra log-odds when the ball is outside the lines
    "dead_ball_fast": -1.5,     # a ball moving faster than ball_fast_ms is in play
    "ball_fast_ms": 8.0,
    "team_scale_m": 0.75,       # this much closer is about 80/20 for the closer team
    "team_reach_m": 1.0,        # full weight while the nearest player is within this distance
    "team_decay_m": 2.5,        # then the weight decays with this length scale
    "team_clip": 0.02,
    "switch_team": 20.0,        # price of H <-> A, in summed log-likelihood
    "switch_dead": 25.0,        # price of entering or leaving D
}

try:
    from numba import njit
except ImportError:  # still correct without numba, about a hundred times slower
    def njit(function=None, **_):
        return function if function is not None else (lambda f: f)


def load_params(path=PARAMS_PATH):
    params = dict(DEFAULT_PARAMS)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            params.update(json.load(handle)["params"])
    return params


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def emissions(table, p):
    """(n, 3) log-likelihoods of H, A, D for every frame of a feature table."""
    speed = table["speed_mean_2s"].fillna(p["dead_speed_mid"]).values
    z = p["dead_speed_slope"] * (p["dead_speed_mid"] - speed)
    z += np.where(table["ball_seen"].values, 0.0, p["dead_ball_missing"])
    z += np.where(table["ball_outside"].values, p["dead_ball_outside"], 0.0)
    z += np.where(table["ball_speed"].fillna(0).values > p["ball_fast_ms"], p["dead_ball_fast"], 0.0)
    p_dead = np.clip(sigmoid(z), 1e-3, 1 - 1e-3)

    d_home = table["d_home"].values
    d_away = table["d_away"].values
    known = np.isfinite(d_home) & np.isfinite(d_away)
    diff = np.clip(np.where(known, d_away - d_home, 0.0), -5, 5)
    q_home = sigmoid(diff / p["team_scale_m"])
    closest = np.where(known, np.minimum(d_home, d_away), np.inf)
    weight = np.exp(-np.maximum(closest - p["team_reach_m"], 0.0) / p["team_decay_m"])
    q_home = 0.5 + weight * (q_home - 0.5)
    q_home = np.clip(np.where(known, q_home, 0.5), p["team_clip"], 1 - p["team_clip"])

    alive = np.log(1 - p_dead)
    out = np.empty((len(table), 3))
    out[:, 0] = alive + np.log(q_home)
    out[:, 1] = alive + np.log(1 - q_home)
    out[:, 2] = np.log(p_dead) + np.log(0.5)  # same footing as one team's share of "alive"
    return out


@njit(cache=True)
def viterbi(log_e, switch_team, switch_dead):
    """Most likely H/A/D sequence. Changes cost a fixed price, staying costs nothing."""
    n = log_e.shape[0]
    cost = np.array([[0.0, switch_team, switch_dead],
                     [switch_team, 0.0, switch_dead],
                     [switch_dead, switch_dead, 0.0]])
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


def blocks(half):
    """[(first, last)] contiguous runs of frames to label (half > 0)."""
    mask = (np.asarray(half) > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def decode(table, p, log_e=None):
    """Codes 0 H, 1 A, 2 D, 3 X for any match's feature table, each half decoded on its own."""
    if log_e is None:
        log_e = emissions(table, p)
    codes = np.full(len(table), pm.X, dtype=np.int8)
    for first, last in blocks(table["half"].values):
        codes[first:last + 1] = viterbi(np.ascontiguousarray(log_e[first:last + 1]),
                                        float(p["switch_team"]), float(p["switch_dead"]))
    return codes


def predict(table, p):
    """Letters for the challenge match, with its two published facts applied."""
    log_e = emissions(table, p)
    before_kickoff = (table["half"].values == 2) & (table["frame"].values < SECOND_HALF_KICKOFF)
    log_e[before_kickoff] = [-50.0, -50.0, 0.0]
    return np.array(pm.LABELS)[decode(table, p, log_e)]


def write(path, labels):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def possessions(labels, fps=25):
    codes = pm.label_codes(labels)
    runs = [(a, b + 1) for a, b in blocks(codes != pm.X)]
    spells = pm.possession_spells(codes, runs)
    return len(spells), (np.mean([b - a for a, b, _ in spells]) / fps if spells else 0.0)


def main():
    import features
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "v1_raw.csv"))
    args = parser.parse_args()

    params = load_params()
    source = "model/params_v1.json" if os.path.isfile(PARAMS_PATH) else "DEFAULT_PARAMS (not tuned yet)"
    labels = predict(features.load(), params)
    write(args.out, labels)
    problem = pm.validate_submission(pm.read_label_csv(args.out), N_FRAMES)
    print("parameters from {}".format(source))
    print("written {}  ({})".format(os.path.relpath(args.out, ROOT), problem or "valid submission"))
    graded = labels != "X"
    print("graded frames: H {:.0%}  A {:.0%}  D {:.0%}   possessions {} (mean {:.1f} s)".format(
        *[(labels[graded] == k).mean() for k in ("H", "A", "D")], *possessions(labels)))
    print("\nscore it:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
