"""What the challenge tracking would show during dead ball, reproduced on the external matches (exp04).

    python external/dead_model.py      # fits and caches external_data/dfl/cache/dead_model.json

The DFL ball is FROZEN during dead ball: it stays exactly where it went out (0 m of movement in every
stoppage, often outside the pitch or on a line) and jumps to the restart spot when play resumes. That
is a convention of the corrected data, not what a camera sees, and it hands the model an easy cue: a
still ball by a line means dead ball. On the challenge tracking INPUT (no ground truth, no video) a
seen ball is outside the pitch 3.5% of the time against 6-16% on the degraded DFL matches, within 2 m
of a line 7% against 12-29%, and a ball that comes back after a gap is by a line 12% of the time
against 36-46%. During live play the DFL matches already look like the challenge (6-7% by a line).

So every DFL stoppage is replaced by:

  1. roll out: the ball keeps its last velocity, slowing with a time constant of ROLL_TAU_S, for an
     exponential time of mean ROLL_S seconds;
  2. gone: the ball is not seen -- except in a share CARRIED of the stoppages, where the tracker
     follows it as it is brought from where it stopped to the restart spot (a straight line at
     constant speed);
  3. placed: for an exponential time of mean PLACED_S seconds before the restart (PLACED_S times
     EDGE_FACTOR when the restart spot is within EDGE_M of a line: throw-ins, corners) the ball lies
     at the restart spot;
  4. picked up late: after the restart the ball stays unseen for an exponential time of mean
     PICKUP_S seconds when the restart spot is by a line (PICKUP_S times INSIDE_FACTOR elsewhere):
     on the challenge a ball that comes back after a gap is rarely by a line or still, so the
     tracker finds it once it has left the thrower or the corner.

The six numbers are chosen on a grid so that statistics of the seen ball (outside the pitch, by a
line, still, seen while the players are slow, where it comes back after a gap) on a degraded TUNE
match match the challenge's. The fit uses FIT_MATCH and the challenge input only.
"""

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "analysis", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

# Params
FPS = 25
D = 2
PATH = os.path.join(ROOT, "external_data", "dfl", "cache", "dead_model.json")
FIT_MATCH = "J03WMX"
FIT_SEED = 2027
ROLL_TAU_S = 1.0
EDGE_M = 2.0
SLOW_MS = 0.8
STILL_MS = 0.5
GAP_S = 2.0
GRID = {"roll_s": [0.5], "placed_s": [1.0, 4.0], "edge_factor": [0.0],
        "carried": [0.35, 0.5], "pickup_s": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0], "inside_factor": [0.0, 1.0]}
# a first, wider grid without the late pick-up (roll 0-2 s, placed 0-4 s, edge factor 0-1, carried 0-0.5)
# chose roll 2, placed 4, edge factor 0, carried 0.2; the ball still came back by a line 38% of the
# time against 12%, hence step 4. A second grid (roll 0.5 / 2, placed 0 / 1 / 4, carried 0.2 / 0.35,
# pick-up 0-3 s, inside factor 0 / 1) put the best pick-up on its edge (3 s); this third grid widens it
# around the second one's best.
SIGNATURE = ("outside | seen", "by a line | seen", "still | seen", "still | seen, slow", "seen | slow",
             "back by a line", "back still", "back slow")


def runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def edge_distance(bx, by, length, width):
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    return np.minimum.reduce([np.abs(x), np.abs(length - x), np.abs(y), np.abs(width - y)])


def apply(bx, by, label, half, length, width, rng, params):
    """Clean ball with the stoppages replaced, and the frames where the ball is not seen."""
    bx, by = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    n = len(bx)
    hidden = np.zeros(n, dtype=bool)
    decay = np.exp(-1.0 / (ROLL_TAU_S * FPS))
    for s, e in zip(*runs((label == D) & (half > 0))):
        if e >= n or not np.isfinite(bx[e]):
            hidden[s:e] = True
            continue
        span = e - s
        roll = min(int(rng.exponential(params["roll_s"]) * FPS) if params["roll_s"] > 0 else 0, span)
        if roll and s >= 5 and np.isfinite(bx[s - 5]):
            vx, vy = (bx[s - 1] - bx[s - 5]) / 4, (by[s - 1] - by[s - 5]) / 4
            k = np.arange(1, roll + 1)
            travel = decay * (1 - decay ** k) / (1 - decay)
            bx[s:s + roll], by[s:s + roll] = bx[s - 1] + vx * travel, by[s - 1] + vy * travel
        else:
            roll = 0
        by_line = edge_distance(bx[e:e + 1], by[e:e + 1], length, width)[0] < EDGE_M
        mean = params["placed_s"] * (params["edge_factor"] if by_line else 1.0)
        placed = min(int(rng.exponential(mean) * FPS) if mean > 0 else 0, span - roll)
        follow = rng.rand() < params.get("carried", 0.0)
        middle = e - placed - (s + roll)
        if follow and middle > 0:
            start_x, start_y = bx[s + roll - 1] if roll else bx[s], by[s + roll - 1] if roll else by[s]
            f = np.arange(1, middle + 1) / middle
            bx[s + roll:e - placed] = start_x + (bx[e] - start_x) * f
            by[s + roll:e - placed] = start_y + (by[e] - start_y) * f
            bx[e - placed:e], by[e - placed:e] = bx[e], by[e]
        else:
            bx[s + roll:e], by[s + roll:e] = bx[e], by[e]
            hidden[s + roll:e - placed] = True
        late = params.get("pickup_s", 0.0) * (1.0 if by_line else params.get("inside_factor", 1.0))
        if late > 0:
            hidden[e:min(e + int(rng.exponential(late) * FPS), n)] = True
    return bx.astype(np.float32), by.astype(np.float32), hidden


def signature(px, py, ps, bx, by, half, length, width):
    """Statistics of the seen ball; no labels."""
    import degrade
    g = half > 0
    seen = np.isfinite(bx)
    slow = degrade.player_speed_2s(ps) < SLOW_MS
    outside = seen & ((bx < -0.5 / length) | (bx > 1 + 0.5 / length) | (by < -0.5 / width) | (by > 1 + 0.5 / width))
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
    still = seen & (pd.Series(step).rolling(5, center=True, min_periods=1).median().values < STILL_MS)
    line = edge_distance(bx, by, length, width) < EDGE_M
    s, e = runs(~seen & g)
    back = e[((e - s) >= GAP_S * FPS) & (e < len(bx))]
    back = back[seen[back]]
    return np.array([outside[g & seen].mean(), line[g & seen].mean(), still[g & seen].mean(),
                     still[g & seen & slow].mean(), seen[g & slow].mean(),
                     line[back].mean(), still[back].mean(), slow[back].mean()])


def challenge_signature():
    import features_v2
    px, py, ps, bx, by, team, half, length, width = features_v2.challenge_inputs()
    return signature(px, py, ps, bx, by, half, length, width)


def fit():
    import degrade
    import dfl
    target = challenge_signature()
    data = dfl.load(FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for values in itertools.product(*GRID.values()):
            params = dict(zip(GRID, values))
            got = []
            for scenario in degrade.MATCHED:
                out = degrade.degrade(data, scenario, FIT_SEED, ball="drift", dead=params)
                got.append(signature(out["px"], out["py"], out["ps"], out["bx"], out["by"], out["half"],
                                     length, width))
            got = np.mean(got, axis=0)
            error = float(np.mean(np.log((got + 0.02) / (target + 0.02)) ** 2))
            rows.append((error, params, got))
    rows.sort(key=lambda r: r[0])
    frozen = []
    for scenario in degrade.MATCHED:
        out = degrade.degrade(data, scenario, FIT_SEED, ball="drift")
        frozen.append(signature(out["px"], out["py"], out["ps"], out["bx"], out["by"], out["half"], length, width))
    payload = {"params": rows[0][1], "error": rows[0][0], "signature": list(SIGNATURE),
               "target": target.round(3).tolist(), "got": rows[0][2].round(3).tolist(),
               "frozen": np.mean(frozen, axis=0).round(3).tolist(),
               "runner_up": [{"params": p, "error": e} for e, p, _ in rows[1:6]]}
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return payload


def params():
    if not os.path.isfile(PATH):
        fit()
    with open(PATH, encoding="utf-8") as handle:
        return json.load(handle)["params"]


if __name__ == "__main__":
    p = fit()
    print("best", p["params"], "error {:.4f}".format(p["error"]))
    for name, t, g, f in zip(p["signature"], p["target"], p["got"], p["frozen"]):
        print("  {:<20} challenge {:.3f}   new {:.3f}   frozen {:.3f}".format(name, t, g, f))
    for r in p["runner_up"]:
        print("  next", r)
