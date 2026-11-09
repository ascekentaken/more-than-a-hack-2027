"""How the challenge tracking moves the ball, reproduced on the external matches (experiment exp03).

    python external/ball_model.py      # fits and caches external_data/dfl/cache/ball_model.json

exp02 matched the ball's speed percentiles with independent noise on every frame. Looking closer at
the challenge tracking INPUT (no ground truth, no video), its ball is smooth -- median acceleration
6 m/s^2 -- but never quite still: even the slowest tenth of frames moves at 1 m/s. Independent noise
cannot do both (it gives 74 m/s^2), and it invents sharp turns, which the touch signals of exp03
would learn to trust. That is the signature of a smoothed position with an error that drifts
slowly. So the external ball gets:

  1. a drifting error: on each axis an AR(1) process with spread SIGMA metres and time constant TAU
     seconds;
  2. then a centred moving average over WINDOW frames.

The three numbers are chosen on a grid so that the ball speed, acceleration, turn angle and speed
change percentiles of the external ball match the challenge's. The fit uses NOISE_FIT_MATCH, a TUNE
match, and the challenge input only.
"""

import itertools
import json
import os
import warnings
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "analysis", "external"):
    sys.path.insert(0, os.path.join(ROOT, sub))

# Params
FPS = 25
PATH = os.path.join(ROOT, "external_data", "dfl", "cache", "ball_model.json")
FIT_MATCH = "J03WMX"
FIT_FRAMES = 40000                 # the first ~27 minutes are enough and keep the grid fast
GRID = {"window": [1, 5, 9, 13, 17, 21, 25, 31, 41], "sigma": [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8],
        "tau": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]}
KICK_FRAMES = 4
MOVING_MS = 1.0


def drift(rng, n, sigma, tau):
    """AR(1) noise, stationary spread sigma, time constant tau seconds."""
    rho = np.exp(-1.0 / (tau * FPS))
    shocks = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + shocks[t]
    return out


def apply(bx, by, length, width, rng, params):
    """The ball as the challenge tracking would see it (normalised coordinates in and out)."""
    n = len(bx)
    x = bx.astype(np.float64) + drift(rng, n, params["sigma"], params["tau"]) / length
    y = by.astype(np.float64) + drift(rng, n, params["sigma"], params["tau"]) / width
    w = int(params["window"])
    if w > 1:
        x = pd.Series(x).rolling(w, center=True, min_periods=1).mean().values
        y = pd.Series(y).rolling(w, center=True, min_periods=1).mean().values
    keep = np.isfinite(bx)
    return np.where(keep, x, np.nan).astype(np.float32), np.where(keep, y, np.nan).astype(np.float32)


def nearest_player(bx, by, px, py, length, width):
    """Distance in metres from the ball to the nearest tracked player, per frame."""
    dist = np.sqrt(((px - bx[:, None]) * length) ** 2 + ((py - by[:, None]) * width) ** 2)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():  # frames with nobody tracked: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.where(np.isfinite(dist), dist, np.nan), axis=1)


def signature(bx, by, length, width, graded, px, py):
    """Percentiles that describe how a tracked ball moves, and how far it sits from the players
    (the position error shows up there: a ball at a player's feet reads a few tens of cm away)."""
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    vx, vy = np.diff(x) * FPS, np.diff(y) * FPS
    speed = np.hypot(vx, vy)
    acc = np.hypot(np.diff(vx), np.diff(vy)) * FPS
    k = KICK_FRAMES
    bxv, byv = (x[k:-k] - x[:-2 * k]) * FPS / k, (y[k:-k] - y[:-2 * k]) * FPS / k
    axv, ayv = (x[2 * k:] - x[k:-k]) * FPS / k, (y[2 * k:] - y[k:-k]) * FPS / k
    before, after = np.hypot(bxv, byv), np.hypot(axv, ayv)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.degrees(np.arccos(np.clip((bxv * axv + byv * ayv) / (before * after), -1, 1)))
    turn = np.where((before > MOVING_MS) & (after > MOVING_MS), turn, np.nan)
    dspeed = np.abs(after - before)
    g1, g2, gk = graded[1:], graded[2:], graded[k:-k]
    with np.errstate(invalid="ignore"):
        d_min = nearest_player(bx, by, px, py, length, width)
    return np.concatenate([np.nanpercentile(speed[g1], [10, 50, 90]), np.nanpercentile(acc[g2], [50, 90, 99]),
                           np.nanpercentile(turn[gk], [50, 90]), np.nanpercentile(dspeed[gk], [50, 90]),
                           np.nanpercentile(d_min[graded], [10, 25, 50])])


def challenge_signature():
    import features_v2
    px, py, ps, bx, by, team, half, length, width = features_v2.challenge_inputs()
    return signature(bx, by, length, width, half > 0, px, py)


def fit():
    import dfl
    target = challenge_signature()
    data = dfl.load(FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    bx, by, graded = data["bx"][:FIT_FRAMES], data["by"][:FIT_FRAMES], data["half"][:FIT_FRAMES] > 0
    px, py = data["px"][:FIT_FRAMES], data["py"][:FIT_FRAMES]
    rows = []
    for window, sigma, tau in itertools.product(*GRID.values()):
        params = {"window": window, "sigma": sigma, "tau": tau}
        x, y = apply(bx, by, length, width, np.random.RandomState(0), params)
        got = signature(x, y, length, width, graded, px, py)
        error = float(np.mean(np.log((got + 0.5) / (target + 0.5)) ** 2))
        rows.append((error, params, got))
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    payload = {"params": best[1], "error": best[0], "target": target.round(2).tolist(),
               "got": best[2].round(2).tolist(),
               "signature": "speed q10/50/90, acceleration q50/90/99, turn q50/90, |speed change| q50/90, "
                            "nearest player q10/25/50",
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
    print("  ", p["signature"])
    print("  challenge", p["target"])
    print("  external ", p["got"])
    for r in p["runner_up"]:
        print("  next", r)
