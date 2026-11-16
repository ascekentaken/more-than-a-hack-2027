"""Touch and control signals (experiment exp03), on top of the v2 table, for any match in the ATD layout.

    import features_v3
    table = features_v3.challenge()                                              # cached
    table = features_v3.compute(px, py, ps, bx, by, team, half, length, width)   # any match

v2 knows who is NEAR the ball. A short possession is decided by who TOUCHES it: the ball changes
direction or speed sharply while a player is within reach. And the challenge counts a possession
only when the ball is CONTROLLED, which on tracking looks like the same player staying with the ball
for a while. This adds:

  touches         a frame where the seen ball turns by more than TURN_DEG or changes speed by more
                  than DSPEED_MS (over KICK_FRAMES either side) with a player within TOUCH_M; one per
                  burst, at the closest approach; the touch belongs to that player's team
  per team        time since its last touch and until its next, touches in the last / next 2 s,
                  which team touched last and touches next
  control         how long the same player has been within CONTROL_M of the ball, in a row; per team,
                  the longest such run in the last and next 2 s
  ball motion     the turn angle and the speed change themselves

Same conventions as features_v2: home / away pairs, compute(..., team=other_side(team)) is the
other side's view, times in frames capped at CAP_FRAMES.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis", "model"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import features_v2  # noqa: E402

# Params
FPS = 25
CAP_FRAMES = features_v2.CAP_FRAMES
KICK_FRAMES = 4          # velocity before / after a frame, over this many frames
MOVING_MS = 1.0          # below this the direction of the ball means nothing
TURN_DEG = 25.0
DSPEED_MS = 2.5
TOUCH_M = 1.5
CONTROL_M = 2.0
WINDOW = 2 * FPS
TABLE_PATH = os.path.join(ROOT, "analysis_out", "features_v3.npz")


def shift(values, k):
    """values[t + k], NaN outside."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out


def runs_of(key):
    """For an integer key per frame (-1 = none): start index and length of the run each frame is in."""
    n = len(key)
    change = np.ones(n, dtype=bool)
    change[1:] = key[1:] != key[:-1]
    start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    ends = np.flatnonzero(np.concatenate((change[1:], [True])))
    length_by_start = np.zeros(n, dtype=np.int64)
    starts = np.flatnonzero(change)
    length_by_start[starts] = ends - starts + 1
    return start, length_by_start[start]


def touch_columns(px, py, bx, by, team, length, width):
    n = len(bx)
    home, away = team == 0, team == 1
    x, y = bx.astype(np.float64) * length, by.astype(np.float64) * width
    k = KICK_FRAMES
    vbx, vby = (x - shift(x, -k)) * FPS / k, (y - shift(y, -k)) * FPS / k
    vax, vay = (shift(x, k) - x) * FPS / k, (shift(y, k) - y) * FPS / k
    before, after = np.hypot(vbx, vby), np.hypot(vax, vay)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (vbx * vax + vby * vay) / (before * after)
    turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    turn = np.where((before > MOVING_MS) & (after > MOVING_MS), turn, np.nan)
    dspeed = after - before

    dist = np.sqrt(((px - bx[:, None]) * length) ** 2 + ((py - by[:, None]) * width) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(axis=1)
    d_min = dist[np.arange(n), nearest]
    near_team = np.where(np.isfinite(d_min), team[nearest], -1)

    kick = (np.nan_to_num(turn) > TURN_DEG) | (np.abs(np.nan_to_num(dspeed)) > DSPEED_MS)
    candidate = kick & (d_min < TOUCH_M)
    # one touch per burst of candidate frames: the frame of closest approach
    touch = np.zeros(n, dtype=bool)
    edges = np.diff(np.concatenate(([0], candidate.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        touch[s + int(np.argmin(d_min[s:e]))] = True
    touch_team = np.where(touch, near_team, -1)
    touch_home, touch_away = touch & (touch_team == 0), touch & (touch_team == 1)

    out = {"ball_turn": turn, "ball_dspeed": dspeed, "touch": touch.astype(np.int8)}
    out["since_touch_home"], out["since_touch_away"] = features_v2.since(touch_home), features_v2.since(touch_away)
    out["until_touch_home"], out["until_touch_away"] = features_v2.until(touch_home), features_v2.until(touch_away)
    out["last_touch_team"] = features_v2.last_value(touch, touch_team).astype(np.int8)
    out["next_touch_team"] = features_v2.next_value(touch, touch_team).astype(np.int8)
    for name, mask in (("home", touch_home), ("away", touch_away)):
        m = mask.astype(float)
        out["touches_{}_past_2s".format(name)] = pd.Series(m).rolling(WINDOW, min_periods=1).sum().values
        out["touches_{}_next_2s".format(name)] = pd.Series(m[::-1]).rolling(WINDOW, min_periods=1).sum().values[::-1]

    # control: the same player (slot) stays within CONTROL_M, frame after frame
    controlling = np.where(d_min < CONTROL_M, nearest, -1)
    _, run_length = runs_of(controlling)
    run_length = np.where(controlling >= 0, run_length, 0)
    control_team = np.where(controlling >= 0, team[np.maximum(controlling, 0)], -1)
    out["control_len"] = np.minimum(run_length, CAP_FRAMES).astype(np.float32)
    out["control_team"] = control_team.astype(np.int8)
    for name, side in (("home", 0), ("away", 1)):
        own = np.where(control_team == side, run_length, 0).astype(float)
        out["control_{}_past_2s".format(name)] = pd.Series(own).rolling(WINDOW, min_periods=1).max().values
        out["control_{}_next_2s".format(name)] = pd.Series(own[::-1]).rolling(WINDOW, min_periods=1).max().values[::-1]
    return out


def compute(px, py, ps, bx, by, team, half, length, width):
    out = features_v2.compute(px, py, ps, bx, by, team, half, length, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out.update(touch_columns(px, py, bx, by, team, length, width))
    return out


def feature_names(table):
    return features_v2.feature_names(table)


def challenge(rebuild=False, data_dir=None, swap=False):
    """The challenge table with the touch signals; swap=True is the other side's view."""
    path = TABLE_PATH.replace(".npz", "_swap.npz") if swap else TABLE_PATH
    if rebuild or not os.path.isfile(path):
        px, py, ps, bx, by, team, half, length, width = features_v2.challenge_inputs(data_dir)
        columns = compute(px, py, ps, bx, by, features_v2.other_side(team) if swap else team, half, length, width)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        return pd.DataFrame({name: saved[name] for name in saved.files})
