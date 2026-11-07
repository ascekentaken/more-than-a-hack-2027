"""The richer per-frame signal table of model v2 (experiment exp02), for any match in the ATD layout.

    import features_v2
    table = features_v2.challenge()                       # the challenge match, cached
    table = features_v2.compute(px, py, ps, bx, by, team, half, length, width)   # any match

It starts from model/features.compute (the v1 signals) and adds what v1 could not see:

  context in time   player speed over 1 / 4 / 8 s windows, and the last 2 s against the next 2 s,
                    so a restart or a stoppage shows up at the frame where it happens
  who touched last  for each team, how long since its nearest player was within 1.5 m of a seen
                    ball, and how long until it is next; which team that was. Possession passes on
                    controlled contact, so these carry the team through passes and short gaps
  ball visibility   how long since / until the ball was or will be missing, and outside the lines
  team shape        spread and centroid of each team along the pitch, oriented so that "forward"
                    means towards the goal the home team attacks
  near the ball     how many players of each team within 3 m and within 8 m

Every signal comes in a home / away pair or is symmetric, and compute(..., team=1 - team) gives the
same table seen from the other side; training uses both to stay blind to which team is home.
Distances in metres, times in frames (25 per second), capped at CAP_FRAMES.
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

import features  # noqa: E402

# Params
FPS = 25
CAP_FRAMES = 1500
CLOSE_M = 1.5
NEAR_M, AROUND_M = 3.0, 8.0
TABLE_PATH = os.path.join(ROOT, "analysis_out", "features_v2.npz")
# the v1 columns that are not signals for the classifier
NOT_FEATURES = ("frame", "half", "nearest_team", "ball_x", "ball_y", "tag_home", "tag_away", "setpiece")


def rolling(values, window, center=True):
    return pd.Series(values).rolling(window, center=center, min_periods=1).mean().values


def future(values, window):
    """Mean over the next `window` frames, current included."""
    return pd.Series(values[::-1]).rolling(window, min_periods=1).mean().values[::-1]


def since(mask, cap=CAP_FRAMES):
    """Frames since `mask` was last true (0 on true frames), capped."""
    n = len(mask)
    last = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(last)
    return np.minimum(np.arange(n) - last, cap).astype(np.float32)


def until(mask, cap=CAP_FRAMES):
    return since(mask[::-1], cap)[::-1]


def last_value(mask, values, fill=-1):
    """values at the most recent frame where mask is true (fill before the first)."""
    n = len(mask)
    index = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    return np.where(index >= 0, values[np.maximum(index, 0)], fill)


def next_value(mask, values, fill=-1):
    return last_value(mask[::-1], values[::-1], fill)[::-1]


def compute(px, py, ps, bx, by, team, half, length, width):
    with warnings.catch_warnings():  # empty slices (no player of a team tracked) give NaN, as intended
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(px, py, ps, bx, by, team, half, length, width)


def _compute(px, py, ps, bx, by, team, half, length, width):
    out = features.compute(px, py, ps, bx, by, team, half, length, width)
    n = len(bx)
    home, away = team == 0, team == 1
    seen = out["ball_seen"]
    ball_x, ball_y = out["ball_x"], out["ball_y"]

    # orientation: on average the away centroid is nearer the goal the home team attacks
    x = px * length
    y = py * width
    with np.errstate(invalid="ignore"):
        cx_home = np.nanmean(np.where(home, x, np.nan), axis=1)
        cx_away = np.nanmean(np.where(away, x, np.nan), axis=1)
        sx_home = np.nanstd(np.where(home, x, np.nan), axis=1)
        sx_away = np.nanstd(np.where(away, x, np.nan), axis=1)
        sy_home = np.nanstd(np.where(home, y, np.nan), axis=1)
        sy_away = np.nanstd(np.where(away, y, np.nan), axis=1)
    direction = np.ones(n)
    for h in (1, 2):
        rows = half == h
        if rows.any():
            direction[rows] = 1.0 if np.nanmedian((cx_away - cx_home)[rows]) >= 0 else -1.0
    out["ball_fwd_home"] = (ball_x - length / 2) * direction
    out["ball_wide"] = np.abs(ball_y - width / 2)
    out["ball_to_touch"] = np.minimum(ball_y, width - ball_y)
    out["ball_to_goalline"] = np.minimum(ball_x, length - ball_x)
    out["centroid_fwd_home"] = (cx_home - length / 2) * direction
    out["centroid_fwd_away"] = (cx_away - length / 2) * direction
    out["centroid_gap"] = (cx_away - cx_home) * direction
    out["spread_x_home"], out["spread_x_away"] = sx_home, sx_away
    out["spread_y_home"], out["spread_y_away"] = sy_home, sy_away

    # near the ball
    dist = np.sqrt((x - ball_x[:, None]) ** 2 + (y - ball_y[:, None]) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    for name, radius in (("near", NEAR_M), ("around", AROUND_M)):
        out["{}_home".format(name)] = (dist[:, home] < radius).sum(axis=1).astype(np.int8)
        out["{}_away".format(name)] = (dist[:, away] < radius).sum(axis=1).astype(np.int8)
    d_home = np.nan_to_num(out["d_home"], nan=np.inf)
    d_away = np.nan_to_num(out["d_away"], nan=np.inf)
    out["d_diff"] = np.clip(np.where(np.isfinite(d_home) & np.isfinite(d_away), d_away - d_home, np.nan), -15, 15)
    out["d_min"] = np.where(seen, np.minimum(d_home, d_away), np.nan)

    # who touched last / next
    close_home = seen & (d_home < CLOSE_M) & (d_home <= d_away)
    close_away = seen & (d_away < CLOSE_M) & (d_away < d_home)
    out["since_close_home"], out["since_close_away"] = since(close_home), since(close_away)
    out["until_close_home"], out["until_close_away"] = until(close_home), until(close_away)
    close_team = np.where(close_home, 0, np.where(close_away, 1, -1))
    close_any = close_team >= 0
    out["last_close_team"] = last_value(close_any, close_team).astype(np.int8)
    out["next_close_team"] = next_value(close_any, close_team).astype(np.int8)
    nearest = out["nearest_team"]
    for name, window in (("1s", FPS), ("3s", 3 * FPS)):
        h = np.where(seen, (nearest == 0).astype(float), np.nan)
        out["home_near_past_" + name] = pd.Series(h).rolling(window, min_periods=1).mean().values
        out["home_near_next_" + name] = pd.Series(h[::-1]).rolling(window, min_periods=1).mean().values[::-1]

    # time context
    speed = out["speed_mean"]
    out["speed_1s"] = rolling(speed, FPS)
    out["speed_4s"] = rolling(speed, 4 * FPS)
    out["speed_8s"] = rolling(speed, 8 * FPS)
    out["speed_past_2s"] = rolling(speed, 2 * FPS, center=False)
    out["speed_next_2s"] = future(speed, 2 * FPS)
    out["speed_change"] = out["speed_next_2s"] - out["speed_past_2s"]
    ok = np.isfinite(ps)
    top = np.sort(np.where(ok, ps, -1.0), axis=1)[:, -3:]
    top = np.where(top >= 0, top, np.nan)
    with np.errstate(invalid="ignore"):
        out["speed_top3"] = rolling(np.nanmean(top, axis=1), 2 * FPS)
        out["speed_sd"] = rolling(np.nanstd(np.where(ok, ps, np.nan), axis=1), 2 * FPS)
    ball_speed = out["ball_speed"]
    out["ball_speed_1s"] = pd.Series(ball_speed).rolling(FPS, center=True, min_periods=1).mean().values
    out["ball_speed_max_1s"] = pd.Series(ball_speed).rolling(FPS, center=True, min_periods=1).max().values

    # ball visibility and the lines
    out["since_missing"], out["until_missing"] = since(~seen), until(~seen)
    out["since_seen"], out["until_seen"] = since(seen), until(seen)
    outside = out["ball_outside"]
    out["since_outside"], out["until_outside"] = since(outside), until(outside)
    out["seen_share_4s"] = rolling(seen.astype(float), 4 * FPS)
    return out


SPEED_COLUMNS = ("speed_mean", "speed_mean_2s", "speed_1s", "speed_4s", "speed_8s", "speed_past_2s",
                 "speed_next_2s", "speed_change", "speed_top3", "speed_sd")


def relative_speeds(table):
    """Player speeds divided by the match's median player speed inside the halves.

    One match is played faster than another, and the challenge tracking reads about 8% slower than
    the DFL matches (input check in exp02). A ratio to the match's own median keeps the meaning of
    "slower than usual" whatever the match and whatever the tracking system.
    """
    out = table.copy()
    median = float(np.nanmedian(table["speed_mean"].values[table["half"].values > 0]))
    for column in SPEED_COLUMNS:
        out[column] = table[column] / median
    return out


def feature_names(table):
    return [c for c in table.columns if c not in NOT_FEATURES]


def challenge_inputs(data_dir=None):
    """The challenge tracking as arrays, through the same loader as model/features.build."""
    import baseline as bl
    import games
    from analyze_data import load_atd
    meta, raw = games.require(["atd_meta", "atd_raw"], features.GAME, data_dir)
    atd = load_atd(raw, features.N_FRAMES, os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz"), False)
    half = np.zeros(features.N_FRAMES, dtype=np.int8)
    for h, first, last in games.half_video_blocks(features.GAME):
        half[first:last + 1] = h
    return (atd["px"], atd["py"], atd["ps"], atd["bx"], atd["by"], bl.team_of_each_track(meta), half,
            features.LENGTH, features.WIDTH)


def other_side(team):
    """Home slots become away and the other way round; anything else stays as it is."""
    return np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(team.dtype)


def challenge(rebuild=False, data_dir=None, swap=False):
    """The challenge table; swap=True is the same match seen from the other side."""
    path = TABLE_PATH.replace(".npz", "_swap.npz") if swap else TABLE_PATH
    if rebuild or not os.path.isfile(path):
        px, py, ps, bx, by, team, half, length, width = challenge_inputs(data_dir)
        columns = compute(px, py, ps, bx, by, other_side(team) if swap else team, half, length, width)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **columns)
    with np.load(path) as saved:
        return pd.DataFrame({name: saved[name] for name in saved.files})
