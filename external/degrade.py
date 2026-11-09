"""Make clean external tracking look like the challenge's automatic tracking, before tuning on it.

The DFL tracking is corrected: every player on every frame, the ball always there. The challenge
tracking is automatic: players drop out, and the ball is missing on 41% of graded frames, mostly in
long stretches. Parameters tuned on clean data would expect a ball that is always there, so the
external matches are degraded first, with fixed seeds so the result is reproducible.

Only PUBLISHED facts about the challenge tracking are used (README.md, DATA.md): the ball is missing
on 41% of graded frames, and about 20.5 objects are tracked out of 22. How the gaps relate to dead
ball cannot be learned from the challenge match without its ground truth, so there are two
scenarios and the tuning has to work in both:

  independent   ball gaps fall anywhere, whatever the state of the game
  stoppage      most dead-ball periods lose the ball, the rest of the 41% falls anywhere

Experiment exp02 added two "matched" scenarios, shaped by statistics measured on the challenge
tracking INPUT (external/challenge_profile.py: no ground truth, no video). On the challenge the ball
goes missing far more when the players are slow -- the same shape as the dead-ball rate on the DFL
matches -- and its gaps have their own length distribution. Both are reproduced:

  matched_speed   gaps start more often at low player speed, so that the missing share in every
                  speed bin and the gap lengths are those of the challenge tracking
  matched_dead    a dead-ball period loses the ball as often as the challenge loses it while the
                  players stand still; more gaps are added per speed bin up to the challenge rates

In both, the ball noise is set so that the ball speed the features see has the challenge's
percentiles, instead of the fixed BALL_NOISE_M.
"""

import json
import os

import numpy as np
import pandas as pd

# Params
FPS = 25
BALL_MISSING_TARGET = 0.41          # README.md / DATA.md
PLAYER_VISIBLE_TARGET = 20.5 / 22   # DATA.md: mean tracked objects over graded frames
PLAYER_GAP_MEAN_S = 4.0
GAP_MEDIAN_S = 10.0                 # ball gaps: log-normal lengths, median and spread
GAP_SIGMA = 1.0
GAP_MIN_S, GAP_MAX_S = 0.2, 240.0
STOPPAGE_HIDE_SHARE = 0.8           # stoppage scenario: share of dead-ball periods that lose the ball
STOPPAGE_PAD_S = 2.0                # ... and up to this much either side of them
BALL_NOISE_M = 0.3
SCENARIOS = ("independent", "stoppage")
MATCHED = ("matched_speed", "matched_dead")
EDGE_SD_S = 1.0                     # matched_dead: the gap edges wander around the stoppage edges
CALIBRATION_STEPS = 10
NOISE_GRID_M = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
NOISE_FIT_MATCH = "J03WMX"          # a TUNE match: fits the simulator to the challenge input
HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_PATH = os.path.join(os.path.dirname(HERE), "external_data", "dfl", "cache", "ball_noise.json")


def runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def renewal(rng, n, mean_on, off_sampler):
    """Boolean 'missing' mask: visible stretches (exponential) alternating with gaps."""
    missing = np.zeros(n, dtype=bool)
    t = int(rng.exponential(mean_on))
    while t < n:
        length = off_sampler()
        missing[t:t + length] = True
        t += length + max(1, int(rng.exponential(mean_on)))
    return missing


def gap_sampler(rng):
    def sample():
        seconds = np.clip(rng.lognormal(np.log(GAP_MEDIAN_S), GAP_SIGMA), GAP_MIN_S, GAP_MAX_S)
        return max(1, int(round(seconds * FPS)))
    return sample


def ball_mask(rng, n, graded, base=None, target=BALL_MISSING_TARGET):
    """Add independent gaps to `base` until about `target` of graded frames are missing."""
    base = np.zeros(n, dtype=bool) if base is None else base
    mean_gap = float(np.exp(np.log(GAP_MEDIAN_S) + GAP_SIGMA ** 2 / 2)) * FPS
    have = base[graded].mean()
    if have >= target:
        return base
    # fraction still to add, of the frames not yet missing
    need = (target - have) / (1 - have)
    lo, hi = 1.0, 1e7
    for _ in range(25):  # pick the mean visible stretch that adds `need`, same seed every try
        mid = np.sqrt(lo * hi)
        state = rng.get_state()
        extra = renewal(rng, n, mid, gap_sampler(rng))
        rng.set_state(state)
        added = (extra & ~base)[graded].sum() / max((~base)[graded].sum(), 1)
        if added > need:
            lo = mid
        else:
            hi = mid
    extra = renewal(rng, n, np.sqrt(lo * hi), gap_sampler(rng))
    return base | extra


def player_speed_2s(ps):
    """Mean player speed over a centred 2-second window, as model/features.compute has it."""
    ok = np.isfinite(ps)
    count = ok.sum(axis=1)
    mean = np.where(count > 0, np.where(ok, ps, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    return pd.Series(mean).rolling(2 * FPS, center=True, min_periods=1).mean().values


def ball_speed_seen(bx, by, length, width):
    """Ball speed as model/features.compute has it (median of 5), on frames with a ball."""
    step = np.hypot(np.diff(bx.astype(np.float64)) * length, np.diff(by.astype(np.float64)) * width) * FPS
    speed = pd.Series(np.concatenate(([np.nan], step))).rolling(5, center=True, min_periods=1).median().values
    return speed[np.isfinite(bx)]


def ball_noise_m():
    """Ball noise (m) that gives the clean DFL ball the challenge's ball-speed percentiles."""
    if os.path.isfile(NOISE_PATH):
        with open(NOISE_PATH, encoding="utf-8") as handle:
            return json.load(handle)["ball_noise_m"]
    import challenge_profile
    import dfl
    target = challenge_profile.load()["ball_speed"]
    data = dfl.load(NOISE_FIT_MATCH)
    length, width = float(data["length"]), float(data["width"])
    rng = np.random.RandomState(0)
    errors = []
    for sigma in NOISE_GRID_M:
        bx = data["bx"] + rng.normal(0, sigma / length, len(data["bx"]))
        by = data["by"] + rng.normal(0, sigma / width, len(data["by"]))
        got = np.nanpercentile(ball_speed_seen(bx, by, length, width), [10, 50, 90])
        errors.append(float(np.sum(np.log((got + 0.1) / (target + 0.1)) ** 2)))
    best = NOISE_GRID_M[int(np.argmin(errors))]
    with open(NOISE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"ball_noise_m": best, "grid": NOISE_GRID_M, "errors": errors}, handle, indent=1)
    return best


def hazard_gaps(rng, bins, graded, base, target, lengths):
    """Add gaps to `base`: in each speed bin a gap starts with its own hazard, and its length is drawn
    from the challenge gaps that started in that bin. The hazards are adjusted until every bin is
    missing about its `target` share (same random draws on every adjustment)."""
    n_bins = len(target)

    def share(mask):
        return np.array([mask[graded & (bins == k)].mean() if (graded & (bins == k)).any() else 0.0
                         for k in range(n_bins)])

    def odds(x):
        x = np.clip(x, 1e-3, 0.999)
        return x / (1 - x)

    have = share(base)
    mean_len = np.array([lengths[k].mean() for k in range(n_bins)])
    need = np.clip((target - have) / np.maximum(1 - have, 1e-6), 0, 0.98)
    hazard = need / (1 - need) / mean_len
    state = rng.get_state()
    for step in range(CALIBRATION_STEPS + 1):
        rng.set_state(state)
        u = rng.rand(len(bins))
        pick = rng.rand(len(bins))
        missing = base.copy()
        h = hazard[bins]
        t, n = 0, len(bins)
        while t < n:
            if graded[t] and not missing[t] and u[t] < h[t]:
                choices = lengths[bins[t]]
                length = int(choices[int(pick[t] * len(choices))])
                missing[t:t + length] = True
                t += length
            else:
                t += 1
        missing &= graded
        if step == CALIBRATION_STEPS:
            return missing
        hazard *= np.clip(odds(target) / odds(share(missing)), 0.25, 4.0)
        hazard[target <= have] = 0.0


def degrade(data, scenario, seed, ball="iid"):
    """A degraded copy of an external match (layout of external/dfl.load); ball: see ball_noise()."""
    if scenario not in SCENARIOS + MATCHED:
        raise ValueError(scenario)
    rng = np.random.RandomState(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    n, n_players = out["px"].shape
    graded = out["half"] > 0
    length, width = float(out["length"]), float(out["width"])

    # players drop out in short stretches until about 20.5 of 22 are left
    on_pitch = np.isfinite(out["px"])
    mean_on = PLAYER_GAP_MEAN_S * FPS * PLAYER_VISIBLE_TARGET / (1 - PLAYER_VISIBLE_TARGET)
    for c in range(n_players):
        drop = renewal(rng, n, mean_on, lambda: max(1, int(rng.exponential(PLAYER_GAP_MEAN_S * FPS))))
        drop &= on_pitch[:, c]
        out["px"][drop, c] = np.nan
        out["py"][drop, c] = np.nan
        out["ps"][drop, c] = np.nan

    if scenario in MATCHED:
        return degrade_matched(out, scenario, rng, graded, length, width, ball)

    # the ball
    base = np.zeros(n, dtype=bool)
    if scenario == "stoppage":
        pad = int(STOPPAGE_PAD_S * FPS)
        for s, e in zip(*runs(out["label"] == 2)):
            if rng.rand() < STOPPAGE_HIDE_SHARE:
                base[max(s - rng.randint(0, pad + 1), 0):min(e + rng.randint(0, pad + 1), n)] = True
    missing = ball_mask(rng, n, graded, base)
    return ball_noise(out, rng, missing, BALL_NOISE_M, ball, length, width)


def ball_noise(out, rng, missing, sigma, ball, length, width):
    """Noise on the seen ball, then the gaps. ball="iid": independent noise of sigma metres per
    frame (exp02). ball="drift": external/ball_model.py, a slowly drifting error then smoothing,
    fitted to how the challenge tracking moves the ball (exp03). The iid draws are made in both
    cases, so the gaps and players of a seed are the same whichever ball is chosen."""
    n = len(out["bx"])
    clean_x, clean_y = out["bx"], out["by"]
    out["bx"] = clean_x + rng.normal(0, sigma / length, n).astype(np.float32)
    out["by"] = clean_y + rng.normal(0, sigma / width, n).astype(np.float32)
    if ball == "drift":
        import ball_model
        out["bx"], out["by"] = ball_model.apply(clean_x, clean_y, length, width,
                                                np.random.RandomState(rng.randint(2 ** 31)), ball_model.params())
    elif ball != "iid":
        raise ValueError(ball)
    out["bx"][missing] = np.nan
    out["by"][missing] = np.nan
    return out


def degrade_matched(out, scenario, rng, graded, length, width, ball="iid"):
    """The ball part of the matched scenarios; players were already dropped by degrade()."""
    import challenge_profile as cp
    profile = cp.load()
    n = len(out["bx"])
    bins = cp.speed_bin(player_speed_2s(out["ps"]))
    pool = profile["gap_frames"]
    lengths = []
    for k in range(len(profile["miss_rate"])):
        own = pool[profile["gap_start_bin"] == k]
        lengths.append(own if len(own) >= 20 else pool)
    target = np.nan_to_num(profile["miss_rate"], nan=BALL_MISSING_TARGET)
    base = np.zeros(n, dtype=bool)
    if scenario == "matched_dead":
        share = float(profile["low_speed_miss"])
        sd = EDGE_SD_S * FPS
        for s, e in zip(*runs((out["label"] == 2) & graded)):
            if rng.rand() < share:
                a = int(round(s + rng.normal(0, sd)))
                b = int(round(e + rng.normal(0, sd)))
                if b > a:
                    base[max(a, 0):min(b, n)] = True
    missing = hazard_gaps(rng, bins, graded, base & graded, target, lengths)
    return ball_noise(out, rng, missing, ball_noise_m(), ball, length, width)
