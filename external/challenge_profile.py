"""How the challenge's automatic tracking behaves, measured on its INPUT only, to shape the simulator.

    python external/challenge_profile.py      # computes and caches external_data/challenge_profile.npz

The rules allow "running automatic checks using only the provided data" and using the automatic
tracking as input; what they forbid is training or tuning on the ground truth, the video or hand
labels. This module reads only the tracking (through model/features.load, which never opens the
ground truth) and measures what external/degrade.py needs to make the external matches look like
it:

  miss_rate     share of graded frames with no ball, per bin of mean player speed (2 s window)
  gap_frames    length of every ball gap inside the halves, and the speed bin where it started
  ball_speed    10th / 50th / 90th percentile of the ball speed the features see
  low_speed_miss  share of frames with no ball while the players are almost still

Numbers derived from the challenge data are not committed: the profile lives in external_data/
(gitignored) and is recomputed from the local data by anyone who has it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "model", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

# Params
PROFILE_PATH = os.path.join(ROOT, "external_data", "challenge_profile.npz")
SPEED_BINS = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0])  # m/s, inner edges
LOW_SPEED = 0.8  # m/s: below this the players are standing, the ball is almost always dead


def speed_bin(speed_2s):
    """Bin index 0..len(SPEED_BINS) of the 2-second mean player speed; NaN goes to the slowest."""
    return np.digitize(np.nan_to_num(speed_2s, nan=0.0), SPEED_BINS)


def runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def measure(table):
    graded = table["half"].values > 0
    seen = table["ball_seen"].values
    b = speed_bin(table["speed_mean_2s"].values)
    miss_rate = np.array([(~seen[graded & (b == k)]).mean() if (graded & (b == k)).any() else np.nan
                          for k in range(len(SPEED_BINS) + 1)])
    starts, ends = runs(~seen & graded)
    speed = table["ball_speed"].values[graded & seen]
    slow = graded & (np.nan_to_num(table["speed_mean_2s"].values, nan=0.0) < LOW_SPEED)
    return {"miss_rate": miss_rate, "gap_frames": (ends - starts).astype(np.int32),
            "gap_start_bin": b[starts].astype(np.int8),
            "ball_speed": np.nanpercentile(speed, [10, 50, 90]),
            "low_speed_miss": np.array((~seen[slow]).mean()),
            "players": np.array((table["n_home"] + table["n_away"]).values[graded].mean())}


def load(rebuild=False):
    if rebuild or not os.path.isfile(PROFILE_PATH):
        import features
        profile = measure(features.load())
        os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
        np.savez(PROFILE_PATH, **profile)
    with np.load(PROFILE_PATH) as saved:
        return {k: saved[k] for k in saved.files}


if __name__ == "__main__":
    p = load(rebuild=True)
    print("miss rate per speed bin:", np.round(p["miss_rate"], 2))
    print("gaps: {}  median {:.1f} s  q90 {:.1f} s".format(
        len(p["gap_frames"]), np.median(p["gap_frames"]) / 25, np.percentile(p["gap_frames"], 90) / 25))
    print("ball speed q10/50/90:", np.round(p["ball_speed"], 2), " low-speed miss", float(p["low_speed_miss"]))
