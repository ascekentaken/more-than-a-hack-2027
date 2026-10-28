"""Choose, once and reproducibly, which stretches of the match the team annotates by hand.

    python annotation/make_plan.py            # writes annotation/labels/plan.json
    python annotation/make_plan.py --force    # overwrite an existing plan (annotations are
                                              # keyed by clip id, so only do this before starting)

The plan is a list of 2-minute clips:

  * calib    the public 5 minutes. Everyone annotates them first and compares with the real
             labels, to learn what the organisers mean by possession. Never used for tuning.
  * gappy    clips where the tracking loses the ball for a large share of frames -- the case the
             public window barely shows, so the case we know least about.
  * other    clips drawn at random from the rest.

Each clip is then put in one of two sets:

  * tune     used to set thresholds and train models
  * test     looked at only to check the final system; a score on these is the honest one

Clips never overlap the public window, never cross halftime, and the stretch of the second half
before its kickoff (all dead ball) is left out. The draw uses a fixed seed: the same command
gives the same plan on every machine.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
from analyze_data import load_atd  # noqa: E402
import common  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames("tactical", GAME)
PUBLIC_WINDOW = (11185, 18684)
SECOND_HALF_KICKOFF = 113742
SEED = 2027
GAPPY_QUANTILE = 0.67  # clips above this share of missing ball count as "gappy"


def candidates(clip_frames):
    """(half, first_frame) for every non-overlapping clip that fits inside a half."""
    out = []
    for half, first, last in games.half_video_blocks(GAME):
        start = first if half == 1 else SECOND_HALF_KICKOFF
        s = start
        while s + clip_frames - 1 <= last:
            overlaps_public = s <= PUBLIC_WINDOW[1] and s + clip_frames - 1 >= PUBLIC_WINDOW[0]
            if not overlaps_public:
                out.append((half, s))
            s += clip_frames
    return out


def pick(rng, pool, n):
    """n clips from pool, split as evenly as possible between the two halves."""
    by_half = {1: [c for c in pool if c[0] == 1], 2: [c for c in pool if c[0] == 2]}
    chosen = []
    for half, want in ((1, (n + 1) // 2), (2, n // 2)):
        options = by_half[half]
        take = min(want, len(options))
        chosen += [options[i] for i in rng.choice(len(options), take, replace=False)]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip-seconds", type=int, default=120)
    parser.add_argument("--n-gappy", type=int, default=5)
    parser.add_argument("--n-other", type=int, default=5)
    parser.add_argument("--test-share", type=float, default=0.4)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(common.PLAN_PATH) and not args.force:
        sys.exit("a plan already exists ({}); annotations refer to its clip ids. "
                 "Use --force only if nobody has started annotating.".format(common.PLAN_PATH))

    raw = games.require(["atd_raw"], GAME, args.data_dir)[0]
    atd = load_atd(raw, N_FRAMES, os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz"), False)
    ball_ok = np.isfinite(atd["bx"]) & np.isfinite(atd["by"])

    clip_frames = args.clip_seconds * FPS
    pool = [(half, s, float(1 - ball_ok[s:s + clip_frames].mean())) for half, s in candidates(clip_frames)]
    threshold = float(np.quantile([m for _, _, m in pool], GAPPY_QUANTILE))
    gappy = [c for c in pool if c[2] >= threshold]
    other = [c for c in pool if c[2] < threshold]

    rng = np.random.RandomState(SEED)
    clips = []
    for kind, group, n in (("gappy", gappy, args.n_gappy), ("other", other, args.n_other)):
        chosen = pick(rng, group, n)
        order = rng.permutation(len(chosen))
        n_test = int(round(len(chosen) * args.test_share))
        for rank, index in enumerate(order):
            half, s, missing = chosen[index]
            clips.append({"start": int(s), "end": int(s + clip_frames - 1), "half": int(half),
                          "kind": kind, "set": "test" if rank < n_test else "tune",
                          "ball_missing": round(missing, 3)})
    clips.sort(key=lambda c: c["start"])
    for i, clip in enumerate(clips):
        clip["id"] = "c{:02d}".format(i + 1)

    calib = {"id": "calib", "start": PUBLIC_WINDOW[0], "end": PUBLIC_WINDOW[1], "half": 1,
             "kind": "public", "set": "calib",
             "ball_missing": round(float(1 - ball_ok[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1] + 1].mean()), 3)}
    plan = {"game": GAME, "fps": FPS, "seed": SEED, "clip_seconds": args.clip_seconds,
            "created": datetime.datetime.now().replace(microsecond=0).isoformat(),
            "clips": [calib] + clips}

    os.makedirs(common.LABEL_DIR, exist_ok=True)
    with open(common.PLAN_PATH, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=1)

    minutes = {s: sum(c["end"] - c["start"] + 1 for c in clips if c["set"] == s) / (60.0 * FPS)
               for s in ("tune", "test")}
    print("plan written: {}".format(common.PLAN_PATH))
    print("  calib: public window, {:.0f} min".format((PUBLIC_WINDOW[1] - PUBLIC_WINDOW[0] + 1) / (60.0 * FPS)))
    print("  {} clips of {} s: tune {:.0f} min, test {:.0f} min".format(
        len(clips), args.clip_seconds, minutes["tune"], minutes["test"]))
    for clip in clips:
        print("  {}  frames {:>6}-{:<6} half {}  {:<5}  {:<5}".format(
            clip["id"], clip["start"], clip["end"], clip["half"], clip["kind"], clip["set"]))


if __name__ == "__main__":
    main()
