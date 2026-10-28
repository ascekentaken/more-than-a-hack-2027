"""Draw which of the 46 tracking slots are occupied, moment by moment.

    python analysis/plot_slots.py                          # first 2 minutes of play
    python analysis/plot_slots.py --start 20000 --seconds 90

One row per slot (the columns of the tracking file), one pixel column per frame. A coloured cell
means "this slot holds a tracked object at that frame": blue for a home slot, red for an away slot.
White means empty. Under the grid, a strip shows the frames where the ball is tracked.

What to look for:
  * about half the rows are filled at any moment -- 46 is the size of the list, not the number of
    players;
  * a long unbroken bar is one slot following one object for that long;
  * a bar that stops and another that starts in a different row is the tracker losing someone and
    putting them back in a free slot. That is why a slot is not a person.

The picture is derived from the gated match data: it goes to analysis_out/ (gitignored). Do not
commit it, do not redistribute it, and do not send it to a hosted LLM.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
from analyze_data import load_atd  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames("tactical", GAME)
HOME_RGB, AWAY_RGB = (31 / 255., 94 / 255., 168 / 255.), (178 / 255., 58 / 255., 47 / 255.)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=11185, help="first video frame")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "analysis_out"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    atd = load_atd(raw, N_FRAMES, os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz"),
                   False)
    team = bl.team_of_each_track(meta)

    first = args.start
    last = min(first + args.seconds * FPS, N_FRAMES)
    occupied = (np.isfinite(atd["px"][first:last]) & np.isfinite(atd["py"][first:last])).T
    ball = (np.isfinite(atd["bx"][first:last]) & np.isfinite(atd["by"][first:last]))

    order = np.argsort(team, kind="stable")  # home slots first, then away
    n_slots, n_frames = occupied.shape
    image = np.ones((n_slots, n_frames, 3))
    for row, slot in enumerate(order):
        colour = HOME_RGB if team[slot] == bl.HOME else AWAY_RGB
        image[row, occupied[slot]] = colour

    fig, (ax, strip) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [n_slots, 2], "hspace": 0.04})
    seconds = (last - first) / float(FPS)
    ax.imshow(image, aspect="auto", interpolation="nearest", extent=[0, seconds, n_slots, 0])
    ax.set_yticks(np.arange(n_slots) + 0.5)
    ax.set_yticklabels([str(slot + 1) for slot in order], fontsize=6)
    ax.set_ylabel("slot (Track number)  --  home on top, away below")
    ax.axhline(int((team == bl.HOME).sum()), color="black", lw=1)
    ax.set_title("Which of the {} slots are occupied, frames {}-{}.  Blue = home slot, red = away "
                 "slot, white = empty".format(n_slots, first, last - 1), fontsize=11, pad=34)

    # A second scale on top in the video's own clock, so a moment in the picture can be found in
    # a player without doing any sums. Frame = first + seconds * FPS.
    step = 10 if seconds <= 180 else 30
    ticks = np.arange(0, seconds + 0.1, step)
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels(["{}:{:04.1f}".format(int((first / float(FPS) + t) // 60),
                                              (first / float(FPS) + t) % 60) for t in ticks],
                        fontsize=8)
    top.set_xlabel("video clock (m:ss.s) -- type this time in the video player", fontsize=9)

    strip.imshow(np.where(ball[None, :, None], np.array([[[0.1, 0.1, 0.1]]]),
                          np.array([[[1.0, 1.0, 1.0]]])) * np.ones((1, n_frames, 3)),
                 aspect="auto", interpolation="nearest", extent=[0, seconds, 0, 1])
    strip.set_yticks([0.5])
    strip.set_yticklabels(["ball"])
    strip.set_xlabel("seconds from frame {}".format(first))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "slots_{}_{}s.png".format(first, args.seconds))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("written: {}".format(path))
    print("slots occupied per frame in this stretch: min {} median {:.0f} max {}".format(
        int(occupied.sum(0).min()), float(np.median(occupied.sum(0))), int(occupied.sum(0).max())))


if __name__ == "__main__":
    main()
