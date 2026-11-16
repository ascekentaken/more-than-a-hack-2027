"""Experiment exp04: exp03 again, with stoppages that look like the challenge's instead of a frozen ball.

    python external/exp04.py            # writes experiments/exp04_dead/results.md and results.json
    python external/exp04.py --final    # retrains the chosen variant on all 7 matches -> model/params_v4.json

Why. The one check of exp03 on the five public minutes, made after its decision, showed v3 calling 22 s
of dead ball a Barcelona possession; on the DFL matches every variant recognised dead ball 98% of the
time. The DFL ball is frozen during stoppages -- exactly still, often outside the pitch or on a line --
which makes dead ball far easier than on the challenge tracking. external/dead_model.py replaces the
frozen ball with one fitted on the challenge tracking INPUT only (seen outside the pitch, by a line,
still, seen while the players are slow, where it comes back after a gap).

Everything else is exp03: same matches, seeds, scenarios, new ball, variants (S the submitted v2;
A / B / C / D the v2 or touch signals with a fixed or varying price), same decoder search, same rule
fixed before the run: highest leave-one-match-out spellF1 on the five TUNE matches (mean of the two
matched scenarios); among the variants within TIE of it, the best worst case over the four degraded
scenarios. The two HOLDOUT matches are only reported.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import exp03  # noqa: E402

exp03.DEAD = "real"
exp03.TABLE_TAG = "v4"
exp03.CHECKPOINT = "exp04"
exp03.TITLE = "exp04"
exp03.OUT_DIR = os.path.join(ROOT, "experiments", "exp04_dead")
exp03.FRAME_MODEL = os.path.join(ROOT, "model", "v4_frame.txt")
exp03.CHANGE_MODEL = os.path.join(ROOT, "model", "v4_change.txt")
exp03.PARAMS_PATH = os.path.join(ROOT, "model", "params_v4.json")

if __name__ == "__main__":
    exp03.final() if "--final" in sys.argv else exp03.main()
