# Team noesi — technical document

More than a Hack 2027, Phase 1: Ball Possession. Submission file: `noesi.csv`.

## 1. Summary

We label every frame with a **LightGBM frame classifier followed by a three-state Viterbi
decoder**. The classifier is trained **only on external data**: the seven open DFL matches of the
IDSSE dataset (Bundesliga and 2. Bundesliga 2022/23, CC BY 4.0). Those matches come with per-frame
ball status and ball possession taken from corrected tracking, which is the same kind of label as
the challenge ground truth.

The external tracking is clean, while the challenge tracking is automatic. Before training, we
**degrade the external matches with a simulator matched to the challenge tracking**. The
simulator reproduces statistics we measured on the challenge *input* only (when and for how long
the ball goes missing, how noisy it is), never on its ground truth.

Every choice was made on external data, with leave-one-match-out cross-validation and two held-out
matches:

| | spellF1 on 2 held-out DFL matches, degraded like the challenge |
|---|---|
| official baseline (`smooth50`) | 0.257 |
| our hand-written model (v1) | 0.375 |
| **submitted model (v2)** | **0.533** |

On the public 5-minute window, scored once and after the model was frozen, the submitted model
reaches spellF1 0.525 and macroF1 0.875, against 0.326 / 0.621 for `smooth50`.

## 2. Approach

### 2.1 Why external data

The clarified rules forbid training or tuning on the challenge match, its ground truth, its video
or hand labels. Any parameter therefore has to be learned somewhere else, and that requires frames
whose answer is known.

The IDSSE matches have it. Every ball frame carries `BallStatus` (in play or dead) and
`BallPossession` (home or away), set by the data provider on its corrected tracking. We map these
to `D` when the ball is dead, and otherwise to `H` or `A`.

Automatic checks on the seven matches (`external/dfl.py`):
- each match lasts 91–98 minutes;
- dead ball covers 41–52% of the frames;
- on 90–94% of frames where a player is within 1 m of the ball, that player's team is the team
  `BallPossession` names;
- the file's speed units match the speeds recomputed from positions.

### 2.2 Making the external matches look like the challenge tracking

Clean tracking would teach a model that the ball is always visible. On the graded frames of the
challenge it is missing 41% of the time. We first measured how the challenge tracking behaves,
**on its input only** (`external/challenge_profile.py`; no ground truth, no video):

- **The ball goes missing mostly while play is stopped.** The missing share falls from 94% when the
  players stand still (mean speed below 0.5 m/s) to about 9% at full pace. This curve has the same
  shape as the dead-ball rate against player speed on the DFL matches. During ball gaps the players
  move at 1.17 m/s on average, against 2.16 m/s when the ball is seen.
- **Ball gaps** last a median of 10.7 s, about 127 of them per 90 minutes, with few short reappearances
  in between.
- **Ball speed** has a median of 5.1 m/s, so ball noise is small.

Our first simulator, built from the published figures alone, got two things wrong. It placed gaps
independently of play, and its ball noise was too strong: the median ball speed came out at
13.6 m/s. The final simulator (`external/degrade.py`) has two "matched" scenarios:

- **`matched_dead`:** a dead-ball period loses the ball with the probability the challenge loses
  it when players stand still (87%). Further gaps are added in each player-speed band up to the
  challenge's missing share. This is the closest to the challenge: players move at 1.19–1.30 m/s
  during its gaps, and 8–9% of visible stretches are shorter than 3 s, against 11% on the
  challenge.
- **`matched_speed`:** the same per-band missing share and challenge gap lengths, but with gaps
  tied only to player speed.

In both scenarios players drop out until 20.5 of 22 are tracked on average, and the ball gets 0.04 m
of positional noise, fitted to the challenge ball-speed percentiles. Two older scenarios with
unrelated gaps (`independent`, `stoppage`) are kept for robustness.

### 2.3 Signals (`model/features_v2.py`)

The same function computes 57 per-frame signals for the external matches and for the challenge
match. They include:

- **who touched the ball last and next:** for each team, the time since and until one of its
  players was within 1.5 m of a seen ball, and which team that was;
- **player speed over several windows:** 1, 2, 4 and 8 s, and the last 2 s against the next 2 s,
  so a stoppage or a restart shows up at the frame where it happens;
- **the ball itself:** its speed and position relative to the lines, time since and until it was
  missing or outside the pitch, and the length of the current gap;
- **team shape:** spread and centroid of each team, oriented towards the goal the home team
  attacks;
- **players around the ball:** nearest-player distances and counts within 3 m and 8 m, for each
  team.

Every signal is either symmetric or comes as a home/away pair.

### 2.4 Model (`model/v2.py`)

- **Classifier:** LightGBM with three classes (`H`, `A`, `D`): 400 trees, 63 leaves, learning
  rate 0.05, trained on every third frame. Each match is used twice, once with home and away
  exchanged, so the model has no preference for either team. At prediction time both readings are
  averaged.
- **Decoder:** a Viterbi decoder over `H`, `A`, `D`, run on each half separately. It pays one price
  for a change of team (6), one for entering dead ball (90) and one for leaving it (10), and adds
  1 to the log-probability of `D`. These four numbers are chosen to maximise spellF1 on the
  out-of-fold predictions.

### 2.5 Protocol and results

Five DFL matches were used for tuning, with leave-one-match-out cross-validation. Two other matches
(J03WN1, J03WR9) were held out and never used for any choice. The decision rule was written down
before the results were seen.

**exp02, v2 against v1 on the held-out matches (spellF1):**

| method | `matched_dead` | `matched_speed` | `matched_dead` at IoU 0.7 | `matched_dead` transitions within ±1 s (F1) |
|---|---|---|---|---|
| baseline `smooth50` | 0.299 | 0.215 | 0.148 | 0.232 |
| v1, as first submitted | 0.418 | 0.332 | 0.265 | 0.398 |
| v1, re-tuned on the matched scenarios | 0.456 | 0.349 | 0.297 | 0.467 |
| **v2** | **0.577** | **0.499** | **0.415** | **0.634** |

The better simulator alone is worth about +0.03; the learned model about +0.13.

**exp02b, four variants chosen on cross-validation only (spellF1):**

| variant | cross-validation, matched | worst of 4 scenarios | held out, matched |
|---|---|---|---|
| base | 0.568 | 0.356 | 0.538 |
| **mixed** (also trained on the old scenarios) | 0.561 | **0.490** | 0.533 |
| relative player speeds | 0.572 | 0.356 | 0.541 |
| mixed + relative | 0.558 | 0.486 | 0.534 |

We submit **mixed**. It costs about 0.007 when the challenge tracking behaves like our matched
scenarios, and gains 0.13 if its gaps turn out unrelated to play. This is a declared deviation
from the pre-written rule, which would have picked `base`: `mixed` missed the tie margin by 0.001.

The full record is in `experiments/exp02_v2/README.md`, `results.md` and `variants.md`.

**Public window (5 minutes, 17 possessions), checked once after the decision:**

| | `smooth125` | `smooth50` | v1 | **v2** |
|---|---|---|---|---|
| spellF1 | 0.424 | 0.326 | 0.471 | **0.525** |
| macroF1 | 0.581 | 0.621 | 0.817 | **0.875** |
| transitions within ±1 s | 3 / 20 | 2 / 20 | 5 / 20 | **9 / 20** |
| mean transition latency | −7.7 frames | +4.0 frames | +8.4 frames | **−0.3 frames** |

With 17 possessions the bootstrap 5–95% interval of v2's spellF1 is 0.36–0.68, so we do not read
much into this window. Nothing was changed because of it.

## 3. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| automatic tracking, tactical (ATD) | **yes** | the only input to the model: player and ball positions, player speeds, team of each slot |
| Smart Tagging | no | the external matches have no equivalent, so how much to trust it could not be tuned without tuning on the challenge match |
| homography | no | everything is computed in pitch coordinates |
| tactical video | no input | the public 5-minute window was watched with its ground truth to understand the possession rules, which the rules allow; no model output was ever checked against the video |
| broadcast video and tracking | no | not aligned with the tactical source, and the tactical tracking was enough |
| public ground truth | evaluation only | scored once, after the model was frozen |
| DFL / IDSSE open data | **yes** | all training and tuning |

Before the organisers' clarification we had built a tool to hand-label frames of the challenge
match. No hand label was ever used by any model in this repository, and the tool is not part of
this submission.

## 4. Possession rules implemented

We do not hand-code the rules. The model learns them from the DFL labels, which follow the
challenge definition: dead where the corrected tracking marks the ball dead, otherwise the team it
names as owning the ball. In practice:

- **Change of possession.** The team changes when the evidence for the other team outweighs a
  fixed price (6 in summed log-probability). The strongest evidence is the other team getting
  close to the ball and staying close, with no player of the previous team near it.
- **Loose balls, challenges and passes in flight.** These keep the previous team unless the other
  team's evidence lasts long enough to pay the price. The "who touched last and next" signals carry
  the team through passes and short ball gaps. A pass or a ball far from everyone gives no team
  evidence of its own.
- **Restarts and dead ball.** A stoppage costs 100 in total (90 to enter, 10 to leave), so the
  model only declares one when the evidence lasts: players slowing down, the ball outside the
  lines, or the ball missing while players are slow. A momentary slowdown does not split a
  possession in two.
- **Explicit rules.**
  - Frames outside the two halves are `X`.
  - The second half is `D` until its kickoff, as stated in FAQ.md.
  - The match is read from both sides and the two readings are averaged, so home and away are
    treated alike.

## 5. Limitations

- **Short possessions.** 30% of the DFL possessions last under 2 s, and every possession counts
  equally in spellF1. The model predicts only about 65–77% as many possessions as the truth has,
  because it merges short ones into their neighbours. The public window shows the same pattern:
  two `H` possessions of 0.8 s and 1.3 s are absorbed into the surrounding `A`. With more time we
  would make the price of a team change vary over time, driven by a model of touches, and add
  finer touch signals such as changes of ball direction near a player.
- **Edges of stoppages** can be off by a few seconds, for example a restart predicted early.
- **Simulator assumptions.** The matched scenarios reproduce what the challenge input shows, but
  whether its gaps are tied to dead ball cannot be confirmed without the ground truth. We
  submitted the variant that stays robust if they are not.
- **Label definition.** The DFL matches have about 437 possessions per match. The public window's
  pace is closer to 300 per 90 minutes. This may be Barcelona's style, or a slightly different
  notion of "controlled" possession; we could not resolve it without the challenge ground truth.
- **Smart Tagging, homography and video are unused** (section 3).

## 6. Reproduction

Requirements: Python 3.12 and the match data bundle unpacked as described in DATA.md.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-tools.txt     # Windows; bin/python elsewhere
# data in ./data, or point MTAH_DATA_DIR at it
.venv/Scripts/python.exe model/v2.py --out evaluation/results/noesi.csv
```

The trained model is in the repository (`model/v2_lgb.txt`, `model/params_v2.json`), so this
regenerates the submission without retraining. We checked this from a fresh clone and a fresh
environment: the file is byte-identical, SHA-256
`78eb7156a43e431a1acf660dbd28b51346e8357982326a8ee21e5540fe89721a`.

To retrain everything from scratch, about an hour on a laptop:

```bash
python external/dfl.py                   # download and parse the 7 DFL matches (~2.7 GB)
python external/challenge_profile.py     # input statistics of the challenge tracking
python external/exp02.py                 # exp02: protocol, held-out results
python external/exp02b.py                # exp02b: variants
python external/exp02b.py --final mixed  # final model on all 7 matches
python model/v2.py --out evaluation/results/noesi.csv
```

Data credit: Bassek, M., Weber, H., Rein, R. & Memmert, D. *An integrated dataset of
spatiotemporal and event data in elite soccer.* Scientific Data (2025). Data: Deutsche Fußball
Liga (DFL), CC BY 4.0.
