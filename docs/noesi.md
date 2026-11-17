# Team noesi: technical document

More than a Hack 2027, Phase 1: Ball Possession. Submission file: `noesi.csv` (model **v11**).

## 1. Summary

Every frame is labelled by LightGBM models in two stages and a Viterbi decoder:

1. a **frame model** that gives P(H), P(A), P(D) from 76 tracking signals;
2. a **change model** that gives the probability that possession passes directly to the other team
   at that frame;
3. two **stackers** that read the first two models' answers over ±2 s, plus signals about what
   happens over the next and previous 5–10 s (how long the players are slow and the ball still, runs
   of touches by the same team, how the two teams move, ball jumps across tracking gaps), and give
   new H / A / D probabilities, which are averaged;
4. a **three-state Viterbi decoder** that turns them into one sequence per half.

Everything is learned and tuned **only on external data**: the seven open DFL matches of the IDSSE
dataset (Bundesliga and 2. Bundesliga 2022/23, CC BY 4.0), whose per-frame ball status and ball
possession come from corrected tracking. The external tracking is clean and the challenge tracking
is not, so the external matches are first **degraded by a simulator matched to the challenge
tracking input** (never to its ground truth).

Three findings shaped the final model:

- the DFL ball is frozen during stoppages, which made dead ball far too easy to learn, so we
  simulate a realistic dead ball (exp04);
- the DFL labels count every touch (median possession 3.7 s), while the challenge counts only
  *controlled* possession, so the decoder is tuned on DFL labels rewritten to that
  definition (exp07);
- possession shows over several seconds, so giving the stacker the sequence of touches and the
  movement of both teams gave the largest single gain (exp10, exp11).

| | DFL, leave-one-match-out CV (controlled-possession labels) | DFL, 2 held-out matches | public 5 minutes, spellF1 / macroF1 |
|---|---|---|---|
| official baseline `smooth125` | – | – | 0.424 / 0.581 |
| previous submission (v2) | – | – | 0.525 / 0.875 |
| v4w (frame model + change model) | 0.677 | 0.623 | 0.575 / 0.860 |
| v7 (one stacker over ±2 s) | 0.697 | 0.649 | 0.588 / 0.833 |
| **v11 (submitted)** | **0.739** | **0.729** | **0.655** / 0.851 |

The public window has only 17 possessions: the bootstrap 5–95% interval of v11's spellF1 there is
0.48–0.81. Every choice was made on the DFL matches, and the public window was scored once after
each decision (section 2.6).

## 2. Approach

### 2.1 Why external data

We chose a stricter protocol than RULES.md requires: nothing is trained or tuned on the challenge
match, its ground truth (public window included), its video or hand labels. The public window is
only a final check. Every parameter must therefore be learned somewhere else, from frames whose
answer is known.

The IDSSE matches have it. Every ball frame carries `BallStatus` (in play or dead) and
`BallPossession` (home or away). We map them to `D` when the ball is dead and to `H` or `A`
otherwise. Automatic checks on the seven matches are in `external/dfl.py`:
- each match lasts 91–98 minutes;
- dead ball covers 41–52% of frames;
- on 90–94% of frames with a player within 1 m of the ball, that player's team is the team
  `BallPossession` names.

### 2.2 Making the external matches look like the challenge tracking

All of this is measured on the challenge **input** only, never on its ground truth or video
(`external/challenge_profile.py`, `external/degrade.py`, `external/dead_model.py`).

- **Ball gaps.** On the challenge the ball is missing on 41% of graded frames: 94% when players
  stand still, about 9% at full pace. The simulator removes the ball in each player-speed band up
  to the challenge's missing share, with the challenge's gap lengths (median 10.7 s). The two
  "matched" scenarios differ in whether gaps also follow dead ball (`matched_dead`) or only player
  speed (`matched_speed`). Two older scenarios with unrelated gaps (`independent`, `stoppage`) are
  kept for robustness: every model is trained on all four and judged on the two matched ones.
- **Players.** Slots drop out until 20.5 of 22 players are tracked on average.
- **Ball motion** (exp03). The challenge ball is smoothed: 9.3 detected touches per minute, against
  31–44 on clean DFL. The simulated ball is therefore given a slowly drifting error and then smoothed (`external/ball_model.py`; window 21, σ 0.5,
  τ 2.0), fitted to the challenge's touch rate, ball acceleration and turn-angle distributions.
- **Dead ball** (exp04). In the DFL data the ball stays perfectly still where it went out
  (0 m of movement in every stoppage), then jumps to the restart spot (more than 5 m in 60% of
  restarts). This comes from how the corrected data is produced; a camera would see something else. Our dead-ball model
  rolls the ball out with decaying speed and hides it, or carries it to the restart spot. It
  places the ball there for a while and picks it up late after the restart. Its six parameters
  are fitted by grid search to eight input-only statistics of the challenge tracking, such as the
  share of seen-ball frames outside the lines or still, and the ball's behaviour after a gap.

### 2.3 Signals (`model/features_v2.py`, `model/features_v3.py`)

One function computes the same 76 per-frame signals for the external matches and for the challenge
match:

- **who was near the ball last and next:** for each team, time since and until one of its
  players was within 1.5 m of a seen ball;
- **player speed** over 1, 2, 4 and 8 s, and the last 2 s against the next 2 s;
- **the ball itself:** speed, position relative to the lines, time since and until missing or
  outside, length of the current gap;
- **team shape:** spread and centroids, oriented towards the goal the home team attacks;
- **players around the ball:** nearest distances and counts within 3 m and 8 m;
- **touches** (exp03): frames where the seen ball turns by more than a threshold angle or changes
  speed sharply with a player within reach; per team, time since and until the last and next
  touch, touches in the last and next 2 s, and which team touched last and next;
- **control:** how long the same player has stayed within reach of the ball, and per team the
  longest such run in the last and next 2 s.

Every signal is symmetric or comes as a home/away pair. Each match is used twice, once with home
and away exchanged, and at prediction time the two readings are averaged, so the model has no
preference for either team.

### 2.4 Models

All are LightGBM models with 400 trees, 63 leaves, learning rate 0.05, trained on every
third frame.

- **Frame model** (`model/v4w_frame.txt`). Three classes: H, A, D. Each training frame is weighted
  by (1 / L)^0.5, where L is the length of the true run it belongs to (at least 5 frames). Without
  it, a 1-second interception is 25 frames among 140,000, while spellF1 counts it as much as a
  one-minute possession (exp04w).
- **Change model** (`model/v4_change.txt`). A binary model whose target is "within 6 frames of a
  direct H↔A change".
- **Stacker** (`model/v6_stack.txt`, exp06). For frame t it reads:
  - the log-probabilities of the frame model and the logit of the change model at t and at
    t ± 1, 3, 6, 12, 25 and 50 frames;
  - their means over centred windows of 25 and 75 frames and over the 50 frames before and after,
    plus the after-minus-before jump for H and A;
  - the maximum change probability over ±12 and ±25 frames;
  - the 76 signals at t.

  It is trained with the same (1 / L)^0.5 weights. The frame model sees one frame with summaries of
  the past; the stacker sees the pattern in time, for example the ball near a home player, then near
  an away player.

  **No leakage.** The stacker must learn from first-stage answers on matches the first stage has
  not seen.
  - *Cross-validation:* in the fold holding out match h, for every other tuning match m the first
    stage is retrained on the tuning matches minus {h, m} and predicts m, which takes 10 pair
    models. The stacker learns on those four matches and is evaluated on h.
  - *Final model:* the stacker learns on leave-one-match-out first-stage answers for all seven
    matches.

  **Stackers in the submission (exp08, exp10, exp11).** v11 averages two stackers that see more than
  the one above (`external/exp06.py: long_context, extra_context`), all computed from the 76
  signals, never from labels:
  - *long context (exp08):* for how many frames the players have been / will stay slow, the ball
    still, the ball still next to a player, the ball unseen; frames since / until the ball is
    kicked; means of player speed, ball speed and ball seen over the 5 and 10 s before and after;
  - *pass chains (exp10):* the number of touches in a row by the same team up to t and from t on.
    A completed pass is the strongest evidence of controlled possession, while a lone touch between
    two touches of the other team is usually a deflection;
  - *collective movement (exp10):* the displacement of each team's centroid over ±1 s and ±3 s, the
    change of each team's width and depth over ±2 s, the ball's advance over ±1 s and ±3 s. When
    a team wins the ball it usually spreads and moves up, and the other team drops back;
  - *ball jumps (exp10):* how far the ball moved across a tracking gap and the speed that implies
    (a ball that reappears where it could not have travelled means a stoppage in between);
  - *restarts (exp10):* for how long the ball has been / will stay still on a touchline or goal line.

  One stacker learns on the DFL labels as they are (X), the other on the labels rewritten to
  controlled possession (XC); their probabilities are averaged. Both use the same first stage
  (v4w frame model, v4 change model) and the same nested folds.

### 2.5 Decoder (`external/exp03.py: decode`)

The decoder is a Viterbi decoder over H, A and D, run on each half separately.
- A change of team at frame t costs `max(0, a − b·logit(p_change[t]))`.
- Entering dead ball and leaving it have fixed prices.
- A bias is added to log P(D).

v11's prices (`model/params_v11.json`) are a = 28, b = 0.75, enter 28, exit 28 and bias 1. The
change price is high, so possession changes only where the new team's evidence lasts or the change
model is confident. (v7's were a = 28, b = 0, enter 90, exit 10: with the longer context the
stackers recognise stoppages themselves and the decoder no longer has to make dead ball expensive.) The prices are chosen by coordinate search to
maximise spellF1 on out-of-fold predictions of the tuning matches.

### 2.6 Protocol

- **Matches.** Five DFL matches (J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ) are used for tuning with
  leave-one-match-out cross-validation. Two others (J03WN1, J03WR9) are held out and only
  reported.
- **Pre-registered rules.** Every experiment's decision rule is written in its README before the
  run. A variant replaces the current model only if its CV spellF1 (mean of the two matched
  scenarios) is at least **0.005** higher and its worst scenario is at most 0.005 lower.
- **Public window.** It is scored once, after each decision. When the public window contradicted
  the DFL result, we did not tune on it. We looked for a cause in the simulator or the labels that
  could be fixed from the input or the rule text, and wrote the fix as a new pre-registered
  experiment.
- **Notes on the protocol.**
  - The idea behind exp04 came from exp03's public check, and the idea behind exp07 from exp06's.
  - The 2 s threshold of exp07 comes from the CHALLENGE.md text ("a player keeps the ball for a
    few seconds"), was fixed before the run and was never varied.
  - In exp02b we deviated once from a pre-written rule, by 0.001; the deviation is declared in
    `experiments/exp02_v2/`.

### 2.7 Results: the experiment chain

Numbers are spellF1, as the mean of the two matched scenarios. Each experiment compares against its
own reference with the same simulator and labels, so rows from different experiments are not
directly comparable. Full records are in `experiments/*/README.md`.

| experiment | what changed | CV | held out | decision | public spellF1 / macroF1 |
|---|---|---|---|---|---|
| exp02 → v2 | learned frame classifier instead of hand rules (v1) | 0.561 | 0.533 | adopted | 0.525 / 0.875 |
| exp02c | ensemble of 5 v2 | – | – | adopted by rule, superseded | 0.494 / 0.833 |
| exp03 → v3 | realistic ball motion, touch and control signals, change model + varying price | 0.526 (v2: 0.495) | 0.497 | adopted | 0.511 / 0.761 (dead-ball recall 0.46) |
| exp04 → v4 | realistic dead ball | 0.544 (v2: 0.495) | 0.495 | adopted | 0.533 / 0.838 |
| exp04m | + 2 Metrica matches in training | 0.547 (v4: 0.544) | 0.506 | dropped | – |
| exp04c | ensemble of 5 v4 | 0.546 | 0.510 | dropped | – |
| exp04w → v4w | frame model weighted (1/L)^0.5 | **0.565** (v4: 0.544) | 0.526 | adopted | 0.575 / 0.860 |
| exp05 | change model weighted / 3 simulator draws / larger model / + Metrica | 0.567 / 0.5702 / 0.565 / 0.561 (v4w: 0.5653) | – | none passed (best +0.0049) | – |
| exp06 → v6 | stacker over ±2 s | **0.595** (v4w: 0.565) | 0.558 | adopted | 0.505 / 0.829 (21 possessions for 17) |
| exp07 → v7 | judged and decoder tuned on controlled-possession labels | **0.697** (v4w: 0.677) | 0.649 | adopted | **0.588** / 0.833 |
| exp08 | stacker also sees 5–10 s of raw context (how long players are slow, ball still, time to the next kick) | 0.701 / 0.696 (v7: 0.6965) | 0.653 / 0.652 | not adopted (+0.0044) | – |
| exp09 → v9 | mean of four stackers (v7's, exp08's two, one fully trained on rewritten labels) | **0.708** (v7: 0.697) | 0.661 | adopted | 0.594 / 0.827 |
| exp10 | two stackers with pass chains, team movement, ball jumps, still ball on a line; added to v9's mean | 0.722 (alone: 0.736 / 0.733) | 0.692 (alone: 0.729 / 0.725) | adopted, superseded by exp11 | – |
| exp11 → v11 | mean of the two exp10 stackers only (decided after seeing exp10, so not blind; confirmed by the held-out matches) | **0.739** | **0.729** | adopted | **0.655** / 0.851 |

**exp07 in detail** (all variants judged on the rewritten DFL labels):

| | CV | worst scenario | held out |
|---|---|---|---|
| R0: v4w, decoder searched again | 0.677 | 0.651 | 0.623 |
| **R6: v6 models, decoder searched again (= v7)** | **0.697** | 0.685 | **0.649** |
| RW: v4w recipe retrained on rewritten labels | 0.673 | 0.655 | 0.604 |
| RS: RW + stacker retrained on rewritten labels | 0.694 | 0.687 | 0.637 |

Retraining on the rewritten labels did not help. What mattered was judging and pricing the decoder
by the challenge's definition.

**Public window** (17 possessions, mean 14.2 s), scored with `evaluation/run_evaluation.py`:

| | spellF1 | macroF1 | possessions predicted | transition F1 (±1 s) |
|---|---|---|---|---|
| `smooth125` | 0.424 | 0.581 | 16 | 0.17 |
| v1 | 0.471 | 0.817 | 17 | 0.24 |
| v2 | 0.525 | 0.875 | 15 | 0.45 |
| v4w | 0.575 | 0.860 | 15 | 0.36 |
| v6 | 0.505 | 0.829 | 21 | 0.40 |
| v7 | 0.588 | 0.833 | 15 | 0.46 |
| v9 | 0.594 | 0.827 | 14 | 0.42 |
| **v11** | **0.655** | 0.851 | 16 | **0.60** |

## 3. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| automatic tracking, tactical (ATD) | **yes** | the only model input: player and ball positions, player speeds, team of each slot |
| DFL / IDSSE open data | **yes** | all training and tuning (seven matches, CC BY 4.0) |
| Metrica Sports sample data (games 1 and 2) | tried, dropped | H/A/D rebuilt from their event logs (`external/metrica.py`). Adding them to training never passed the decision rule (exp04m: +0.003; exp05 WM: −0.004). It did show that v4, trained on DFL only, carries over to another provider (spellF1 0.53–0.57 on these matches). |
| Smart Tagging | no | the external matches have no equivalent, so how much to trust it could not be tuned without tuning on the challenge match |
| homography | no | everything is computed in pitch coordinates |
| tactical video | not an input | the public 5-minute window was watched with its ground truth to understand the possession rules, which the rules allow; no model output was ever checked against the video |
| broadcast video and tracking | no | not aligned with the tactical source, and the tactical tracking was enough |
| public ground truth | evaluation only | scored once per decision (section 2.6) |

**Training data.** The only labelled data used to train and tune the submitted model:

| dataset | source | who labelled it | how much |
|---|---|---|---|
| DFL / IDSSE open data (Bassek et al., Scientific Data 2025, CC BY 4.0) | Deutsche Fußball Liga, public download | the DFL's data provider: per-frame `BallStatus` and `BallPossession` shipped with the official tracking. We added no labels | 7 full matches (5 for tuning with cross-validation, 2 held out), about 1 million frames at 25 Hz |

None of it comes from the provided game. The Metrica sample data (2 matches, labels rebuilt by us
from Metrica's own event logs) was used in experiments only and is not in the submitted model. The
public 5 minutes were used only to check the pipeline, once after each decision.

Before the organisers' clarification we built a tool to hand-label frames of the challenge match.
No hand label was ever used by any model, and the tool is not part of this submission.

## 4. Possession rules implemented

The rules are not hand-coded. The models learn them from the DFL labels, and the decoder is priced
on DFL labels rewritten to the challenge's *controlled possession*:

- **Controlled possession (exp07).** Within each half, going left to right, a run of one team
  shorter than **2 s** (50 frames) that directly follows a run of the other team is given to that
  other team: it never gained control. A run that directly follows a dead ball is kept whatever
  its length, because a restart has a clear possessor. This changes 2% of DFL frames. The DFL
  possession count drops from about 450 to about 290 per match, and the mean possession grows from
  3–4 s to 10–11 s, close to the public window's 14 s. The submitted model predicts 246
  possessions with a mean of 13.7 s over the full challenge match.
- **Loose balls, challenges, deflections, passes in flight.** These keep the previous team unless
  the other team's evidence lasts long enough to pay the change price (28 in summed
  log-probability). The "near last and next", touch and control signals carry the team through
  passes and short ball gaps. A ball far from everyone gives no team evidence.
- **Dead ball and restarts.** A stoppage costs 56 in total (28 to enter, 28 to leave), so the model
  declares one only when the evidence lasts: players slowing down, the ball outside the lines, or
  the ball missing while players are slow. A momentary slowdown does not split a possession.
- **Explicit rules.**
  - Frames outside the two halves are `X`.
  - The second half is `D` until its kickoff (FAQ.md).
  - Both sides of the match are read and averaged.

## 5. Limitations

- **Everything rests on the simulator.** All choices were made on degraded DFL matches. Twice a model that won on
  DFL lost on the public window (frozen dead ball in exp03, label convention in exp06). There are
  certainly other mismatches we have not found.
- **The controlled-possession rewrite is approximate.** One duration threshold cannot tell a controlled
  1.5 s interception, which should count, from a 2.5 s scramble, which should not. With more time
  we would use DFL event data (passes, duels) to rewrite the labels from events instead of
  durations.
- **Short possessions and transition timing.** On the DFL held-out matches only about 40% of
  possession starts were taken within 1 s (diagnosis before exp04w). Touches are hard to see on a smoothed ball that is
  missing 41% of the time.
- **macroF1 (the tie-break)** is lower than v2's on the public window: 0.851 against 0.875. Part of
  this comes from stoppage edges a few seconds off.
- **More data variety helped more than model size** (exp05). Training on several simulator draws
  came within 0.0001 of the threshold. It is the first thing we would add.
- **Smart Tagging, homography and video are unused** (section 3).

## 6. Reproduction

Requirements: Python 3.12 and the match data bundle unpacked as described in DATA.md, in `./data`
or anywhere `MTAH_DATA_DIR` points.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-tools.txt     # Windows; .venv/bin/python elsewhere
.venv/Scripts/python.exe model/v3.py --params model/params_v11.json --out evaluation/results/noesi.csv
```

The trained models are in the repository (`model/params_v11.json`, `model/v4w_frame.txt`,
`model/v4_change.txt`, `model/v11_x_stack.txt`, `model/v11_xc_stack.txt`), so this regenerates the submission without
retraining. It takes a few minutes on a laptop. We regenerated it from the saved models: the file is byte-identical, SHA-256
`8a7355e286aa40dfb2dbcc46d5fa85df98501e25d4f10d194503a376a22dfc49`.

To retrain from scratch (several hours on a laptop; the DFL download is about 2.7 GB). Each step
reads the previous step's results and checkpoints:

```bash
python external/dfl.py                 # download and parse the 7 DFL matches
python external/challenge_profile.py   # input statistics of the challenge tracking
python external/ball_model.py          # fit the ball motion to the challenge input
python external/dead_model.py          # fit the dead-ball model to the challenge input
python external/exp04.py               # degraded tables, exp04 (touch signals, change model)
python external/exp04.py --final       # -> model/v4_change.txt, model/params_v4.json
python external/exp04w.py              # frame weighting
python external/exp04w.py --final      # -> model/v4w_frame.txt, model/params_v4w.json
python external/exp06.py               # stacker, nested cross-validation
python external/exp06.py --final       # -> model/v6_stack.txt, model/params_v6.json
python external/exp07.py               # controlled possession
python external/exp07.py --final       # -> model/params_v7.json
python external/exp08.py               # longer stacker context
python external/exp08.py --final --force   # -> model/v8_stack.txt (needed by exp09)
python external/exp09.py               # mean of four stackers
python external/exp10.py               # stackers with the new signals
python external/exp11.py               # mean of the two new stackers
python external/exp11.py --final       # -> model/v11_x_stack.txt, model/v11_xc_stack.txt, model/params_v11.json
python external/exp12.py               # team-continuity signals (not adopted)
python model/v3.py --params model/params_v11.json --out evaluation/results/noesi.csv
```

Data credit: Bassek, M., Weber, H., Rein, R. & Memmert, D. *An integrated dataset of
spatiotemporal and event data in elite soccer.* Scientific Data (2025). Data: Deutsche Fußball
Liga (DFL), CC BY 4.0. Metrica Sports sample data:
github.com/metrica-sports/sample-data (used in experiments only, not in the submitted model).
