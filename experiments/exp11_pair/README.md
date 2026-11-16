# exp11 — media dei soli due stacker con i segnali nuovi (X + XC)

**Dichiarazione onesta.** Questo esperimento NON è alla cieca: è stato deciso dopo aver visto in exp10
che X (CV 0.7358, holdout 0.729) e XC (0.7334, 0.725) da soli battono la media a sei E6
(0.7222, 0.692), che la regola di exp10 aveva adottato. I vecchi stacker nella media frenano i nuovi.

**Candidato.** `EX` = media di X e XC, decoder ricercato da capo.

**Regola, scritta prima di calcolare EX.** EX sostituisce E6 solo se `CV matched >= 0.7222 + 0.005`
e `CV worst >= 0.7021 − 0.005`. L'holdout (mai usato per scegliere) deve confermare: se EX fosse
sotto E6 sull'holdout, resta E6. 5 minuti pubblici: una volta sola sul vincitore.

## Risultato

| | CV matched | CV worst | holdout |
|---|---|---|---|
| v9 (exp09 E4) | 0.7076 | 0.6915 | 0.661 |
| E6 (exp10) | 0.7222 | 0.7021 | 0.692 |
| **EX** | **0.7386** | **0.7222** | **0.729** |

Regola superata → **v11** (`python external/exp11.py --final`, `model/params_v11.json`).
Decoder: team_a 28, team_b 0.75, enter_dead 28, exit_dead 28, dead_bias 1.

Controllo unico sui 5 minuti pubblici: v11 spellF1 0.6545 (bootstrap 0.48–0.81), macroF1 0.851,
16 possessi contro 17 veri, transF1 0.60. v7: 0.5875 / 0.833 / 15 / 0.46.
