# exp09 — media di più stacker (pre-registrazione, scritta prima di eseguire)

**Idea.** exp08 L ha dato +0.0044 da solo, sotto soglia. Stacker addestrati in modo diverso sbagliano
in punti diversi: la media delle loro probabilità riduce il rumore (meno possessi spezzati o persi)
senza nuovi dati né nuovi segnali. Costa pochi minuti: le risposte fold per fold sono già in cache.

**Varianti** (media aritmetica delle probabilità per frame, stesso change model, decoder ricercato da capo):
- `E2` = v7 (exp06 TW) + L (exp08)
- `E4` = v7 + L + LC (exp08) + RS (exp07, stacker su verità riscritta)

**Giudizio.** Come exp07/exp08: verità DFL riscritta a possesso controllato (50 frame), CV
leave-one-match-out sulle 5 partite TUNE, 4 scenari; HOLDOUT solo riportato.

**Regola, fissata ora.** La migliore fra E2 ed E4 per CV matched sostituisce v7 solo se
`CV matched >= 0.6965 + 0.005` e `CV worst >= 0.6853 − 0.005`. Altrimenti resta v7.
I 5 minuti pubblici si guardano una volta sola, e solo se la regola è superata.

## Risultato

| | CV matched | CV worst | holdout |
|---|---|---|---|
| v7 | 0.6965 | 0.6853 | 0.649 |
| E2 | 0.7005 | 0.6854 | 0.650 |
| **E4** | **0.7076** | **0.6915** | **0.661** |

Regola superata (+0.0111): **E4 adottato → v9** (`python external/exp09.py --final`, `model/params_v9.json`).
Decoder: team_a 20, team_b 0.5, enter_dead 60, exit_dead 6, dead_bias 0.

Controllo unico sui 5 minuti pubblici: v9 spellF1 0.5935 (bootstrap 0.41–0.76), macroF1 0.827,
14 possessi contro 17 veri; v7 0.5875 / 0.833 / 15. Nessuna sorpresa: differenza dentro il rumore.
