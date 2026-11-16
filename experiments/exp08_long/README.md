# exp08 -- lo stacker di v7 con una vista più lunga dei segnali grezzi, per le interruzioni

Pre-registrazione, scritta il 2026-11-15 prima di lanciare `python external/exp08.py`.

**Perché.** Nei 5 minuti pubblici gli errori più costosi di v7 sono sulla palla morta: interruzione presa tardi, possesso inventato mentre un giocatore sta vicino alla palla prima di una ripresa, ripartenza anticipata prima del calcio, palla morta inventata durante un lungo buco della palla. Un'interruzione si riconosce da **quanto a lungo** i giocatori sono lenti e la palla ferma, e il gioco riparte con un calcio, non quando un giocatore arriva sulla palla. Lo stacker di v7 vede +-2 s di risposte e i segnali grezzi solo all'istante t. (Diagnosi fatta confrontando le etichette di v7 con la verità pubblica; la correzione è generale e si giudica solo sulle partite DFL.)

**Cosa aggiunge** (`exp06.long_context()`, dai segnali già nelle tabelle): frame da cui i giocatori sono lenti / per cui lo resteranno (velocità media < 2 m/s); frame da / fino al prossimo calcio (palla > 5 m/s); palla ferma (< 1.5 m/s, vista); palla ferma con un giocatore entro 1.5 m; palla non vista; medie di velocità dei giocatori, palla vista e velocità della palla sui 5 e 10 s prima e dopo.

Tutto il resto è v7 (exp07 R6): primo stadio v4w + cambi v4, stessi fold annidati e risposte del primo stadio (checkpoint di exp06), pesi (1/L)^0.5 dello stacker, decoder ricercato di nuovo; giudicato sulla verità DFL riscritta al possesso controllato (50 frame).

**Varianti:** L (stacker imparato sulla verità DFL com'è, come in v7), LC (imparato sulla verità riscritta).

**Regola, fissata prima:** la migliore tra L e LC (CV spellF1, media dei due scenari matched, verità riscritta) sostituisce v7 solo se lo supera di almeno 0.005 e il suo caso peggiore sui quattro scenari non è più di 0.005 sotto. HOLDOUT solo riportato. Un solo controllo sui 5 minuti pubblici dopo la decisione, solo se adottata.

## Risultato

Regola -> **si tiene v7** (55 min).

| | CV matched | CV worst | holdout matched |
|---|---|---|---|
| v7 (exp07 R6) | 0.6965 | 0.6853 | 0.649 |
| L | 0.7009 | 0.6859 | 0.653 |
| LC | 0.6961 | 0.6779 | 0.652 |

L supera v7 di 0.0044 in CV matched, sotto la soglia di 0.005 fissata prima, quindi non è adottato (anche se non peggiora da nessuna parte e il suo decoder entra in palla morta molto più facilmente: enter_dead 28 contro 90). Nessun controllo sui 5 minuti pubblici.

## Controllo sui 5 minuti pubblici (fatto su richiesta, anche se L non era adottato)

`python external/exp08.py --final --force` e poi `model/v3.py --params model/params_v8.json` → `v8_raw.csv`.

| | spellF1 | macroF1 | possessi (veri 17) | transF1 τ25 |
|---|---|---|---|---|
| v7 | 0.5875 | 0.833 | 15 | 0.46 |
| L  | 0.5875 | 0.817 | 15 | 0.45 |

Stesso spellF1, macroF1 più basso: al tie-break v7 resta davanti. Si conferma v7.
