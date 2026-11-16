# exp07 -- imparare e giudicare sul "possesso controllato" della sfida

Pre-registrazione, scritta il 2026-11-15 prima di lanciare `python external/exp07.py`.

**Perché.** CHALLENGE.md: il possesso cambia solo quando l'altra squadra ottiene il possesso *controllato*. Un tocco senza controllo, una deviazione che non intercetta, un duello aereo vinto senza controllo successivo non contano; con la palla contesa o libera il possesso resta all'ultima squadra. Una delle approssimazioni suggerite: "un giocatore tiene palla per qualche secondo". Le etichette DFL contano ogni tocco (mediana 3.7 s, 30% sotto 2 s). Il controllo pubblico di v6 (21 possessi predetti contro 17, frammentazione 1.24) è coerente con questo. Onestà: l'idea nasce dopo aver visto quel controllo; la soglia invece viene dal testo e non è stata provata sulla verità pubblica.

**La riscrittura** (`controlled()`, su ogni verità DFL, sia per addestrare sia per giudicare). Si scorre ogni tempo da sinistra a destra. Un tratto di una squadra più corto di **50 frame (2 s)**, subito dopo un tratto dell'altra squadra, diventa dell'altra squadra. Un tratto subito dopo una palla morta resta com'è (la ripresa ha un possessore chiaro). 2 s è il minimo di "qualche secondo", fissato dal testo. Effetto su due partite: da circa 450 a circa 290 possessi, mediana da 3–4 s a 7 s, 2% dei frame cambiati.

**Varianti**, tutte giudicate sulla verità riscritta (leave-one-TUNE-match-out, estrazione 0), prezzi del decoder ricercati per ognuna:

| | |
|---|---|
| R0 | v4w così com'è (riferimento) |
| R6 | v6 così com'è |
| RW | la ricetta di v4w addestrata sulla verità riscritta |
| RS | RW + lo stacker di exp06 (TW), addestrato sulla verità riscritta, annidato come in exp06 |

**Regola, fissata prima:** la migliore tra R6, RW e RS (CV spellF1, media dei due scenari matched) sostituisce R0 solo se lo supera di almeno 0.005 e il suo caso peggiore sui quattro scenari non è più di 0.005 sotto. HOLDOUT solo riportato. Poi un solo controllo sui 5 minuti pubblici del modello scelto (anche se resta R0, perché cambiano i prezzi del decoder).

## Risultato

Regola -> **adottato R6** (61 min): v6 (v4w + stacker TW) con i prezzi del decoder ricercati sulla verità riscritta: team_a 28, team_b 0, enter_dead 90, exit_dead 10, dead_bias 1.

| | CV matched | CV worst | holdout matched |
|---|---|---|---|
| R0 v4w | 0.6765 | 0.6511 | 0.623 |
| **R6 v6** | **0.6965** | 0.6853 | **0.649** |
| RW | 0.6727 | 0.6545 | 0.604 |
| RS | 0.6942 | **0.6874** | 0.637 |

R6 è meglio di R0 in tutti e quattro gli scenari, in CV e in holdout. Imparare sulla verità riscritta (RW, RS) non aiuta: conta giudicare e tarare il decoder sul possesso controllato.

Finale: `python external/exp07.py --final` -> `model/params_v7.json` (stessi modelli di v6, decoder nuovo); `python model/v3.py --params model/params_v7.json --out evaluation/results/v7_raw.csv` -> sha256 1b8394466cc4fefa9175353d70b2b84d67d5471190e1139bb6ea2e83d4403bc2, rigenerato identico da zero. Partita intera: 259 possessi, media 12.9 s.

**Controllo unico sui 5 minuti pubblici (dopo la decisione):**

| | spellF1 | macroF1 | possessi (veri 17) | frammentazione | transF1 |
|---|---|---|---|---|---|
| **v7** | **0.588** (bootstrap 0.41–0.75) | 0.833 | 15 | 0.95 | **0.462** |
| v4w | 0.575 | **0.860** | 15 | 0.95 | 0.359 |
| v6 | 0.505 | 0.829 | 21 | 1.24 | 0.400 |
| v2 | 0.525 | 0.875 | 15 | 1.00 | 0.450 |
