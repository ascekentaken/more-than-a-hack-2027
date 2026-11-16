# exp06 -- un secondo modello che legge le risposte di v4w su +-2 secondi (stacking con contesto)

Pre-registrazione, scritta il 2026-11-15 prima di lanciare `python external/exp06.py`.

**Perché.** Il modello per frame di v4w guarda un fotogramma alla volta. Un intercetto di 1 secondo è un disegno nel tempo (palla vicino a un giocatore di casa, poi a uno ospite): nel 60% dei possessi brevi persi il modello per frame non dà mai più di 0.5 alla squadra giusta (analysis/dfl_errors.py).

**Cosa aggiunge.** Un secondo LightGBM (stacker). Per il frame t riceve:
- le probabilità di v4w (H, A, D) e la probabilità di cambio a t e a t+-1, 3, 6, 12, 25, 50 frame;
- le loro medie su finestre centrate di 25 e 75 frame, sui 50 frame prima e i 50 dopo, la differenza dopo-meno-prima per H e A, il massimo della probabilità di cambio su +-12 e +-25;
- i 76 segnali di v4w a t.

Il bersaglio è H/A/D a t. La sua uscita sostituisce le probabilità per frame di v4w nel decoder; le probabilità di cambio restano quelle di v4w. Letto dai due lati e mediato.

**Senza fughe di informazione.** Nel fold che tiene fuori la partita h, per ogni altra partita TUNE m il primo stadio (v4w per frame + v4 cambi) è addestrato su TUNE meno {h, m} e predice m. Lo stacker impara su quelle quattro partite ed è valutato su h con le risposte di v4w per h (addestrato su TUNE meno h). Per HOLDOUT lo stacker impara sulle risposte leave-one-out delle cinque partite TUNE.

**Varianti:** T (ogni frame pesa uguale), TW (pesi (1/L)^0.5 come v4w). Prezzi del decoder ricercati per ognuna.

**Regola di decisione, fissata prima:** la variante con lo spellF1 leave-one-match-out più alto su TUNE (media dei due scenari matched) sostituisce v4w solo se lo supera di almeno 0.005 e il suo caso peggiore sui quattro scenari non è più di 0.005 sotto. HOLDOUT solo riportato. I 5 minuti pubblici: un solo controllo dopo la decisione, solo se adottata.

**Se passano sia exp05 sia exp06:** si sceglie quello con CV matched più alto; la combinazione (stacker sopra la variante exp05) solo se c'è tempo, con la stessa regola rispetto al migliore dei due.

## Risultato

Regola -> **adottato TW** (63 min). CV matched 0.5949 contro v4w 0.5653 (+0.030), caso peggiore 0.5785 contro 0.5365, holdout 0.558 contro 0.526; meglio in tutti e quattro gli scenari, in CV e in holdout. T (senza pesi) 0.5773.

Finale: `python external/exp06.py --final` -> `model/v6_stack.txt`, `model/params_v6.json`; `python model/v3.py --params model/params_v6.json --out evaluation/results/v6_raw.csv` (sha256 e7628d1d6c69809793395357893a37739a03959320d72c27d11cc51889eea3a6, rigenerato identico da zero).

**Controllo unico sui 5 minuti pubblici (dopo la decisione): va nella direzione opposta.**

| | spellF1 | macroF1 | possessi predetti (veri 17) | frammentazione | transF1 |
|---|---|---|---|---|---|
| v4w | 0.575 | 0.860 | 15 | 0.95 | 0.359 |
| v6 | 0.505 | 0.829 | 21 | 1.24 | 0.400 |

Diagnosi (dal testo di CHALLENGE.md, non dalla verità): la sfida conta solo il possesso *controllato* (tocchi, deviazioni e duelli senza controllo non cambiano possesso), mentre le etichette DFL contano ogni tocco (mediana 3.7 s, 30% sotto 2 s). Lo stacker impara a prendere proprio quei possessi brevi e sulla partita della sfida spezza i possessi. Si prosegue con exp07.
