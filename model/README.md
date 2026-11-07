# Modelli

**Il modello da consegnare è il v2** (`python model/v2.py` → `evaluation/results/v2_raw.csv`). È
un classificatore LightGBM allenato sulle 7 partite DFL, degradate da un simulatore tarato sulle
statistiche dell'input del nostro tracking, seguito da un decodificatore a tre prezzi. Come è
stato costruito, scelto e verificato è descritto in
[`experiments/exp02_v2/README.md`](../experiments/exp02_v2/README.md).

Sulle partite tedesche di controllo lo spellF1 passa da 0,375 (v1) a 0,533. Sotto resta la
descrizione del v1, che è il punto di partenza.

# Modello v1

Prima versione completa: dai dati della partita al CSV da consegnare, con un solo comando. I
parametri sono regolati **solo su dati esterni**, come chiedono le regole (chiarimento degli
organizzatori, novembre 2026).

```bash
# 1. una volta: scarica le 7 partite DFL (~2,7 GB) e regola i parametri (~10 minuti)
./.venv/Scripts/python.exe external/dfl.py
./.venv/Scripts/python.exe external/tune.py

# 2. il file per la nostra partita
./.venv/Scripts/python.exe model/v1.py              # -> evaluation/results/v1_raw.csv

# 3. controllo sui 5 minuti pubblici (solo controllo: non si cambia niente in base a questo)
./.venv/Scripts/python.exe evaluation/run_evaluation.py
```

`model/params_v1.json` è già nella repository, quindi il passo 1 serve solo a rifare la
regolazione: il passo 2 da solo rigenera lo stesso file.

## Come decide

Per ogni frame della partita:

1. **Palla viva o morta?** Una probabilità che cresce quando i giocatori rallentano (media sugli
   ultimi 2 secondi), quando il tracking non vede la palla e quando la palla è fuori dalle linee;
   scende quando la palla si muove veloce.
2. **Quale squadra?** Se la palla si vede e qualcuno le è vicino, conta la squadra del giocatore più
   vicino; più tutti sono lontani (un lancio in volo), meno conta. Senza palla, nessuna indicazione.
3. **Una sequenza, non frame isolati.** Un decodificatore di Viterbi a tre stati (H, A, D) sceglie
   per ogni tempo la sequenza più probabile, pagando un prezzo fisso per ogni cambio di squadra e
   uno per entrare o uscire dalla palla morta. Legge tutto il tempo insieme, quindi un tratto senza
   palla si decide guardando cosa succede prima **e** dopo.

Due fatti pubblicati sono usati così come sono: fuori dai due tempi si scrive `X`, e il secondo
tempo è palla morta fino al calcio d'inizio (FAQ.md).

## Da dove vengono i parametri

| Cosa | Fonte |
|---|---|
| Dati per regolare | 7 partite aperte della DFL (Bundesliga e 2. Bundesliga 2022/23), licenza CC BY 4.0, Bassek et al., *Scientific Data* 2025 |
| Etichette | per ogni frame la DFL indica chi ha la palla e se è in gioco, preso dal suo tracking corretto: lo stesso tipo di etichetta della verità della sfida |
| Come le rendiamo simili alla nostra partita | tolgo la palla nel 41% dei frame a tratti lunghi e faccio sparire i giocatori fino a 20,5 su 22 (numeri pubblicati in README.md e DATA.md); due scenari: buchi casuali, e buchi soprattutto durante le interruzioni |
| Ricerca | un parametro alla volta, più giri, massimizzando lo spellF1 (lo stesso codice dello scorer ufficiale) su 5 partite × 2 scenari |
| Controllo | 2 partite mai usate nella ricerca |

**Non si usano mai** la verità della sfida (nemmeno i 5 minuti pubblici), il video, né annotazioni
fatte a mano.

## Limiti noti

- **Le partite tedesche non sono la nostra.** Tracking diverso, campionati diversi, e un ritmo di
  possessi un po' più fitto di quello che ci si aspetterebbe. Il degrado imita i numeri pubblicati,
  non tutti i difetti del nostro tracking.
- **Il legame tra palla mancante e gioco fermo non è noto** per la nostra partita: per questo ci
  sono due scenari, e i parametri devono reggere in entrambi.
- **I tratti lunghi senza palla** si decidono solo con la velocità dei giocatori e con quello che
  succede ai bordi: è il punto più debole e il prossimo da migliorare.
- **Lo Smart Tagging non è usato:** le partite esterne non lo hanno, quindi non si può regolare
  quanto fidarsene.

## File

| File | Cosa fa |
|---|---|
| `model/features.py` | la tabella dei segnali per frame, uguale per la nostra partita e per quelle esterne |
| `model/v1.py` | il modello: probabilità, decodifica, scrittura del CSV |
| `model/params_v1.json` | i parametri regolati, con la descrizione di come sono stati ottenuti |
| `external/dfl.py` | scarica e legge le partite DFL nello stesso formato del nostro tracking |
| `external/degrade.py` | rende le partite esterne simili al nostro tracking automatico |
| `external/tune.py` | la regolazione e il confronto con la baseline |

Citazione richiesta dalla licenza dei dati DFL: Bassek, M., Weber, H., Rein, R. & Memmert, D. *An
integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data (2025). Dati:
Deutsche Fußball Liga (DFL), CC BY 4.0.
