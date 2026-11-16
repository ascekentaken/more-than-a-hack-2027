# exp03: segnali del tocco e multa di cambio variabile, contro i possessi brevi

**Data:** 9 novembre 2026. **Branch:** `dev`. **Codice:** `external/ball_model.py`,
`external/degrade.py` (opzione `ball="drift"`), `model/features_v3.py`, `external/exp03.py`.
**Risultati generati:** `results.md`, `results.json`, `run.log` in questa cartella.

## Da dove partiamo

Il v2 consegnato (exp02b, variante `mixed`) fa spellF1 0,533 sulle 2 partite tedesche di
controllo. Il suo difetto principale sono i **possessi brevi**: il 30% dei possessi delle partite
tedesche dura meno di 2 s e nello spellF1 ognuno vale come uno di un minuto, ma il v2 ne prevede
solo il 65–77% e unisce gli altri ai vicini. Sui 5 minuti pubblici è successo lo stesso con due
possessi del Barcellona di 0,8 s e 1,3 s.

## Una scoperta prima di cominciare: la palla simulata non si muoveva come la nostra

Per riconoscere i tocchi bisogna vedere la palla cambiare direzione o velocità vicino a un
giocatore. Misurando solo l'**input** del nostro tracking (mai la verità) è emerso che il
simulatore di exp02 sbagliava proprio questo:

| | tocchi al minuto | accelerazione della palla, mediana | svolta, mediana |
|---|---|---|---|
| **nostra partita** | **9,3** | **5,7 m/s²** | **2,5°** |
| tedesche con la palla di exp02 | 31–44 | 77 m/s² | 5,5° |
| tedesche con la palla nuova | 7–11 | 6,8 m/s² | 3,0–3,3° |

La nostra palla è **liscia, ma non sta mai ferma del tutto**. È la firma di una posizione filtrata
con un errore che varia lentamente. Il rumore indipendente a ogni frame di exp02 la rendeva invece
nervosa e inventava cambi di direzione.

La nuova palla (`external/ball_model.py`) aggiunge un errore che deriva lentamente (0,5 m, tempo
caratteristico 2 s) e poi liscia la traiettoria su 21 frame. I tre numeri sono scelti su una
griglia perché velocità, accelerazione, svolte, cambi di velocità e distanza dal giocatore più
vicino coincidano con quelli del nostro tracking. Con la palla nuova anche la quota di controllo
lungo coincide: 21–23% contro 24%.

Questo può riguardare anche il v2 consegnato: il suo segnale più usato è la velocità della palla.
Per questo nel confronto c'è anche il v2 consegnato così com'è (variante S), valutato con la palla
nuova.

## Le due idee

1. **Segnali del tocco** (`model/features_v3.py`, 19 segnali in più):
   - un tocco è un frame in cui la palla vista svolta di più di 25° o cambia velocità di più di
     2,5 m/s, con un giocatore entro 1,5 m; ne teniamo uno per raffica, quello di massimo
     avvicinamento;
   - per ogni squadra: da quanto ha toccato e fra quanto toccherà, quanti tocchi negli ultimi e
     prossimi 2 s, chi ha toccato per ultimo e chi toccherà;
   - controllo: da quanti frame lo stesso giocatore resta entro 2 m dalla palla.

   Sulle partite tedesche la squadra del tocco coincide con la squadra in possesso nell'84–87% dei
   casi con la palla nuova.
2. **Multa di cambio variabile.** Un secondo modello LightGBM stima per ogni frame la probabilità
   che entro 6 frame il possesso passi direttamente da una squadra all'altra. Il decodificatore fa
   pagare il cambio di squadra `max(0, a − b · logit(p))`. Dove un cambio è probabile costa poco,
   altrove costa di più; con b = 0 è la multa fissa del v2.

## Varianti

Tutte allenate come `mixed` (quattro scenari degradati, entrambe le letture casa/trasferta) e
valutate con la palla nuova.

| | segnali | multa di cambio |
|---|---|---|
| A | v2 | fissa |
| B | v2 + tocco | fissa |
| C | v2 | variabile |
| D | v2 + tocco | variabile |
| S | il v2 consegnato, allenato con la palla di exp02 | quella consegnata |

## Protocollo e regola, fissati prima della corsa

- Validazione incrociata sulle 5 partite di regolazione, lasciandone fuori una alla volta. Sceglie
  i prezzi del decodificatore di ogni variante e la vincitrice.
- Vince la variante con lo spellF1 medio più alto sui due scenari `matched`. Fra quelle entro
  0,01 dalla migliore vince quella con il caso peggiore più alto sui quattro scenari degradati.
- Le 2 partite di controllo (J03WN1, J03WR9) sono solo riportate.
- Se vince S, il v2 consegnato resta com'è. Se vince un'altra, verrà riallenata su tutte e 7 le
  partite, rifatta la prova di riproduzione e aggiornato il documento tecnico.
- I 5 minuti pubblici: un solo controllo, dopo la decisione, che non cambia niente.

## Risultati

Corsa del 12–13 novembre 2026, 68 minuti (`run.log`; tabelle complete in `results.md`). Tutte le
varianti sono valutate con la palla nuova. Validazione incrociata (CV) sulle 5 partite di
regolazione, controllo sulle 2 partite mai viste; spellF1 medio dei due scenari `matched`.

| variante | CV matched | CV caso peggiore | controllo matched |
|---|---|---|---|
| S  v2 consegnato (palla di exp02) | 0,495 | 0,464 | 0,455 |
| A  segnali v2, multa fissa | 0,510 | 0,477 | 0,481 |
| B  + tocco, multa fissa | 0,518 | 0,482 | 0,492 |
| C  segnali v2, multa variabile | 0,517 | 0,476 | 0,479 |
| **D  + tocco, multa variabile** | **0,526** | **0,489** | **0,497** |

**Decisione secondo la regola: vince D.** Ha lo spellF1 CV più alto. B e C sono entro 0,01, ma D ha
anche il caso peggiore più alto. Sulle partite di controllo l'ordine è lo stesso: D batte S di
+0,042, in tutti e quattro gli scenari.

Cosa si impara:

- **La palla conta.** Con la palla nuova tutti i numeri sono più bassi di exp02: il problema
  realistico è più difficile. Il v2 consegnato perde di più, perché ha imparato su una palla che
  non somiglia alla nostra. Basta riallenarlo con la palla nuova (A) per guadagnare +0,015 in CV e
  +0,026 sulle partite di controllo.
- **I due miglioramenti si sommano.** I segnali del tocco (B) e la multa variabile (C) valgono
  circa +0,008 ciascuno in CV; insieme (D) +0,016 su A.
- **La multa variabile** scelta è `max(0, 3 − 1,5 · logit(p))`: dove il modello dei cambi vede un
  cambio probabile, cambiare squadra costa quasi niente; dove non lo vede, costa più del 6 fisso.
- D prevede ancora meno possessi del vero (pred/true 0,62–0,66): i possessi brevissimi restano il
  difetto principale.

L'insieme di 5 modelli di exp02c era stato adottato secondo la sua regola, ma è valutato solo con
la palla vecchia e somiglia a S. Non è confrontabile alla pari con D e non lo sostituisce.

Come da protocollo, D viene riallenata su tutte e 7 le partite (`python external/exp03.py --final`,
che scrive `model/v3_frame.txt`, `model/v3_change.txt`, `model/params_v3.json`) e usata da
`model/v3.py`. `evaluation/results/v3_raw.csv` si rigenera identico anche ricalcolando i segnali da
zero (sha256 `72702d22…7363b7b`).

## Controllo unico sui 5 minuti pubblici (dopo la decisione)

| | spellF1 | macroF1 | possessi (veri 17) | cambi entro 1 s | richiamo D |
|---|---|---|---|---|---|
| v2 consegnato | 0,525 | 0,875 | 15 | 9/20 | 0,82 |
| v2 insieme di 5 | 0,494 | 0,833 | 17 | 9/20 | 0,64 |
| v3 (D) | 0,511 | 0,761 | 19 | 7/20 | 0,46 |

Lo spellF1 è dentro il rumore: su 17 possessi l'intervallo bootstrap è largo ±0,15. Il macroF1 cala
invece per un motivo preciso: v3 chiama possesso del Barcellona 560 frame (22 s) di palla morta.
Sulle partite tedesche questo non si vede: il richiamo di D è 0,98 per S, A e D in tutti gli
scenari. La palla morta simulata è quindi molto più facile di quella vera, e i decodificatori
regolati sulle tedesche (anche quello dell'insieme) tendono a chiamarla meno del v2. Questo è un
limite del simulatore, non di una variante, e i 5 minuti non sono stati usati per scegliere.
