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

*(dopo la corsa)*
