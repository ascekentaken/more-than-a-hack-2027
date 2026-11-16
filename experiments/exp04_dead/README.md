# exp04: una palla morta realistica nel simulatore

**Data:** 14 novembre 2026. **Branch:** `dev`. **Codice:** `external/dead_model.py`,
`external/degrade.py` (opzione `dead="real"`), `external/exp04.py` (riusa `external/exp03.py`).
**Risultati generati:** `results.md`, `results.json`, `run.log` in questa cartella.

## Da dove partiamo

exp03 ha scelto v3 (segnali del tocco e multa di cambio variabile): sulle partite tedesche di
controllo batteva il v2 consegnato di +0,042. Il controllo unico sui 5 minuti pubblici, fatto dopo
la decisione, ha però mostrato un punto cieco. v3 chiama "possesso Barcellona" 22 s di palla morta
e riconosce solo il 46% della palla morta, contro l'82% del v2. Sulle partite tedesche invece tutte
le varianti la riconoscono al 98%: la palla morta simulata è troppo facile.

## Perché: nei dati tedeschi la palla morta è congelata

Durante le interruzioni la palla dei dati tedeschi resta **perfettamente ferma** nel punto in cui è
uscita (0 m di spostamento in tutte le interruzioni, spesso fuori dal campo o su una linea) e alla
ripresa salta nel punto di battuta (più di 5 m nel 60% dei casi). È una convenzione dei dati
corretti, non quello che vede una telecamera. Il modello impara "palla ferma su una linea = palla
morta", un indizio che nel nostro tracking quasi non esiste.

Misure solo sull'**input** (nessuna verità, nessun video):

| | nostra partita | tedesche degradate (exp03) |
|---|---|---|
| palla vista fuori dal campo | 3,5% | 6–16% |
| palla vista entro 2 m da una linea | 7% | 12–29% |
| palla ferma, fra quelle viste | 4,8% | 4–14% |
| palla che ricompare dopo un buco vicino a una linea | 12% | 36–46% |
| palla che ricompare dopo un buco ferma | 2% | 5–22% |

Durante il gioco vivo le tedesche somigliano già alla nostra partita (palla vicino a una linea
6–7%): il difetto è solo nella palla morta.

## La palla morta nuova (`external/dead_model.py`)

Ogni interruzione delle partite tedesche diventa:

1. **la palla rotola fuori:** continua con la sua ultima velocità e rallenta, per un tempo medio
   `roll_s`;
2. **sparisce,** tranne in una quota `carried` delle interruzioni, in cui il tracker la segue mentre
   viene riportata al punto di battuta;
3. **ricompare ferma nel punto di battuta** per un tempo medio `placed_s` prima della ripresa. Il
   tempo è moltiplicato per `edge_factor` se il punto è vicino a una linea (rimesse, angoli).

4. **viene riagganciata in ritardo:** dopo la ripresa la palla resta invisibile per un tempo medio
   `pickup_s`, moltiplicato per `inside_factor` se il punto di battuta non è vicino a una linea.

I numeri sono scelti su una griglia perché le statistiche della palla vista sulla partita tedesca
J03WMX degradata coincidano con quelle del nostro tracking. Le statistiche sono: fuori campo, vicino
a una linea, ferma, vista mentre i giocatori sono lenti, dove ricompare dopo un buco. Poi i buchi
della palla sono completati come in exp02/exp03, perché la palla manchi nella stessa misura in ogni
fascia di velocità dei giocatori.

Le griglie sono state tre:

- la prima, senza riaggancio, lasciava la palla ricomparire vicino a una linea nel 38% dei casi
  contro il 12%: è così che è nato il passo 4;
- la seconda metteva il riaggancio migliore sul bordo (3 s);
- la terza lo allarga a 4 e 6 s e conferma 3 s.

Risultato: `roll_s` 0,5, `placed_s` 4, `edge_factor` 0, `carried` 0,35, `pickup_s` 3,
`inside_factor` 1 (errore 0,141, contro 0,231 della prima griglia).

| statistica della palla vista (solo input) | nostra partita | palla morta nuova | congelata (exp03) |
|---|---|---|---|
| fuori dal campo | 0,035 | 0,037 | 0,090 |
| entro 2 m da una linea | 0,070 | 0,100 | 0,182 |
| ferma | 0,048 | 0,032 | 0,071 |
| ferma, con giocatori lenti | 0,269 | 0,259 | 0,386 |
| vista con giocatori lenti | 0,128 | 0,090 | 0,098 |
| ricompare vicino a una linea | 0,122 | 0,169 | 0,407 |
| ricompare ferma | 0,023 | 0,076 | 0,154 |
| ricompare con giocatori lenti | 0,183 | 0,119 | 0,155 |

Un limite noto: nello scenario `matched_dead`, che già nascondeva quasi tutte le interruzioni, ora
la palla si vede il 44–50% del tempo contro il 59% del nostro tracking. È uno scenario pessimista;
in `matched_speed` la palla si vede il 53–62% del tempo.

## Varianti, protocollo e regola, fissati prima della corsa

Identici a exp03, cambia solo la palla morta del simulatore:

- varianti S (v2 consegnato), A, B, C, D (segnali v2 o + tocco, multa fissa o variabile), tutte
  allenate e valutate con la palla nuova e la palla morta nuova;
- validazione incrociata sulle 5 partite di regolazione, lasciandone fuori una alla volta: sceglie i
  prezzi del decodificatore e la vincitrice;
- vince lo spellF1 medio più alto sui due scenari `matched`; fra le varianti entro 0,01 dalla
  migliore vince quella con il caso peggiore più alto sui quattro scenari;
- le 2 partite di controllo (J03WN1, J03WR9) sono solo riportate;
- se vince S, il v2 consegnato resta com'è; se vince un'altra variante, viene riallenata su tutte e
  7 le partite (`python external/exp04.py --final` → `model/params_v4.json`), rifatta la prova di
  riproduzione e aggiornato il documento tecnico;
- i 5 minuti pubblici: un solo controllo, dopo la decisione, che non cambia niente.

Una nota onesta: questo esperimento nasce dal controllo pubblico di exp03. Il controllo ha indicato
*dove* guardare, la palla morta. Ma la correzione usa solo l'input del nostro tracking e i dati
tedeschi: nessun numero è scelto sui 5 minuti pubblici.

## Risultati

Corsa del 14 novembre 2026, 59 minuti (`run.log`; tabelle complete in `results.md`). SpellF1 medio
dei due scenari `matched`.

| variante | CV matched | CV caso peggiore | controllo matched |
|---|---|---|---|
| S  v2 consegnato | 0,495 | 0,478 | 0,449 |
| A  segnali v2, multa fissa | 0,526 | 0,512 | 0,478 |
| B  + tocco, multa fissa | 0,536 | 0,516 | 0,487 |
| C  segnali v2, multa variabile | 0,534 | 0,522 | 0,486 |
| **D  + tocco, multa variabile** | **0,544** | **0,526** | **0,495** |

**Decisione secondo la regola: vince D**, come in exp03. Ha lo spellF1 CV più alto. B e C sono entro
0,01, ma D ha anche il caso peggiore più alto. Sulle partite di controllo D batte S di +0,046 e lo
batte in tutti e quattro gli scenari. L'ordine delle varianti è lo stesso di exp03: il risultato non
dipende da come si simula la palla morta.

Cosa cambia rispetto a exp03: il decodificatore scelto per D è più prudente sulla palla morta
(`dead_bias` 1,0 come il v2, contro 0,5 in exp03; multa di cambio `max(0, 2 − 2 · logit(p))`).
Sulle tedesche la palla morta resta riconosciuta al 98–99%: il simulatore nuovo non la rende più
difficile in media, ma toglie l'indizio della palla congelata.

D è stata riallenata su tutte e 7 le partite (`python external/exp04.py --final` →
`model/v4_frame.txt`, `model/v4_change.txt`, `model/params_v4.json`) e si usa con
`python model/v3.py --params model/params_v4.json --out evaluation/results/v4_raw.csv`. Il file si
rigenera identico anche ricalcolando i segnali da zero (sha256 `53449c32…17caf1b9`).

## Controllo unico sui 5 minuti pubblici (dopo la decisione)

| | spellF1 | macroF1 | possessi (veri 17) | cambi entro 1 s | richiamo D | squadra giusta, palla viva |
|---|---|---|---|---|---|---|
| v2 consegnato | 0,525 | **0,875** | 15 | **9/20** | **0,82** | **0,90** |
| v3 (exp03) | 0,511 | 0,761 | 19 | 7/20 | 0,46 | 0,89 |
| **v4 (exp04)** | **0,533** | 0,838 | 16 | 6/20 | 0,75 | 0,88 |

Il buco sulla palla morta di v3 è in gran parte chiuso (richiamo 0,46 → 0,75). Lo spellF1 di v4 è il
più alto, ma con 17 possessi l'intervallo bootstrap è 0,36–0,69: le tre versioni non si distinguono
su questi 5 minuti. Sul macroF1 (criterio di spareggio) il v2 resta davanti.

## Due esperimenti successivi su v4 (14 novembre 2026, regole fissate prima della corsa)

Entrambi usano la stessa valutazione di exp04: validazione incrociata sulle 5 partite tedesche di
regolazione, estrazione 0 del simulatore. Entrambi devono battere la D di exp04 con la stessa regola
di exp02c: almeno +0,005 di spellF1 CV (media dei due scenari `matched`) e un caso peggiore non più
basso di 0,005. Le 2 partite tedesche di controllo sono solo riportate.

**exp04c, insieme di 5 modelli v4** (`external/exp04c.py`; risultati in `ensemble.md`). Si
allenano 5 coppie (modello dei frame e modello dei cambi): la coppia k usa l'estrazione k del
simulatore e il seme LightGBM 2027 + k. Le probabilità si mediano e i prezzi del decodificatore si
cercano di nuovo.

**exp04m, due partite Metrica in più** (`external/metrica.py`, `external/exp04m.py`; risultati in
`metrica.md`). Le partite di esempio 1 e 2 di Metrica Sports (github.com/metrica-sports/sample-data,
da citare come fonte) vengono degradate con lo stesso simulatore e aggiunte all'allenamento di ogni
fold. Le etichette H/A/D sono ricostruite dal registro degli eventi:
- palla morta da un'uscita, un fallo, un gol o un fuorigioco fino al calcio piazzato successivo;
- negli altri frame il possesso va alla squadra dell'ultimo evento giocato sulla palla.

Il controllo automatico è buono: la squadra del giocatore più vicino coincide con quella in possesso
nel 95% dei frame, contro il 90–93% dei dati tedeschi. Le etichette sono però un po' più larghe:
palla morta 36–37% contro 42–52%, e circa 450 cambi di etichetta a partita contro circa 610.
Viene anche riportato, senza usarlo per decidere, come va su Metrica un modello allenato solo sulle
tedesche.

**exp04w, ogni possesso pesa uguale in allenamento** (`external/exp04w.py`; risultati in
`weights.md`). Sulle partite tedesche di controllo v4 perde soprattutto i possessi sotto i 2 s, e
la colpa è soprattutto del modello dei frame (`analysis/dfl_errors.py`): nel 60% dei possessi brevi
persi non dà mai più di 0,5 alla squadra giusta. Impara frame per frame, quindi un intercetto di
1 s vale 25 frame su 140.000, mentre nello spellF1 conta quanto un possesso di un minuto.

Ogni frame di allenamento riceve un peso `(1 / L)^alpha`, dove L è la durata in frame del tratto
vero (H, A o D) in cui cade, con un minimo di 5. Due varianti: W05 (alpha 0,5) e W10 (alpha 1,
cioè ogni tratto pesa uguale in totale). Il modello dei cambi resta quello di v4. Adotta la
migliore W solo se batte D di +0,005 in CV, con un caso peggiore non più basso di 0,005.

### Risultato di exp04m (14 novembre 2026)

| | CV matched | CV caso peggiore | controllo matched |
|---|---|---|---|
| D (v4, solo tedesche) | 0,544 | 0,526 | 0,495 |
| DM (tedesche + Metrica) | 0,547 | 0,528 | 0,506 |

**Decisione secondo la regola: resta D.** Il guadagno in CV (+0,003) è sotto la soglia di +0,005.
Sulle partite tedesche di controllo DM fa +0,011, ma quelle sono solo riportate e non decidono.

Il risultato più utile è un altro. **v4 allenato solo sulle tedesche funziona anche sui dati di un
altro fornitore:** sulle due partite Metrica fa spellF1 0,53–0,57 nei quattro scenari, cioè quanto
o più che sulle tedesche mai viste (0,49–0,52). Il modello non è legato al formato o alle
convenzioni della DFL. Aggiungendo l'altra partita Metrica sale di poco (0,55–0,58).

### Risultato di exp04w (14 novembre 2026)

| | CV matched | CV caso peggiore | controllo matched | possessi previsti / veri (controllo) |
|---|---|---|---|---|
| D (v4, senza pesi) | 0,544 | 0,526 | 0,495 | 0,66 |
| **W05 (alpha 0,5)** | **0,565** | **0,536** | 0,526 | 0,67 |
| W10 (alpha 1) | 0,565 | 0,535 | 0,540 | 0,75 |

**Decisione secondo la regola: W05** (CV 0,5653 contro 0,5647 di W10; +0,021 su D, caso peggiore
+0,010). È il guadagno più grande dall'inizio, ed è coerente in tutti e quattro gli scenari e sulle
partite di controllo (+0,031). W10 fa ancora meglio sulle partite di controllo (+0,045) e prevede
più possessi, ma la regola sceglie sulla CV, dove le due sono pari: resta W05.

W05 è stata riallenata su tutte e 7 le partite (`python external/exp04w.py --final` →
`model/v4w_frame.txt`, `model/params_v4w.json`; il modello dei cambi è `model/v4_change.txt`). Si usa
con `python model/v3.py --params model/params_v4w.json --out evaluation/results/v4w_raw.csv`, che si
rigenera identico anche ricalcolando i segnali da zero (sha256 `109cc7c7…ef61e9a3c`).

Controllo unico sui 5 minuti pubblici, dopo la decisione:

| | spellF1 | macroF1 | possessi (veri 17) | cambi entro 1 s | richiamo D | squadra giusta, palla viva |
|---|---|---|---|---|---|---|
| v2 consegnato | 0,525 | 0,875 | 15 | 9/20 | 0,82 | 0,90 |
| v4 | 0,533 | 0,838 | 16 | 6/20 | 0,75 | 0,88 |
| **v4w (W05)** | **0,575** | 0,860 | 15 | 7/20 | 0,75 | 0,90 |

Su 17 possessi l'intervallo bootstrap dello spellF1 è ancora largo (0,40–0,74), ma per la prima
volta il controllo pubblico va nella stessa direzione della CV tedesca.

### Risultato di exp04c (insieme di 5 v4, senza pesi)

Regola pre-registrata -> **si tiene il v4 singolo**: CV matched 0.546 contro 0.544 (serviva +0.005), CV worst 0.525 contro 0.526, holdout 0.510 contro 0.495 (solo riportato). Superato comunque da exp04w; nessun controllo sui 5 minuti pubblici, perché non è stato adottato.
