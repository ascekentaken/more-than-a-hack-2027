# exp02: modello v2, imparato su dati esterni, con un simulatore tarato sul nostro tracking

**Data:** 5 novembre 2026. **Codice:** `external/challenge_profile.py`, `external/degrade.py` (scenari
`matched_*`), `model/features_v2.py`, `external/exp02.py`, `model/v2.py`. **Risultati generati:**
`results.md`, `results.json`, `run.log` in questa cartella.

## Da dove partiamo

Il v1 sui 5 minuti pubblici (un controllo, non usato per regolare niente): spellF1 0,471,
macroF1 0,817. Numero e durata dei possessi giusti (17 su 17), ma **confini imprecisi**: lo
spellF1 crolla da 0,59 (IoU 0,3) a 0,24 (IoU 0,7), solo 5 cambi su 20 presi entro un secondo,
ritardo medio di 8 frame. Circa il 10% dei frame con la squadra sbagliata, palla morta un po'
troppo frequente. Sono gli stessi difetti che si vedevano sulle partite tedesche.

## Due ipotesi

**1. Il simulatore era sbagliato.** Misurando solo l'input del nostro tracking (mai la verità) è
emerso che:

- la palla manca soprattutto quando i giocatori sono lenti: 1,17 m/s di media durante i buchi
  contro 2,16 m/s quando la palla si vede. La percentuale di palla mancante per fascia di velocità
  ha la stessa forma della percentuale di palla morta nelle partite tedesche;
- quindi il vecchio scenario "buchi casuali" non somigliava alla nostra partita, e quello
  "interruzioni" aveva buchi troppo lunghi;
- il rumore sulla palla era troppo forte: velocità mediana della palla 13,6 m/s nel simulatore
  contro 5,1 m/s nel nostro tracking.

I nuovi scenari `matched_speed` e `matched_dead` riproducono queste statistiche (vedi
`external/degrade.py`). Il confronto diretto:

| | velocità giocatori nei buchi | durata mediana buchi | buchi ogni 90' | tratti visibili sotto 3 s | velocità palla |
|---|---|---|---|---|---|
| **nostra partita** | **1,17** | **10,7 s** | **127** | **11%** | **5,1** |
| vecchio "buchi casuali" | 1,85 | 9,5 s | 142 | | 13,6 |
| vecchio "interruzioni" | 1,33 | 15,7 s | 92 | | 14,2 |
| `matched_speed` | 1,44–1,54 | 10,7–11,5 s | 145–153 | 33–38% | 3,8–4,4 |
| `matched_dead` | 1,19–1,30 | 16–19 s | 95–102 | 8–9% | 5,1–5,8 |

`matched_dead` è il più somigliante; `matched_speed` resta come prova di robustezza.

**2. Il modello vedeva troppo poco.** Il v1 ha una regola scritta a mano con 12 numeri. Il v2
impara da 7 partite tedesche (circa 1 milione di frame etichettati) con 57 segnali, tra cui:

- chi è stato vicino alla palla per ultimo e chi lo sarà dopo, e da quanto tempo;
- la velocità dei giocatori negli ultimi 2 secondi e nei prossimi 2, che segna il momento esatto
  di una ripresa o di un'interruzione;
- da quanto la palla manca o manca ancora, e da quanto è uscita;
- la forma delle squadre.

Poi un decodificatore con tre prezzi separati: cambio di squadra, entrata in palla morta, uscita
dalla palla morta. La partita viene letta anche con casa e trasferta scambiate, e le due risposte
vengono mediate: così il modello non ha preferenze per una squadra.

## Protocollo, fissato prima di vedere i risultati

1. **5 partite di regolazione, lasciandone fuori una alla volta.** Per ogni partita, LightGBM è
   allenato sulle altre 4 e la predice. Con queste previsioni "mai viste" si scelgono i tre prezzi
   del decodificatore.
2. **2 partite di controllo**, mai usate per nessuna scelta. Confronto tra la baseline ufficiale,
   il v1 consegnato, il v1 rifatto sugli scenari nuovi (per separare l'effetto del simulatore da
   quello del modello) e il v2.
3. **Regola di decisione:** si consegna il metodo con lo spellF1 medio più alto sulle 2 partite di
   controllo nei due scenari `matched`. Gli scenari vecchi e i dati puliti servono solo come prova
   di robustezza.
4. Il v2 finale viene riallenato su tutte e 7 le partite con le stesse impostazioni.
5. Solo dopo la decisione: un controllo sui 5 minuti pubblici, riportato e non usato per cambiare
   niente.

## Regole rispettate

- Verità della sfida, video e annotazioni non vengono mai letti dal codice di allenamento o di
  regolazione.
- Della nostra partita si usano solo statistiche dell'input (il tracking), come permette la regola
  "running automatic checks using only the provided data". Sono ricalcolate in locale e non
  vengono salvate nella repository.
- Dati esterni: DFL/IDSSE, licenza CC BY 4.0 (Bassek et al., *Scientific Data* 2025).

## Risultati sulle partite tedesche

### exp02: v2 contro v1, sulle 2 partite di controllo (`results.md`)

| metodo | spellF1 `matched_dead` | spellF1 `matched_speed` | @IoU 0,7 (`matched_dead`) | cambi presi entro 1 s (`matched_dead`) |
|---|---|---|---|---|
| baseline smooth50 | 0,299 | 0,215 | 0,148 | 0,232 |
| v1 consegnato | 0,418 | 0,332 | 0,265 | 0,398 |
| v1 rifatto sugli scenari nuovi | 0,456 | 0,349 | 0,297 | 0,467 |
| **v2** | **0,577** | **0,499** | **0,415** | **0,634** |

- Il simulatore nuovo da solo vale circa +0,03 (v1 rifatto contro v1 consegnato).
- Il modello nuovo vale circa +0,13: è lì il grosso del guadagno.
- La validazione incrociata (0,60 / 0,53) e le partite di controllo (0,58 / 0,50) concordano:
  il modello generalizza a partite mai viste.
- I segnali più usati: velocità della palla nell'ultimo secondo (20%), da quanto ciascuna squadra
  non tocca la palla (28% in tutto), velocità dei giocatori su 8 secondi e nei prossimi 2.

Il punto debole: nello scenario vecchio "buchi casuali", dove la palla mancante non ha niente a
che fare con la palla morta, il v2 scendeva a macroF1 0,62 (il v1 faceva 0,72).

### exp02b: tre varianti, scelte solo con la validazione incrociata (`variants.md`)

| variante | spellF1 sui due `matched` (validazione incrociata) | caso peggiore sui 4 scenari | controllo sui `matched` |
|---|---|---|---|
| base | 0,568 | 0,356 | 0,538 |
| **mixed** (allenato anche sugli scenari vecchi) | 0,561 | **0,490** | 0,533 |
| relative (velocità relative alla partita) | 0,572 | 0,356 | 0,541 |
| mixed + relative | 0,558 | 0,486 | 0,534 |

**Decisione: `mixed`, con una deviazione dichiarata dalla regola.** La regola scritta prima della
corsa ("il migliore entro 0,01, a parità il più robusto") indicava `base` (o `relative`): `mixed`
resta fuori dal margine per 0,001. Abbiamo scelto `mixed` lo stesso perché:

- perde 0,007 dove il tracking somiglia al nostro;
- guadagna 0,13 nel caso peggiore, e nello scenario "buchi casuali" il macroF1 passa da 0,64 a
  0,86.

È un'assicurazione: le misure sull'input dicono che i nostri buchi sono legati alla palla morta, ma
non possiamo verificarlo senza guardare la verità. Le velocità relative non aiutano in modo
misurabile, quindi non le usiamo. La scelta usa solo la validazione incrociata: né le partite di
controllo né i 5 minuti pubblici hanno contribuito.

Prezzi del decodificatore scelti: cambio di squadra 6, entrata in palla morta 90, uscita 10, più
1 alla probabilità di palla morta.

### Il file consegnabile

Il modello finale è `mixed` riallenato su tutte e 7 le partite, quindi `model/v2_lgb.txt` e
`model/params_v2.json`. Il comando `python model/v2.py` produce `evaluation/results/v2_raw.csv`
(valido): 32% casa, 19% trasferta, 49% palla morta, 330 possessi di 9,8 s di media. Il v1
produceva 352 possessi e il 53% di palla morta.

Decisione registrata qui **prima** di lanciare il controllo sui 5 minuti pubblici.

## Controllo sui 5 minuti pubblici (dopo la decisione, non usato per cambiare niente)

| | v1_raw | **v2_raw** |
|---|---|---|
| spellF1 | 0,471 | **0,525** |
| spellF1 a IoU 0,3 / 0,5 / 0,7 | 0,59 / 0,53 / 0,24 | **0,81** / 0,44 / **0,31** |
| macroF1 | 0,817 | **0,875** |
| cambi presi entro 1 s | 5 su 20 | **9 su 20** |
| ritardo medio sui cambi | +8,4 frame | **−0,3 frame** |
| squadra giusta (palla viva) | 81,5% | **90,1%** |
| palla viva/morta F1 | 0,939 | **0,962** |
| possessi (veri: 17) | 17 | 15 |

Va nella stessa direzione delle partite tedesche, che era quello che speravamo di vedere: i
confini sono più precisi (ritardo azzerato, il doppio dei cambi presi) e la squadra è giusta più
spesso. Con 17 possessi però il rumore è grande (intervallo bootstrap 5–95% dello spellF1: v2
0,36–0,68, v1 0,31–0,62), quindi questo controllo da solo non dimostra niente. La prova sono le
partite tedesche di controllo.

## Rifare tutto da zero

```bash
python external/dfl.py                  # le 7 partite DFL (una volta, ~2,7 GB)
python external/challenge_profile.py    # le statistiche dell'input della nostra partita
python external/exp02.py                # exp02, ~10 min  -> results.md
python external/exp02b.py               # exp02b, ~36 min -> variants.md
python external/exp02b.py --final mixed # modello finale
python model/v2.py                      # evaluation/results/v2_raw.csv
```
