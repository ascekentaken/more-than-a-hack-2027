# Annotazione a mano

Strumento e protocollo per segnare a mano chi ha la palla in alcuni spezzoni della partita. Le
annotazioni servono a **misurare** il modello e, se gli organizzatori lo confermano, ad
**addestrarlo**. Non si consegnano mai: le etichette consegnate devono uscire dall'algoritmo
(`RULES.md`).

## Avvio

Dalla cartella `more-than-a-hack-2027`. In PowerShell si usano le barre `\`, in WSL le `/`.

```bash
# una volta sola, per tutto il team: sceglie gli spezzoni (sempre gli stessi, seme fisso)
./.venv/Scripts/python.exe annotation/make_plan.py

# ogni volta: il tuo nome in minuscolo, poi apri http://127.0.0.1:8765 nel browser
./.venv/Scripts/python.exe annotation/server.py --annotator fosco
```

Il piano deve essere **lo stesso per tutti**. Lo genera una persona e lo passa agli altri (il file
`annotation/labels/plan.json`), oppure ognuno lo rigenera con lo stesso comando: il seme è fisso,
quindi esce identico.

Ogni modifica si salva da sola in `annotation/labels/<nome>/<spezzone>.json`. Si può chiudere e
riprendere quando si vuole: la pagina ricorda lo spezzone e il frame a cui eri arrivato.

## Gli spezzoni

| Tipo | Cosa | A cosa serve |
|---|---|---|
| `calib` | i 5 minuti pubblici | **tutti lo fanno per primo**, poi premono "Confronta con la verità" e imparano come gli organizzatori intendono il possesso. Non serve a regolare il modello |
| `tune` (regolazione) | 2 minuti ciascuno | per regolare soglie e addestrare il modello |
| `test` (controllo) | 2 minuti ciascuno | **solo per il controllo finale**: guardatene i punteggi il meno possibile |

Metà degli spezzoni è scelta dove il tracking perde spesso la palla (`gappy`), perché è il caso
che i 5 minuti pubblici non mostrano quasi mai.

**Chi fa cosa.** Ognuno fa `calib`, poi una parte degli spezzoni. Almeno 2 spezzoni vanno fatti da
**due persone**, senza guardare il lavoro dell'altra: servono a misurare quanto siamo d'accordo.

## Come si annota

Non si etichetta ogni frame: si segnano solo i **punti di cambio**. Vai al frame giusto e premi
`H`, `A` o `D`: da quel frame in poi lo stato è quello, fino al cambio successivo.

1. **Il primo frame dello spezzone** deve avere un'etichetta: premi `Home`, guarda chi ha la palla
   (o se il gioco è fermo) e premi la lettera.
2. Guarda il video a 1× o 0,5× (tasto `Spazio`). Quando succede qualcosa, fermati.
3. Torna indietro con `←` e `Shift+←` fino al **frame esatto** (vedi le regole sotto) e premi la
   lettera.
4. Se non sei sicuro, premi `F` (dubbio) e, se serve, `N` per una nota breve.
5. A fine spezzone, "Segna spezzone come completato".

La barra sotto il video mostra quello che hai segnato (blu Barcellona, rosso Athletic, grigio palla
morta) e, sotto, in rosso scuro, i tratti in cui il **tracking** non vede la palla.

Il tasto `O` disegna il tracking sopra il video. **Annota con l'overlay spento**: l'etichetta deve
venire da quello che vedi, non da quello che dice il tracking che poi il modello userà.

## Dove mettere il segno

Le definizioni vengono da `CHALLENGE.md`. Dove il protocollo e la verità pubblica non concordano,
**ha ragione la verità pubblica**: usa lo spezzone `calib` per capirlo.

| Situazione | Cosa segni | Dove esattamente |
|---|---|---|
| Una squadra conquista la palla **con controllo** (intercetto con controllo, contrasto vinto, colpo di testa che arriva a un compagno) | la nuova squadra, `H` o `A` | il frame del **primo tocco della squadra** che poi tiene la palla, purché tra quel tocco e il controllo la palla non resti libera |
| Il tocco della nuova squadra lascia la palla **libera** (rimpalli, nessuno la controlla), poi la nuova squadra la prende | la nuova squadra | il frame in cui un suo giocatore **la prende** |
| Palla contesa o vagante | **niente**: resta la squadra che l'aveva | — |
| Tocco senza controllo, deviazione che non intercetta, duello aereo senza conquista | **niente** | — |
| La palla esce dal campo (linea superata del tutto) | `D` | il frame in cui la palla supera la linea |
| Fischio dell'arbitro (fallo, fuorigioco, infortunio) | `D` | il frame del fischio; senza audio, quando il gioco si ferma (i giocatori rallentano e si girano) |
| Gol | `D` | quando la palla supera la linea di porta |
| Ripresa del gioco (rimessa, punizione, corner, rinvio, calcio d'inizio) | la squadra che batte | il frame in cui la palla viene **colpita o lanciata** |

Lo strumento non ha l'audio. Per un fischio dubbio, apri lo stesso momento nel video con VLC
(l'orario del video è sotto l'immagine).

**Possessi brevi.** Contano anche quelli di meno di un secondo, se c'è controllo: nella verità
pubblica ce n'è uno di 0,8 s.

## Dopo la calibrazione

"Confronta con la verità" mostra il tuo spellF1, i possessi tuoi e veri e, per ogni cambio vero,
di quanti frame il tuo segno è in anticipo (negativo) o in ritardo (positivo). Se sei
sistematicamente in ritardo o in anticipo di qualche frame, correggi il modo in cui scegli il frame,
non i singoli segni. Condividete i risultati di `calib` nel team e mettetevi d'accordo sui casi
dubbi prima di passare agli altri spezzoni.

## Esportare

```bash
./.venv/Scripts/python.exe annotation/export.py
./.venv/Scripts/python.exe evaluation/run_evaluation.py --gt annotation/labels/out/labels_tune.csv --bootstrap 1000
```

`export.py` scrive `labels_tune.csv` e `labels_test.csv` nello stesso formato della verità pubblica,
più un rapporto sull'accordo tra annotatori.

## Attenzione

- `annotation/labels/` è **escluso da git**: sono etichette della partita, come le submission.
- Non mandate fotogrammi, annotazioni o tabelle a un assistente AI online.
- **Riproducibilità.** Se il modello finale viene addestrato su queste annotazioni, il repository
  deve poter rigenerare il file consegnato, e quindi le annotazioni servono anche a chi lo
  rigenera. Come consegnarle (nel repository o a parte) va chiesto agli organizzatori insieme alla
  domanda su quanto annotare.
