# exp05 -- quattro modifiche a v4w, ognuna da sola

Pre-registrazione, scritta il 2026-11-15 prima di lanciare `python external/exp05.py`.

Base: **v4w** (exp04w, W05): exp04 variante D con il modello per frame addestrato con pesi (1/L)^0.5.
Ogni variante cambia una sola cosa e riusa il resto di v4w; i prezzi del decoder sono ricercati di nuovo per ognuna.

| variante | cosa cambia |
|---|---|
| WC | anche il modello dei cambi è addestrato con i pesi (1/L)^0.5 |
| WA | il modello per frame è addestrato su tre estrazioni del simulatore di degrado (0, 1, 2) invece di una: dati degradati più vari in un solo modello, non un insieme |
| WB | modello per frame più grande: num_leaves 127, min_child_samples 100, 800 alberi |
| WM | le due partite Metrica aggiunte all'addestramento di ogni fold (modello per frame pesato come v4w; modello dei cambi quello di exp04m) |

**Regola di decisione, fissata prima:** la variante con lo spellF1 leave-one-match-out più alto sulle partite TUNE (media dei due scenari matched) sostituisce v4w solo se supera v4w di almeno 0.005 e il suo caso peggiore sui quattro scenari degradati non è più di 0.005 sotto quello di v4w. Le partite HOLDOUT sono solo riportate. I 5 minuti pubblici: un solo controllo, dopo la decisione e solo se una variante è adottata.

Le varianti girano in quest'ordine: WC, WA, WB, WM. Se il tempo prima della consegna (17 novembre) non basta, si decide tra quelle finite e qui si scrive quali mancano.

Seconda fase, solo se c'è tempo: se due o più varianti passano la regola da sole, si prova la loro combinazione; la combinazione è adottata solo se supera la migliore variante singola di almeno 0.005 (CV matched) con lo stesso vincolo sul caso peggiore.

## Risultato

Tutte e quattro le varianti finite (75 min). Regola -> **si tiene v4w**.

| | CV matched | CV worst | holdout matched |
|---|---|---|---|
| v4w | 0.5653 | 0.5365 | 0.526 |
| WC | 0.5671 | 0.5383 | 0.527 |
| WA | **0.5702** | 0.5475 | 0.540 |
| WB | 0.5654 | 0.5424 | 0.521 |
| WM | 0.5614 | 0.5499 | 0.544 |

La migliore, WA (tre estrazioni del simulatore), supera v4w di 0.0049 in CV matched: sotto la soglia di 0.005 fissata prima, quindi non è adottata, anche se migliora anche il caso peggiore (+0.011) e l'holdout (+0.014). Nessuna variante passa da sola, quindi la seconda fase (combinazioni) non si fa. Nessun controllo sui 5 minuti pubblici.
