# exp12 — continuità di squadra per lo stacker (pre-registrazione, scritta prima di eseguire)

**Idea.** exp10 ha mostrato che i segnali sulla sequenza dei tocchi valgono molto. Il possesso
"controllato" è per definizione una cosa che dura: aggiungo (`exp06.team_context`), sempre da colonne
già nelle tabelle:
- da quanti frame / per quanti frame ancora la stessa squadra è l'ultima ad aver toccato, quella in
  controllo, l'ultima vicina alla palla (e le versioni "prossima");
- quota di tempo di ogni squadra come ultima a toccare / in controllo / vicina, nei 5 e 10 s prima e dopo.

**Varianti.** `Y` (verità DFL com'è) e `YC` (verità riscritta): gli stacker X e XC di exp10 più questi
segnali. Stesse pieghe annidate e stessi checkpoint.

**Candidato.** `EY` = media di Y e YC, decoder ricercato da capo.

**Regola, fissata ora.** EY sostituisce v11 (exp11 EX) solo se `CV matched >= 0.7386 + 0.005`,
`CV worst >= 0.7222 − 0.005` e holdout non sotto 0.729. Altrimenti resta v11.
5 minuti pubblici: una volta sola e solo se la regola è superata.

## Risultato

| | CV matched | CV worst | holdout |
|---|---|---|---|
| v11 (exp11 EX) | 0.7386 | 0.7222 | 0.729 |
| Y | 0.7371 | 0.7174 | 0.725 |
| YC | 0.7343 | 0.7202 | 0.724 |
| EY (media Y + YC) | 0.7396 | 0.7229 | 0.722 |

+0.0010 in CV (soglia 0.005) e holdout più basso: **resta v11**. Nessun controllo pubblico.
I segnali di continuità non aggiungono informazione a quelli di exp10.
