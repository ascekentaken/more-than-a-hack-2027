# exp10 — segnali nuovi per lo stacker (pre-registrazione, scritta prima di eseguire)

**Idea.** I 76 segnali sono centrati sulla palla. Aggiungo allo stacker (oltre al contesto lungo di exp08),
calcolati da colonne già presenti nelle tabelle (`exp06.extra_context`):
- catena di tocchi consecutivi della stessa squadra, prima e dopo (passaggio riuscito vs rimpallo);
- movimento collettivo: spostamento del baricentro di ogni squadra su ±1 s e ±3 s, variazione di
  larghezza/profondità su ±2 s, avanzamento della palla su ±1 s e ±3 s;
- salto della palla attraverso un buco di tracking (distanza e velocità implicita);
- da quanto / per quanto la palla è ferma su una linea (rimessa, angolo, rinvio).

**Varianti.** `X` (stacker su verità DFL com'è) e `XC` (su verità riscritta), stesse pieghe annidate e
stessi checkpoint di exp06/exp08. Da sole sono solo riportate.

**Candidato.** `E6` = media dei quattro stacker di v9 (exp09 E4) + X + XC, decoder ricercato da capo.

**Regola, fissata ora.** E6 sostituisce v9 solo se `CV matched >= 0.7076 + 0.005` e
`CV worst >= 0.6915 − 0.005` (verità a possesso controllato). Altrimenti resta v9.
Holdout solo riportato; 5 minuti pubblici una volta sola e solo se la regola è superata.

## Risultato

X da solo: CV 0.7358, worst 0.7112, holdout 0.729. XC: 0.7334 / 0.7183 / 0.725.
E6: CV 0.7222, worst 0.7021, holdout 0.692 → regola superata rispetto a v9 (0.7076).
Ma X e XC da soli battono E6: vedi exp11 (media dei soli X e XC), che diventa v11.
