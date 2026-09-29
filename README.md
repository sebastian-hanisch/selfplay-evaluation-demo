# Selbstspiel: die Bewertungsfunktion lernt aus eigenen Partien

**[→ Demo live ausprobieren](https://sebastianhanisch-selfplay-evaluation-demo.streamlit.app/)**

Konvergenz-Knoten der Adversarische-Suche-Linie: verbindet Ast A
(**[alpha-beta-demo](https://github.com/sebastian-hanisch/alpha-beta-demo)** →
**[transposition-table-demo](https://github.com/sebastian-hanisch/transposition-table-demo)** – Alpha-Beta
mit Transpositionstabelle) mit Ast B
(**[evaluation-function-demo](https://github.com/sebastian-hanisch/evaluation-function-demo)** –
tiefenbegrenzte Suche mit linearer Bewertungsfunktion). Kind BEIDER Stücke. Vehikel: dasselbe
Mini-Vier-Gewinnt wie die gesamte Linie.

## Warum dieses Problem

evaluation-function-demo hat die Gewichte an **Zufallspartien** gefittet – realitätsfern, weil kein echter
Gegner je zufällig spielt. Das AlphaZero-Prinzip (die Baumsuche selbst bleibt hier Alpha-Beta statt MCTS,
siehe `mcts-demo` für Letzteres) trainiert stattdessen auf Partien der **eigenen, sich verbessernden**
Suche: jede Generation spielt gegen sich selbst, sammelt die besuchten Stellungen mit ihrem exakten Wert und
passt die Gewichte neu an – der nächste Trainingsdatensatz kommt dann von der verbesserten Gewichtung.

## Modell

Wiederverwendet aus den Vorgängerstücken: dieselbe tiefenbegrenzte Alpha-Beta-Suche mit
Zobrist-Transpositionstabelle (Ast A) und dieselben drei Bedrohungsstufen-Merkmale + nicht-negative
kleinste Quadrate (Ast B). Neu ist nur die **Trainingsschleife**:

$$
w_g = \arg\min_{w \geq 0} \sum_{s \in D_g} \big(w^\top x(s) - y_s\big)^2
$$

wobei $D_g$ die Stellungen aus Selbstspielpartien der VORHERIGEN Generation ($w_{g-1}$) sind, nicht aus
Zufallspartien. 6 Generationen, je 1.000 Partien, Suchtiefe 3, $\epsilon$-greedy mit $\epsilon = 0{,}7$ auf
dem 4×3-Trainingsbrett.

## Befunde (gemessen, keine Behauptungen)

**Echter Fund beim Bau**: reines "vernünftiges" Selbstspiel (wenig Zufall) bringt auf diesem kleinen Brett
fast keine Trainingsvielfalt – mit nur 2 zufälligen Eröffnungszügen bestand der gesamte Trainingsdatensatz
aus 100 % Remis-Stellungen (die Suche lenkt danach zuverlässig zurück zum theoretischen Remis dieses
Bretts). Erst $\epsilon$-greedy bei JEDEM Zug (nicht nur den Eröffnungszügen), mit $\epsilon = 0{,}7$ und
1.000 statt 40 Partien je Generation, lieferte genug entscheidende (nicht-remis) Stellungen für ein
sinnvolles Training.

Die Gewichte konvergieren bereits nach der ersten Generation auf dieselbe Ein-Merkmal-Lösung wie
evaluation-function-demo (nur Stufe 3 zählt, Stufe 1 und 2 bleiben bei 0):

| Generation | $w_1$ | $w_2$ | $w_3$ |
|---|---|---|---|
| 0 (Start) | 0,000 | 0,000 | 0,000 |
| 1 | 0,000 | 0,000 | 0,232 |
| 2 | 0,000 | 0,000 | 0,184 |
| 3 | 0,000 | 0,000 | 0,225 |
| 4 | 0,000 | 0,000 | 0,195 |
| 5 | 0,000 | 0,000 | 0,215 |
| 6 | 0,000 | 0,000 | 0,222 |

## Befunde und Korrekturen gegenüber dem Plan

**Ehrlicher, unerwarteter Befund**: erwartet war, dass die realistischeren Selbstspiel-Stellungen einen
Vorsprung gegenüber dem Zufallspartien-Fit aus evaluation-function-demo bringen ("gemessene
Spielstärkeentwicklung ... Vergleich zur Handbewertung" – ursprüngliche DAG-Scoping-Notiz). Gemessen
(Zugübereinstimmung mit dem exakten Optimalzug, 500 Testpositionen auf dem im Training ungesehenen
3×4-Brett, Suchtiefe 3):

| Gewichtung | Übereinstimmung |
|---|---|
| Untrainiert (nur Suche, keine Gewichte) | 94,0 % |
| Zufallspartien-Fit (evaluation-function-demo) | 97,6 % |
| Selbstspiel-Fit (dieses Stück) | 97,6 % |
| Handgewichtet (evaluation-function-demo) | 99,6 % |

Training hilft deutlich gegenüber gar keinem Training (94,0 % → 97,6 %) – aber Selbstspiel- und
Zufallspartien-Training landen exakt auf demselben Wert. Mechanismus: bei nur 3 verfügbaren Merkmalen
entdecken beide Trainingsarten dieselbe Ein-Merkmal-Lösung (nur Stufe 3 zählt, siehe Tabelle oben); sobald
diese Struktur einmal gefunden ist, ändert die Datenquelle (Zufall vs. Selbstspiel) nichts mehr an der
resultierenden Zugrangfolge. Die Handgewichtung (die alle drei Stufen nutzt) bleibt weiterhin vorn – exakt
derselbe Generalisierungs-Befund wie in evaluation-function-demo, hier bestätigt statt widerlegt.

## Ehrliche Grenzen

- **Kein echtes AlphaZero.** Die Baumsuche bleibt Alpha-Beta (nicht MCTS), die Bewertungsfunktion bleibt
  linear (nicht neuronal) – bewusst klein und nachvollziehbar, nicht die volle Architektur.
- **$\epsilon = 0{,}7$ ist unüblich hoch** und eine Notlösung für die Degenerations-Problematik dieses
  kleinen Bretts (siehe Befunde oben), keine allgemeine Empfehlung für Selbstspiel-Training.
- **Die Gewichte konvergieren schon nach einer Generation** – weitere Generationen ändern hier kaum noch
  etwas; auf einem größeren Brett mit mehr Merkmalen wäre vermutlich mehr Entwicklung über die Generationen
  zu beobachten.
- **Nur 3 Merkmale** – bei so wenigen Freiheitsgraden ist ein Unentschieden zwischen zwei Trainingsmethoden
  (siehe oben) plausibler als bei einer reichhaltigeren Merkmalsmenge.

## Tests

44 Tests (`pytest tests/ -v`): Brettmechanik, Merkmalsextraktion, exakter Löser (Kreuzprobe gegen die
Vorgängerstücke), Suche (Konvergenz gegen den exakten Wert, Transpositionstabelle liefert dasselbe Ergebnis
wie ohne), Gewichtsanpassung, Selbstspiel-Erzeugung (jedes gesammelte Label stimmt mit dem exakten Wert
überein), Trainingsschleife, PDF-Export, Visualisierung, Streamlit-Rauchtests (jedes Preset, jede
Bewertungsauswahl, jede Brettgröße, Permalink-Rundlauf inkl. Kein-Überschreiben-Regression). Jede
README-Zahl wird in `tests/test_claims.py` gegen den tatsächlichen Code nachgerechnet, inklusive der vollen
Gewichts-Historie und der vier Übereinstimmungsraten.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `sp_constants.py` | Brettgrößen, Trainingsparameter, gemessene Referenzwerte |
| `sp_game.py` | Brettmechanik (Kopie der Vorgängerstücke) |
| `sp_features.py` | Merkmalsextraktion (Kopie aus evaluation-function-demo) |
| `sp_zobrist.py` | Zobrist-Hashing (Kopie aus transposition-table-demo) |
| `sp_exact.py` | Exakter Löser (Alpha-Beta+Tabelle) – Grundwahrheit für Trainingslabels |
| `sp_search.py` | Tiefenbegrenzte Alpha-Beta-Suche MIT Transpositionstabelle UND Bewertungsfunktion |
| `sp_fit.py` | Gewichtsanpassung (nicht-negative kleinste Quadrate, Kopie aus evaluation-function-demo) |
| `sp_selfplay.py` | Selbstspiel-Erzeugung + generationenweise Trainingsschleife |
| `sp_evaluation.py` | Gecachte Gewichts-Historie, Zahlenformatierung |
| `sp_visualization.py` | Plotly-Brett, Gewichts-Verlauf, Vergleichsdiagramm |
| `sp_presets.py` | Presets, Permalink, Session-Defaults |
| `sp_pdf_export.py` | PDF-Export |

## Bewusst nicht umgesetzt

- Echtes MCTS statt Alpha-Beta (siehe `mcts-demo` für die eigenständige MCTS-Behandlung) – dieses Stück
  zeigt gezielt nur die Selbstspiel-Trainingsschleife, nicht die volle AlphaZero-Architektur.
  Transpositionstabelle bewusst KOMBINIERT (anders als evaluation-function-demo, das sie bewusst wegließ) –
  hier sinnvoll, weil die Suche jetzt wiederholt (jede Generation, jede Partie) läuft.
- Neuronale/nichtlineare Bewertungsfunktionen – die Linie bleibt bei linearen Merkmalen, wie in
  evaluation-function-demo begründet.
- Größere Trainingsbretter (die Degenerations-Problematik – siehe Befunde oben – dürfte auf einem größeren
  Brett schwächer ausfallen, hier nicht systematisch untersucht).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und fpdf2.
