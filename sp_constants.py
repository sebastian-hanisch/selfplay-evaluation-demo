"""Regler-Grenzen, feste Annahmen, Farben."""

WIN_LENGTH = 4
EMPTY = 0
PLAYER_ONE = 1  # beginnt immer, entspricht "Rot"
PLAYER_TWO = 2  # "Gelb"

PLAYER_NAMES = {PLAYER_ONE: "Rot", PLAYER_TWO: "Gelb"}
PLAYER_COLOURS = {PLAYER_ONE: "#d62728", PLAYER_TWO: "#f2c744"}
EMPTY_COLOUR = "#e5e5e5"

# Training: Selbstspiel auf 4x3 (dieselbe Größe wie evaluation-function-demo,
# direkte Vergleichbarkeit), exakte Zielwerte vom transposition-table-demo-
# artigen Alpha-Beta+Tabelle-Löser. EPSILON so hoch, weil rein "vernünftiges"
# Spiel (wenig Zufall) auf diesem kleinen Brett fast immer zum theoretischen
# Remis zurückfindet - siehe README, echter Fund beim Bau.
TRAIN_ROWS, TRAIN_COLS = 4, 3
GAMES_PER_GENERATION = 1000
N_GENERATIONS = 6
SEARCH_DEPTH = 3
SELFPLAY_EPSILON = 0.7
TRAIN_SEED = 1

# Aus evaluation-function-demo: Handgewichtung und der dort per Zufallspartien
# gefittete Vektor - feste Referenzen für den Vergleich (nicht neu berechnet,
# damit der Vergleich stabil bleibt).
HAND_WEIGHTS = [1.0, 8.0, 40.0]
RANDOM_FIT_WEIGHTS = [0.0, 0.0, 0.2882882882882883]

# Gemessen (Zugübereinstimmung mit dem exakten Optimalzug, 500 Testpositionen
# auf dem im Training ungesehenen 3×4-Brett, tests/test_claims.py). EHRLICHER
# BEFUND: Selbstspiel- und Zufallspartien-Training landen auf demselben Wert -
# siehe README für die Erklärung (beide entdecken dieselbe Ein-Merkmal-Lösung,
# die Datenquelle ändert bei nur 3 verfügbaren Merkmalen nichts mehr daran).
UNTRAINED_AGREEMENT = 0.940
RANDOM_FIT_AGREEMENT = 0.976
SELFPLAY_AGREEMENT = 0.976
HAND_AGREEMENT = 0.996

BOARD_OPTIONS = [
    {"rows": 4, "cols": 3, "label": "4 Zeilen × 3 Spalten (Trainingsbrett)"},
    {"rows": 3, "cols": 4, "label": "3 Zeilen × 4 Spalten (ungesehen im Training)"},
    {"rows": 4, "cols": 4, "label": "4 × 4 (ungesehen, größer)"},
]
DEFAULT_BOARD_INDEX = 0
