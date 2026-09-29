"""Selbstspiel-Trainingsschleife: statt Zufallspartien (evaluation-function-
demo) erzeugt JEDE Generation ihre Trainingsdaten durch Partien der
VORHERIGEN Generation gegen sich selbst (Alpha-Beta+Tabelle-Suche mit der
aktuellen Gewichtung, siehe sp_search.py) - die Zielwerte bleiben exakt
(sp_exact.py), nur die BESUCHTEN Stellungen kommen jetzt aus realistischerem
Spiel statt aus reinem Zufall.

Ohne Zufall wären alle Partien einer Generation identisch (dieselbe
Gewichtung sucht denselben besten Zug). Echter Fund beim Bau: ein paar
zufällige ERÖFFNUNGSzüge reichen nicht aus (die Suche lenkt danach fast immer
zurück zum theoretischen Remis dieses Bretts, siehe README) - erst
Epsilon-Greedy (bei JEDEM Zug mit Wahrscheinlichkeit `epsilon` ein
Zufallszug statt des Suchergebnisses) liefert genug entscheidende
(nicht-remis) Stellungen für ein sinnvolles Training.
"""

from __future__ import annotations

import random

from sp_constants import PLAYER_ONE
from sp_exact import exact_value
from sp_fit import fit_weights
from sp_game import apply_move, check_win_at, empty_board, is_full, legal_columns, other_player
from sp_search import search


def play_self_play_game(rows: int, cols: int, weights: list[float], max_depth: int, rng: random.Random, epsilon: float) -> list[int]:
    board = empty_board(rows, cols)
    player = PLAYER_ONE
    moves: list[int] = []
    while True:
        cols_left = legal_columns(board)
        if not cols_left:
            break
        if rng.random() < epsilon:
            col = rng.choice(cols_left)
        else:
            col = search(board, player, max_depth, weights, use_tt=True).best_column
        move = apply_move(board, col, player)
        moves.append(col)
        if check_win_at(board, move, player):
            break
        if is_full(board):
            break
        player = other_player(player)
    return moves


def generate_selfplay_dataset(rows: int, cols: int, weights: list[float], max_depth: int, n_games: int, seed: int, epsilon: float = 0.7) -> list[tuple[tuple, int, int]]:
    rng = random.Random(seed)
    seen: dict[tuple, tuple[tuple, int, int]] = {}
    for _ in range(n_games):
        game_moves = play_self_play_game(rows, cols, weights, max_depth, rng, epsilon)
        board = empty_board(rows, cols)
        player = PLAYER_ONE
        for col in game_moves:
            key = (tuple(tuple(row) for row in board), player)
            if key not in seen:
                value = exact_value([row[:] for row in board], player)
                seen[key] = (key[0], player, value)
            move = apply_move(board, col, player)
            if check_win_at(board, move, player) or is_full(board):
                break
            player = other_player(player)
    return list(seen.values())


def train_generations(rows: int, cols: int, n_generations: int, games_per_generation: int, max_depth: int, seed: int, epsilon: float = 0.7, init_weights: list[float] | None = None) -> list[list[float]]:
    """Gibt eine Liste von Gewichtsvektoren zurück, Index 0 = Startgewichte
    (vor jedem Training), Index i = Gewichte nach Generation i."""
    weights = list(init_weights) if init_weights is not None else [0.0, 0.0, 0.0]
    history = [list(weights)]
    for gen in range(n_generations):
        data = generate_selfplay_dataset(rows, cols, weights, max_depth, games_per_generation, seed=seed + gen, epsilon=epsilon)
        weights = fit_weights(data)
        history.append(list(weights))
    return history
