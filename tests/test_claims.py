"""Jede Zahl aus dem README/der App wird hier gegen den tatsächlichen Code
nachgerechnet."""

from __future__ import annotations

import random

import pytest

from sp_constants import (
    GAMES_PER_GENERATION,
    HAND_AGREEMENT,
    HAND_WEIGHTS,
    N_GENERATIONS,
    PLAYER_ONE,
    RANDOM_FIT_AGREEMENT,
    RANDOM_FIT_WEIGHTS,
    SEARCH_DEPTH,
    SELFPLAY_AGREEMENT,
    SELFPLAY_EPSILON,
    TRAIN_COLS,
    TRAIN_ROWS,
    TRAIN_SEED,
    UNTRAINED_AGREEMENT,
)
from sp_exact import exact_value
from sp_game import apply_move, check_win_at, empty_board, is_full, legal_columns, other_player, undo_move
from sp_search import search
from sp_selfplay import train_generations


def test_readme_weight_history():
    history = train_generations(TRAIN_ROWS, TRAIN_COLS, N_GENERATIONS, GAMES_PER_GENERATION, SEARCH_DEPTH, TRAIN_SEED, SELFPLAY_EPSILON)
    assert len(history) == N_GENERATIONS + 1
    assert history[0] == [0.0, 0.0, 0.0]
    expected = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.23214285714285707],
        [0.0, 0.0, 0.1837837837837838],
        [0.0, 0.0, 0.22457627118644072],
        [0.0, 0.0, 0.19512195121951217],
        [0.0, 0.0, 0.21524663677130051],
        [0.0, 0.0, 0.22222222222222224],
    ]
    for got, want in zip(history, expected):
        assert got == pytest.approx(want)


def _generate_random_dataset(rows, cols, n_games, seed):
    """Reine Zufallspartien (KEIN Selbstspiel) - dieselbe Erzeugungsmethode
    wie evaluation-function-demo's ev_data.generate_dataset, hier nur für den
    unabhängigen Out-of-distribution-Testsplit gebraucht."""
    rng = random.Random(seed)
    seen: dict[tuple, tuple[tuple, int, int]] = {}
    for _ in range(n_games):
        board = empty_board(rows, cols)
        player = PLAYER_ONE
        while True:
            cols_left = legal_columns(board)
            if not cols_left:
                break
            key = (tuple(tuple(row) for row in board), player)
            if key not in seen:
                value = exact_value([row[:] for row in board], player)
                seen[key] = (key[0], player, value)
            move = apply_move(board, rng.choice(cols_left), player)
            if check_win_at(board, move, player):
                break
            if is_full(board):
                break
            player = other_player(player)
    return list(seen.values())


def _true_best_columns(board, player):
    values = {}
    for col in legal_columns(board):
        move = apply_move(board, col, player)
        if check_win_at(board, move, player):
            v = 1 if player == PLAYER_ONE else -1
        elif is_full(board):
            v = 0
        else:
            v = exact_value([row[:] for row in board], other_player(player))
        undo_move(board, move)
        values[col] = v
    best = max(values.values()) if player == PLAYER_ONE else min(values.values())
    return [c for c, v in values.items() if v == best]


def test_readme_agreement_out_of_distribution_3x4():
    # Out-of-distribution-Vergleich (3x4-Brett, im Training nie gesehen -
    # Training lief auf 4x3): untrainiert vs. zufallspartien-gefittet
    # (Stück 4) vs. selbstspiel-gefittet (dieses Stück) vs. handgewichtet.
    history = train_generations(TRAIN_ROWS, TRAIN_COLS, N_GENERATIONS, GAMES_PER_GENERATION, SEARCH_DEPTH, TRAIN_SEED, SELFPLAY_EPSILON)
    selfplay_weights = history[-1]

    test_data = _generate_random_dataset(3, 4, 3000, seed=99)
    random.Random(7).shuffle(test_data)

    weight_sets = {
        "untrained": ([0.0, 0.0, 0.0], UNTRAINED_AGREEMENT),
        "random_fit": (RANDOM_FIT_WEIGHTS, RANDOM_FIT_AGREEMENT),
        "selfplay": (selfplay_weights, SELFPLAY_AGREEMENT),
        "hand": (HAND_WEIGHTS, HAND_AGREEMENT),
    }
    agree = {name: 0 for name in weight_sets}
    total = 0
    for board_key, player, _value in test_data:
        board = [list(row) for row in board_key]
        if sum(row.count(0) for row in board) <= SEARCH_DEPTH:
            continue
        total += 1
        best_cols = _true_best_columns(board, player)
        for name, (weights, _expected) in weight_sets.items():
            r = search([row[:] for row in board], player, SEARCH_DEPTH, weights)
            agree[name] += r.best_column in best_cols
        if total >= 500:
            break

    assert total == 500
    for name, (_weights, expected) in weight_sets.items():
        assert agree[name] / total == pytest.approx(expected, abs=0.005)


def test_selfplay_and_random_fit_land_on_the_same_single_feature_solution():
    # Der Kernbefund hinter dem App-Text: beide Trainingsarten setzen bei nur
    # 3 verfuegbaren Merkmalen ausschliesslich auf Stufe 3.
    history = train_generations(TRAIN_ROWS, TRAIN_COLS, N_GENERATIONS, GAMES_PER_GENERATION, SEARCH_DEPTH, TRAIN_SEED, SELFPLAY_EPSILON)
    selfplay_weights = history[-1]
    assert selfplay_weights[0] == 0.0
    assert selfplay_weights[1] == 0.0
    assert RANDOM_FIT_WEIGHTS[0] == 0.0
    assert RANDOM_FIT_WEIGHTS[1] == 0.0
