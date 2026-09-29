"""Selbstspiel-Erzeugung + Trainingsschleife."""

import random

from sp_exact import exact_value
from sp_game import apply_move, check_win_at, empty_board, is_full
from sp_selfplay import generate_selfplay_dataset, play_self_play_game, train_generations


def test_self_play_game_produces_a_legal_terminal_game():
    rng = random.Random(1)
    moves = play_self_play_game(4, 3, [0.0, 0.0, 0.0], max_depth=2, rng=rng, epsilon=0.5)
    assert 1 <= len(moves) <= 12
    board = empty_board(4, 3)
    player = 1
    for col in moves:
        move = apply_move(board, col, player)
        if check_win_at(board, move, player) or is_full(board):
            break
        player = 3 - player


def test_epsilon_zero_is_deterministic_given_same_weights():
    rng1 = random.Random(5)
    rng2 = random.Random(5)
    m1 = play_self_play_game(3, 3, [1.0, 5.0, 20.0], max_depth=3, rng=rng1, epsilon=0.0)
    m2 = play_self_play_game(3, 3, [1.0, 5.0, 20.0], max_depth=3, rng=rng2, epsilon=0.0)
    assert m1 == m2


def test_generate_selfplay_dataset_has_ground_truth_labels():
    data = generate_selfplay_dataset(4, 3, [0.0, 0.0, 0.0], max_depth=2, n_games=20, seed=1, epsilon=0.7)
    assert len(data) > 0
    for board_key, player, value in data:
        board = [list(row) for row in board_key]
        assert exact_value(board, player) == value


def test_train_generations_returns_one_more_entry_than_generations():
    history = train_generations(4, 3, n_generations=2, games_per_generation=50, max_depth=2, seed=1, epsilon=0.7)
    assert len(history) == 3
    assert history[0] == [0.0, 0.0, 0.0]


def test_train_generations_weights_are_non_negative():
    history = train_generations(4, 3, n_generations=2, games_per_generation=50, max_depth=2, seed=1, epsilon=0.7)
    for w in history:
        assert all(wi >= 0 for wi in w)
