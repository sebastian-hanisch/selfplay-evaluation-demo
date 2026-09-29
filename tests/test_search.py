"""Tiefenbegrenzte Alpha-Beta+Tabelle-Suche: Grundverhalten + Konvergenz gegen
den exakten Wert bei voller Tiefe, MIT und OHNE Transpositionstabelle."""

from sp_constants import HAND_WEIGHTS, PLAYER_ONE, PLAYER_TWO
from sp_exact import exact_value
from sp_game import apply_move, empty_board
from sp_search import search


def test_search_finds_immediate_win():
    board = empty_board(4, 2)
    for col, player in [(0, PLAYER_ONE), (1, PLAYER_TWO)] * 3:
        apply_move(board, col, player)
    result = search(board, PLAYER_ONE, max_depth=1, weights=HAND_WEIGHTS)
    assert result.best_column == 0
    assert result.value >= 1e5


def test_search_at_full_depth_matches_exact_value():
    board = empty_board(3, 3)
    exact = exact_value([row[:] for row in board], PLAYER_ONE)
    result = search([row[:] for row in board], PLAYER_ONE, max_depth=9, weights=HAND_WEIGHTS)
    assert result.value == exact


def test_tt_and_no_tt_agree_on_value():
    board_a = empty_board(4, 3)
    board_b = empty_board(4, 3)
    with_tt = search(board_a, PLAYER_ONE, max_depth=4, weights=HAND_WEIGHTS, use_tt=True)
    without_tt = search(board_b, PLAYER_ONE, max_depth=4, weights=HAND_WEIGHTS, use_tt=False)
    assert with_tt.value == without_tt.value
    assert with_tt.best_column == without_tt.best_column


def test_tt_reduces_node_count():
    board_a = empty_board(4, 3)
    board_b = empty_board(4, 3)
    with_tt = search(board_a, PLAYER_ONE, max_depth=4, weights=HAND_WEIGHTS, use_tt=True)
    without_tt = search(board_b, PLAYER_ONE, max_depth=4, weights=HAND_WEIGHTS, use_tt=False)
    assert with_tt.node_count <= without_tt.node_count


def test_search_is_pure_no_board_mutation():
    board = empty_board(4, 3)
    before = [row[:] for row in board]
    search(board, PLAYER_ONE, max_depth=3, weights=HAND_WEIGHTS)
    assert board == before
