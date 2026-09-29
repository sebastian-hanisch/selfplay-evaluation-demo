"""Merkmalsextraktion - wortgleiche Kopie der Tests aus evaluation-function-demo."""

from sp_constants import PLAYER_ONE, PLAYER_TWO
from sp_game import apply_move, empty_board
from sp_features import feature_vector


def test_empty_board_has_zero_features():
    board = empty_board(4, 4)
    assert feature_vector(board) == [0, 0, 0]


def test_feature_vector_is_antisymmetric_under_colour_swap():
    board_rot = empty_board(4, 3)
    apply_move(board_rot, 0, PLAYER_ONE)
    apply_move(board_rot, 0, PLAYER_ONE)

    board_gelb = empty_board(4, 3)
    apply_move(board_gelb, 0, PLAYER_TWO)
    apply_move(board_gelb, 0, PLAYER_TWO)

    x_rot = feature_vector(board_rot)
    x_gelb = feature_vector(board_gelb)
    assert x_rot == [-v for v in x_gelb]


def test_three_in_open_window_is_stufe_3():
    board = empty_board(1, 4)
    apply_move(board, 0, PLAYER_ONE)
    apply_move(board, 1, PLAYER_ONE)
    apply_move(board, 2, PLAYER_ONE)
    assert feature_vector(board) == [0, 0, 1]
