"""Kreuzprobe: der exakte Löser muss dieselben Werte liefern wie in den
Vorgängerstücken gemessen."""

import pytest

from sp_constants import PLAYER_ONE
from sp_exact import exact_value
from sp_game import empty_board


@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 3), (3, 4), (4, 4)])
def test_value_matches_previous_pieces(rows, cols):
    board = empty_board(rows, cols)
    assert exact_value(board, PLAYER_ONE) == 0
