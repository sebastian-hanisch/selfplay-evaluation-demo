"""Regressionstest gegen den in minimax-demo gefundenen scaleanchor+range-Bug."""

from sp_game import empty_board
from sp_visualization import agreement_comparison_figure, board_figure, weight_evolution_figure


def test_board_figure_uses_autorange_not_explicit_range():
    board = empty_board(4, 3)
    fig = board_figure(board, None, None, 1)
    assert fig.layout.xaxis.autorange is True
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.autorange is True
    assert fig.layout.yaxis.range is None


def test_weight_evolution_has_three_lines():
    history = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.22]]
    fig = weight_evolution_figure(history)
    assert len(fig.data) == 3
    assert list(fig.data[2].y) == [0.0, 0.2, 0.22]


def test_agreement_comparison_supports_four_bars():
    fig = agreement_comparison_figure(["a", "b", "c", "d"], [0.9, 0.95, 0.95, 0.99])
    assert len(fig.data[0].x) == 4
    assert len(fig.data[0].marker.color) == 4
