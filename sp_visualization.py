"""Plotly-Figuren: Brett, Gewichts-Verlauf über Generationen, Vergleichsbalken
(autorange statt expliziter range, siehe minimax-demo: scaleanchor+range
friert den Bereich anhand der Containerbreite ein)."""

from __future__ import annotations

import plotly.graph_objects as go

from sp_constants import EMPTY, EMPTY_COLOUR, PLAYER_COLOURS
from sp_game import winning_line


def board_figure(board, last_move, winner: int | None, best_column: int | None = None) -> go.Figure:
    rows, cols = len(board), len(board[0])
    fig = go.Figure()

    win_cells = set()
    if winner is not None and last_move is not None:
        line = winning_line(board, last_move, winner)
        if line:
            win_cells = set(line)

    xs, ys, colours, line_widths, line_colours = [], [], [], [], []
    for r in range(rows):
        for c in range(cols):
            value = board[r][c]
            xs.append(c)
            ys.append(rows - 1 - r)
            colours.append(EMPTY_COLOUR if value == EMPTY else PLAYER_COLOURS[value])
            if (r, c) in win_cells:
                line_widths.append(4)
                line_colours.append("#1a1a1a")
            else:
                line_widths.append(1)
                line_colours.append("#9a9a9a")

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=46, color=colours, line=dict(width=line_widths, color=line_colours)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if best_column is not None:
        fig.add_trace(
            go.Scatter(
                x=[best_column],
                y=[rows + 0.35],
                mode="markers",
                marker=dict(size=16, color="#2ca02c", symbol="triangle-down"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[-0.7, cols - 0.3],
            y=[-0.7, rows + 0.9],
            mode="markers",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(autorange=True, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True)
    fig.update_yaxes(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(height=110 * rows + 140, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#f7f7f7")
    return fig


def weight_evolution_figure(history: list[list[float]]) -> go.Figure:
    generations = list(range(len(history)))
    labels = ["Stufe 1", "Stufe 2", "Stufe 3"]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig = go.Figure()
    for i, (label, colour) in enumerate(zip(labels, colours)):
        fig.add_trace(go.Scatter(x=generations, y=[w[i] for w in history], mode="lines+markers", name=label, marker_color=colour))
    fig.update_xaxes(title="Generation", dtick=1, fixedrange=True)
    fig.update_yaxes(title="Gewicht", fixedrange=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def agreement_comparison_figure(labels: list[str], values: list[float]) -> go.Figure:
    palette = ["#9a9a9a", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colours = [palette[i % len(palette)] for i in range(len(labels))]
    fig = go.Figure(go.Bar(x=labels, y=[v * 100 for v in values], marker_color=colours, text=[f"{v:.1%}" for v in values], textposition="outside"))
    fig.update_yaxes(title="Zugübereinstimmung mit exaktem Optimalzug (%)", range=[90, 101], fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    return fig
