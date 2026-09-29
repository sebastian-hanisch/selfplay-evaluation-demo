"""Merkmale für die Bewertungsfunktion: offene Linien je Bedrohungsstufe.

Wortgleiche Kopie aus evaluation-function-demo - dasselbe Merkmalsdesign,
diesmal mit SELBSTSPIEL-trainierten statt zufallspartien-trainierten
Gewichten (siehe sp_selfplay.py)."""

from __future__ import annotations

from sp_constants import PLAYER_ONE, PLAYER_TWO
from sp_game import all_windows

N_FEATURES = 3

_WINDOW_CACHE: dict[tuple[int, int], list[list[tuple[int, int]]]] = {}


def _windows_for(rows: int, cols: int) -> list[list[tuple[int, int]]]:
    key = (rows, cols)
    if key not in _WINDOW_CACHE:
        _WINDOW_CACHE[key] = all_windows(rows, cols)
    return _WINDOW_CACHE[key]


def feature_vector(board: list[list[int]]) -> list[int]:
    rows, cols = len(board), len(board[0])
    rot_counts = [0, 0, 0, 0]
    gelb_counts = [0, 0, 0, 0]
    for window in _windows_for(rows, cols):
        values = [board[r][c] for r, c in window]
        has_rot = PLAYER_ONE in values
        has_gelb = PLAYER_TWO in values
        if has_rot and not has_gelb:
            rot_counts[values.count(PLAYER_ONE)] += 1
        elif has_gelb and not has_rot:
            gelb_counts[values.count(PLAYER_TWO)] += 1
    return [rot_counts[k] - gelb_counts[k] for k in (1, 2, 3)]
