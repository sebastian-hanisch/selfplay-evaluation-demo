"""Zobrist-Hashing für die Transpositionstabelle - wortgleiche Kopie aus
transposition-table-demo."""

from __future__ import annotations

import random

from sp_game import Move

MAX_ROWS = 8
MAX_COLS = 8
_SEED = 20260929

_rng = random.Random(_SEED)
ZOBRIST = [[{1: _rng.getrandbits(64), 2: _rng.getrandbits(64)} for _ in range(MAX_COLS)] for _ in range(MAX_ROWS)]


def toggle(hash_value: int, move: Move, player: int) -> int:
    return hash_value ^ ZOBRIST[move.row][move.column][player]


def board_hash(board: list[list[int]]) -> int:
    h = 0
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value:
                h ^= ZOBRIST[r][c][value]
    return h
