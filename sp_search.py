"""Konvergenz dieser Linie: tiefenbegrenzte Alpha-Beta-Suche MIT
Transpositionstabelle (Ast A, aus alpha-beta-demo/transposition-table-demo)
UND einer linearen Bewertungsfunktion am Tiefenlimit (Ast B, aus
evaluation-function-demo) - beide Techniken zusammen statt getrennt.
"""

from __future__ import annotations

from dataclasses import dataclass

from sp_constants import PLAYER_ONE
from sp_features import feature_vector
from sp_game import apply_move, check_win_at, is_full, legal_columns, other_player, undo_move
from sp_zobrist import board_hash, toggle

_NEG_INF, _POS_INF = -1e9, 1e9
EXACT, LOWER, UPPER = "exact", "lower", "upper"


@dataclass
class SearchResult:
    value: float
    best_column: int
    node_count: int


def evaluate(board, weights) -> float:
    x = feature_vector(board)
    return sum(w * xi for w, xi in zip(weights, x))


def _terminal_value(board, move, player) -> float | None:
    if check_win_at(board, move, player):
        return 1e6 if player == PLAYER_ONE else -1e6
    if is_full(board):
        return 0.0
    return None


def _recurse(board, player, depth, alpha, beta, weights, h, table, counter) -> float:
    counter[0] += 1
    if depth == 0:
        return evaluate(board, weights)

    original_alpha = alpha
    key = (h, player, depth)
    if table is not None and key in table:
        value, flag = table[key]
        if flag == EXACT:
            return value
        if flag == LOWER:
            alpha = max(alpha, value)
        elif flag == UPPER:
            beta = min(beta, value)
        if alpha >= beta:
            return value

    best = None
    for col in legal_columns(board):
        move = apply_move(board, col, player)
        h2 = toggle(h, move, player)
        terminal = _terminal_value(board, move, player)
        value = terminal if terminal is not None else _recurse(board, other_player(player), depth - 1, alpha, beta, weights, h2, table, counter)
        undo_move(board, move)
        if player == PLAYER_ONE:
            if best is None or value > best:
                best = value
            alpha = max(alpha, best)
        else:
            if best is None or value < best:
                best = value
            beta = min(beta, best)
        if alpha >= beta:
            break

    if table is not None:
        if best <= original_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        table[key] = (best, flag)
    return best


def search(board, player: int, max_depth: int, weights, use_tt: bool = True) -> SearchResult:
    counter = [0]
    table: dict | None = {} if use_tt else None
    h = board_hash(board)
    alpha, beta = _NEG_INF, _POS_INF
    best_value = None
    best_col = None
    for col in legal_columns(board):
        move = apply_move(board, col, player)
        h2 = toggle(h, move, player)
        terminal = _terminal_value(board, move, player)
        value = terminal if terminal is not None else _recurse(board, other_player(player), max_depth - 1, alpha, beta, weights, h2, table, counter)
        undo_move(board, move)
        if player == PLAYER_ONE:
            if best_value is None or value > best_value:
                best_value, best_col = value, col
            alpha = max(alpha, best_value)
        else:
            if best_value is None or value < best_value:
                best_value, best_col = value, col
            beta = min(beta, best_value)
        if alpha >= beta:
            break
    return SearchResult(value=best_value, best_column=best_col, node_count=counter[0])
