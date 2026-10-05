"""Unabhängiges Orakel für exakten Löser, Suche mit Transpositionstabelle, Gewichtsfit und
die Selbstspiel-Trainingsschleife.

Anderer Rechenweg als die Demo-Module: Stellung als Spalten-Tupel, Siegprüfung und Merkmale
durch Aufzählen aller Viererlinien, exakte Werte per memoisiertem Minimax, tiefenbegrenzte Suche
als reines Minimax (ohne Alpha-Beta und ohne Tabelle), nicht-negative kleinste Quadrate durch
Enumeration aller Stützmengen (statt `scipy.optimize.nnls`). Die Trainingsschleife wird mit diesen
Bausteinen unabhängig nachgebaut und muss dieselbe Gewichts-Historie liefern.
"""

import itertools
import random
from functools import lru_cache

import numpy as np
import pytest

from sp_exact import exact_value
from sp_game import apply_move, check_win_at, empty_board, legal_columns
from sp_search import search
from sp_selfplay import train_generations


class _Oracle:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.lines = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(0 <= a < rows and 0 <= b < cols for a, b in cells):
                        self.lines.append(cells)
        self.value = lru_cache(maxsize=None)(self._value)

    def cell(self, st, r, c):
        h = self.rows - 1 - r
        return st[c][h] if h < len(st[c]) else 0

    def won(self, st, p):
        return any(all(self.cell(st, r, c) == p for r, c in line) for line in self.lines)

    def moves(self, st):
        return [c for c in range(self.cols) if len(st[c]) < self.rows]

    def play(self, st, c, p):
        return st[:c] + (st[c] + (p,),) + st[c + 1 :]

    def _value(self, st, p):
        vals = []
        for c in self.moves(st):
            ns = self.play(st, c, p)
            if self.won(ns, p):
                vals.append(1 if p == 1 else -1)
            elif not self.moves(ns):
                vals.append(0)
            else:
                vals.append(self.value(ns, 3 - p))
        return max(vals) if p == 1 else min(vals)

    def features(self, st):
        f = [0, 0, 0]
        for line in self.lines:
            vals = [self.cell(st, r, c) for r, c in line]
            owners = set(vals) - {0}
            if len(owners) == 1:
                f[vals.count(next(iter(owners))) - 1] += 1 if 1 in owners else -1
        return f

    def child_values(self, st, p, depth, w):
        out = {}
        for c in self.moves(st):
            ns = self.play(st, c, p)
            if self.won(ns, p):
                out[c] = 1e6 if p == 1 else -1e6
            elif not self.moves(ns):
                out[c] = 0.0
            else:
                out[c] = self.plain(ns, 3 - p, depth - 1, w)
        return out

    def plain(self, st, p, depth, w):
        if depth == 0:
            return sum(a * b for a, b in zip(w, self.features(st)))
        vals = list(self.child_values(st, p, depth, w).values())
        return max(vals) if p == 1 else min(vals)

    def best_move(self, st, p, depth, w):
        vals = self.child_values(st, p, depth, w)
        best = max(vals.values()) if p == 1 else min(vals.values())
        return best, [c for c in self.moves(st) if vals[c] == best][0]


def _to_state(board):
    rows, cols = len(board), len(board[0])
    out = []
    for c in range(cols):
        col = []
        for r in range(rows - 1, -1, -1):
            if board[r][c] == 0:
                break
            col.append(board[r][c])
        out.append(tuple(col))
    return tuple(out)


def _random_position(rng, rows, cols, max_free, min_free=0):
    board, p = empty_board(rows, cols), 1
    for _ in range(rng.randint(max(0, rows * cols - max_free), rows * cols - min_free)):
        free = legal_columns(board)
        if not free:
            return None
        mv = apply_move(board, rng.choice(free), p)
        if check_win_at(board, mv, p) or not legal_columns(board):
            return None
        p = 3 - p
    return (board, p) if legal_columns(board) else None


def test_exact_value_matches_oracle():
    rng = random.Random(1)
    done = 0
    while done < 120:
        rows, cols = rng.randint(1, 5), rng.randint(1, 5)
        pos = _random_position(rng, rows, cols, 9)
        if pos is None:
            continue
        board, p = pos
        assert exact_value([r[:] for r in board], p) == _Oracle(rows, cols).value(_to_state(board), p)
        done += 1


@pytest.mark.parametrize("use_tt", [True, False])
def test_search_with_and_without_table_equals_plain_minimax(use_tt):
    rng = random.Random(2)
    weight_sets = [[1.0, 8.0, 40.0], [0.0, 0.0, 0.2882882882882883], [0.3, 1.7, 2.2], [0.0, 0.0, 0.0]]
    done = 0
    while done < 60:
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        pos = _random_position(rng, rows, cols, rows * cols, min_free=1)
        if pos is None:
            continue
        board, p = pos
        depth, w = rng.randint(1, 4), rng.choice(weight_sets)
        best, first = _Oracle(rows, cols).best_move(_to_state(board), p, depth, w)
        res = search([r[:] for r in board], p, depth, w, use_tt=use_tt)
        assert res.value == pytest.approx(best)
        assert res.best_column == first
        done += 1


def _oracle_nnls(X, y):
    best = None
    for k in range(4):
        for support in itertools.combinations(range(3), k):
            w = np.zeros(3)
            if support:
                sol = np.linalg.lstsq(X[:, list(support)], y, rcond=None)[0]
                if (sol < 0).any():
                    continue
                w[list(support)] = sol
            rss = float(((X @ w - y) ** 2).sum())
            if best is None or rss < best[0] - 1e-12:
                best = (rss, w)
    return list(best[1])


def _oracle_training(rows, cols, n_generations, games, depth, seed, epsilon):
    o = _Oracle(rows, cols)
    w = [0.0, 0.0, 0.0]
    history = [list(w)]
    for g in range(n_generations):
        rng = random.Random(seed + g)
        seen = {}
        for _ in range(games):
            st, p = tuple(() for _ in range(cols)), 1
            while o.moves(st):
                c = rng.choice(o.moves(st)) if rng.random() < epsilon else o.best_move(st, p, depth, w)[1]
                if (st, p) not in seen:
                    seen[(st, p)] = o.value(st, p)
                st = o.play(st, c, p)
                if o.won(st, p) or not o.moves(st):
                    break
                p = 3 - p
        X = np.array([o.features(s) for s, _p in seen], dtype=float)
        y = np.array(list(seen.values()), dtype=float)
        w = _oracle_nnls(X, y)
        history.append(list(w))
    return history


def test_training_loop_matches_independent_reimplementation():
    got = train_generations(4, 3, 3, 60, 3, 1, 0.7)
    want = _oracle_training(4, 3, 3, 60, 3, 1, 0.7)
    assert len(got) == len(want) == 4
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=1e-9)
