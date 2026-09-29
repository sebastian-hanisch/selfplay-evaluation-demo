"""Gewichte per nicht-negativen kleinsten Quadraten an bekannte exakte
Spielwerte anpassen - wortgleiche Methode wie in evaluation-function-demo,
hier auf SELBSTSPIEL-Daten statt Zufallspartien angewendet (siehe
sp_selfplay.py)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from sp_features import feature_vector


def fit_weights(dataset: list[tuple[tuple, int, int]]) -> list[float]:
    X = []
    y = []
    for board_key, _player, value in dataset:
        board = [list(row) for row in board_key]
        X.append(feature_vector(board))
        y.append(value)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights, _residual = nnls(X, y)
    return weights.tolist()
