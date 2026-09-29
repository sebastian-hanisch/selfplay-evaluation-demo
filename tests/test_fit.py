"""Gewichtsanpassung auf Selbstspiel-Daten: reproduzierbar (fester Seed) und
mit Domänenwissen konsistent (nicht-negativ)."""

from sp_constants import SEARCH_DEPTH, TRAIN_COLS, TRAIN_ROWS
from sp_fit import fit_weights
from sp_selfplay import generate_selfplay_dataset


def test_fit_is_deterministic_for_fixed_seed():
    data = generate_selfplay_dataset(TRAIN_ROWS, TRAIN_COLS, [0.0, 0.0, 0.0], SEARCH_DEPTH, n_games=50, seed=1, epsilon=0.7)
    w1 = fit_weights(data)
    w2 = fit_weights(data)
    assert w1 == w2


def test_fitted_weights_are_non_negative():
    data = generate_selfplay_dataset(TRAIN_ROWS, TRAIN_COLS, [0.0, 0.0, 0.0], SEARCH_DEPTH, n_games=50, seed=3, epsilon=0.7)
    w = fit_weights(data)
    assert all(wi >= 0 for wi in w)
