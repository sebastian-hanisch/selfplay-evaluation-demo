"""Kennzahlen, Gewichts-Verlauf (gecacht) und Formatierung."""

from __future__ import annotations

import streamlit as st

from sp_constants import GAMES_PER_GENERATION, N_GENERATIONS, SEARCH_DEPTH, SELFPLAY_EPSILON, TRAIN_COLS, TRAIN_ROWS, TRAIN_SEED
from sp_selfplay import train_generations


@st.cache_data(show_spinner="Trainiere per Selbstspiel (mehrere Generationen) ...")
def weight_history() -> list[list[float]]:
    return train_generations(TRAIN_ROWS, TRAIN_COLS, N_GENERATIONS, GAMES_PER_GENERATION, SEARCH_DEPTH, TRAIN_SEED, SELFPLAY_EPSILON)


def format_de_number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", ".")
