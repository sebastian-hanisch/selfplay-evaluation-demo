"""PRESETS, Permalink (Begrenzen/Einrasten), Session-Defaults."""

from __future__ import annotations

import streamlit as st

from sp_constants import BOARD_OPTIONS, DEFAULT_BOARD_INDEX

EVAL_HAND = "hand"
EVAL_RANDOM_FIT = "random_fit"
EVAL_SELFPLAY = "selfplay"
EVAL_CHOICES = [EVAL_HAND, EVAL_RANDOM_FIT, EVAL_SELFPLAY]
EVAL_LABELS = {
    EVAL_HAND: "Handgewichtet (1, 8, 40) - evaluation-function-demo",
    EVAL_RANDOM_FIT: "An Zufallspartien gefittet - evaluation-function-demo",
    EVAL_SELFPLAY: "An Selbstspiel-Partien gefittet (letzte Generation) - dieses Stück",
}

PRESETS = {
    "Selbstspiel-Gewichte (4×3)": {"board_index": 0, "eval_choice": EVAL_SELFPLAY, "moves": []},
    "Handgewichtet zum Vergleich (4×3)": {"board_index": 0, "eval_choice": EVAL_HAND, "moves": []},
    "Ungesehenes Brett (3×4)": {"board_index": 1, "eval_choice": EVAL_SELFPLAY, "moves": []},
}
PRESET_HELP = {
    "Selbstspiel-Gewichte (4×3)": "Die in diesem Stück per Selbstspiel trainierte Gewichtung.",
    "Handgewichtet zum Vergleich (4×3)": "Dieselbe Größe mit der Handgewichtung aus evaluation-function-demo.",
    "Ungesehenes Brett (3×4)": "Testet die Selbstspiel-Gewichte auf einer Brettform, die im Training nie vorkam.",
}

_DEFAULTS = {"board_index": DEFAULT_BOARD_INDEX, "eval_choice": EVAL_SELFPLAY, "moves": []}


def apply_preset(name: str) -> None:
    preset = PRESETS[name]
    st.session_state["board_index_select"] = preset["board_index"]
    st.session_state["eval_choice_select"] = preset["eval_choice"]
    st.session_state["moves"] = list(preset["moves"])


def init_session_state_defaults() -> None:
    if "board_index_select" not in st.session_state:
        st.session_state["board_index_select"] = _DEFAULTS["board_index"]
    if "eval_choice_select" not in st.session_state:
        st.session_state["eval_choice_select"] = _DEFAULTS["eval_choice"]
    if "moves" not in st.session_state:
        st.session_state["moves"] = list(_DEFAULTS["moves"])


def _parse_moves(raw: str) -> list[int]:
    if not raw:
        return []
    try:
        return [int(x) for x in raw.split(",") if x != ""]
    except ValueError:
        return []


def load_permalink_settings() -> None:
    """Lädt Einstellungen aus der URL - NUR beim allerersten Lauf dieser
    Session (sonst würde jeder Klick sofort wieder rückgängig gemacht - echter,
    bereits einmal gefundener Bug in minimax-demo, siehe dortige Moduldoku)."""
    if "board_index_select" in st.session_state:
        return
    params = st.query_params
    if not any(k in params for k in ("board", "moves", "eval")):
        return

    board_index = _DEFAULTS["board_index"]
    if "board" in params:
        try:
            candidate = int(params["board"])
        except ValueError:
            candidate = board_index
        if 0 <= candidate < len(BOARD_OPTIONS):
            board_index = candidate

    eval_choice = params.get("eval", _DEFAULTS["eval_choice"])
    if eval_choice not in EVAL_CHOICES:
        eval_choice = _DEFAULTS["eval_choice"]

    moves = _parse_moves(params.get("moves", ""))

    st.session_state["board_index_select"] = board_index
    st.session_state["eval_choice_select"] = eval_choice
    st.session_state["moves"] = moves


def sync_query_params(board_index: int, eval_choice: str, moves: list[int]) -> None:
    st.query_params["board"] = str(board_index)
    st.query_params["eval"] = eval_choice
    st.query_params["moves"] = ",".join(str(m) for m in moves)
