"""Rauchtests der Streamlit-Oberfläche per AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import sp_constants as C
from sp_presets import EVAL_HAND, EVAL_RANDOM_FIT, EVAL_SELFPLAY

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None):
    at = AppTest.from_file(str(APP), default_timeout=90)
    if setup is not None:
        setup(at)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_renders_without_exception():
    at = _run()
    assert any("Stellung" in h.value for h in at.subheader)


def test_every_board_option_renders():
    for index in range(len(C.BOARD_OPTIONS)):

        def setup(at, index=index):
            at.session_state["board_index_select"] = index
            at.session_state["moves"] = []

        _run(setup)


def test_every_eval_choice_renders():
    for choice in (EVAL_HAND, EVAL_RANDOM_FIT, EVAL_SELFPLAY):

        def setup(at, choice=choice):
            at.session_state["eval_choice_select"] = choice

        _run(setup)


def test_clicking_a_column_button_plays_a_move():
    at = _run()
    drop_buttons = [b for b in at.button if b.key and b.key.startswith("drop_")]
    assert drop_buttons
    drop_buttons[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["moves"]) == 1


def test_switching_board_size_resets_moves():
    def setup(at):
        at.session_state["board_index_select"] = 0
        at.session_state["moves"] = [0]

    at = _run(setup)
    at.radio(key="board_index_select").set_value(1)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["moves"] == []


def test_permalink_restores_everything():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.query_params["board"] = "1"
    at.query_params["eval"] = EVAL_HAND
    at.query_params["moves"] = "0,1"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["board_index_select"] == 1
    assert at.session_state["eval_choice_select"] == EVAL_HAND
    assert at.session_state["moves"] == [0, 1]


def test_permalink_does_not_get_clobbered_by_later_rerun():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.query_params["board"] = "0"
    at.run()
    drop_buttons = [b for b in at.button if b.key and b.key.startswith("drop_")]
    drop_buttons[0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["moves"] == [0]
