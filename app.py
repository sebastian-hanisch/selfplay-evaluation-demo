"""Selbstspiel mit gelernter Bewertungsfunktion: Konvergenz-Knoten der Linie -
nutzt die Alpha-Beta+Tabelle-Suche aus Ast A (alpha-beta-demo,
transposition-table-demo) UND ersetzt die Bewertungsfunktion aus Ast B
(evaluation-function-demo) durch eine per Selbstspiel trainierte.

Kind von transposition-table-demo UND evaluation-function-demo.
"""

from __future__ import annotations

import streamlit as st

import sp_constants as C
from sp_evaluation import format_de_number, weight_history
from sp_exact import exact_value
from sp_game import apply_move, check_win_at, is_full, legal_columns, other_player, replay, undo_move
from sp_pdf_export import build_pdf
from sp_presets import (
    EVAL_HAND,
    EVAL_LABELS,
    EVAL_RANDOM_FIT,
    EVAL_SELFPLAY,
    PRESET_HELP,
    PRESETS,
    apply_preset,
    init_session_state_defaults,
    load_permalink_settings,
    sync_query_params,
)
from sp_search import search
from sp_visualization import agreement_comparison_figure, board_figure, weight_evolution_figure

_de = format_de_number

st.set_page_config(page_title="Selbstspiel – Sebastian Hanisch", layout="wide")

st.title("🔁 Selbstspiel: die Bewertungsfunktion lernt aus eigenen Partien")
st.markdown(
    """
    **Konvergenz der Linie:** Alpha-Beta-Suche + Transpositionstabelle aus dem ersten Ast (alpha-beta-demo,
    transposition-table-demo) UND eine Bewertungsfunktion aus dem zweiten Ast (evaluation-function-demo) -
    aber diesmal per **Selbstspiel** trainiert (das AlphaZero-Prinzip: die Baumsuche selbst bleibt
    Alpha-Beta statt MCTS, siehe `mcts-demo` für Letzteres). Statt Zufallspartien erzeugt jede Generation
    ihre Trainingsdaten durch Partien der VORHERIGEN Generation gegen sich selbst. Am Ende der Seite: die
    📐 Mathematische Formulierung.
    """
)

st.caption("🎯 Schnellstart")
preset_cols = st.columns(len(PRESETS))
for col, name in zip(preset_cols, PRESETS):
    col.button(name, use_container_width=True, on_click=apply_preset, args=(name,), help=PRESET_HELP[name])
st.caption("🔗 Die URL merkt sich Brettgröße, Bewertung und Zugfolge (Permalink).")

load_permalink_settings()
init_session_state_defaults()


def _reset_moves() -> None:
    st.session_state["moves"] = []


with st.sidebar:
    st.header("⚙️ Einstellungen")
    eval_choice = st.radio(
        "Bewertungsfunktion",
        options=[EVAL_SELFPLAY, EVAL_HAND, EVAL_RANDOM_FIT],
        format_func=lambda o: EVAL_LABELS[o],
        key="eval_choice_select",
        on_change=_reset_moves,
    )
    board_index = st.radio(
        "Brettgröße",
        options=range(len(C.BOARD_OPTIONS)),
        format_func=lambda i: C.BOARD_OPTIONS[i]["label"],
        key="board_index_select",
        on_change=_reset_moves,
    )
    if st.button("↺ Neues Spiel", use_container_width=True):
        st.session_state["moves"] = []
        st.rerun()

board_spec = C.BOARD_OPTIONS[board_index]
rows, cols = board_spec["rows"], board_spec["cols"]
moves: list[int] = [m for m in st.session_state["moves"] if 0 <= m < cols]
sync_query_params(board_index, eval_choice, moves)

_history = weight_history()


def _weights_for(choice: str) -> list[float]:
    if choice == EVAL_HAND:
        return C.HAND_WEIGHTS
    if choice == EVAL_SELFPLAY:
        return _history[-1]
    # EVAL_RANDOM_FIT: aus evaluation-function-demo (dort trainiert, hier als
    # feste Referenz - NICHT neu berechnet, damit der Vergleich stabil bleibt)
    return C.RANDOM_FIT_WEIGHTS


weights = _weights_for(eval_choice)


@st.cache_data(show_spinner="Suche ...")
def _replay_and_search(rows: int, cols: int, moves: tuple[int, ...], weights: tuple[float, ...]):
    state = replay(rows, cols, list(moves))
    if state.is_terminal:
        return state, None
    result = search([row[:] for row in state.board], state.player_to_move, C.SEARCH_DEPTH, list(weights))
    return state, result


state, result = _replay_and_search(rows, cols, tuple(moves), tuple(weights))

st.subheader("Stellung")
board_col, info_col = st.columns([2, 1])

with board_col:
    best_col = result.best_column if result is not None else None
    fig = board_figure(state.board, state.last_move, state.winner, best_col)
    st.plotly_chart(fig, use_container_width=True, key="board_chart")

    if not state.is_terminal:
        click_cols = st.columns(cols)
        for c, click_col in enumerate(click_cols):
            full_column = state.board[0][c] != C.EMPTY
            label = f"⬇{c}" + (" ★" if c == best_col else "")
            if click_col.button(label, key=f"drop_{c}", disabled=full_column, use_container_width=True):
                st.session_state["moves"] = moves + [c]
                st.rerun()

with info_col:
    if state.is_terminal:
        if state.winner is not None:
            st.success(f"Spiel beendet: {C.PLAYER_NAMES[state.winner]} hat gewonnen.")
        else:
            st.info("Spiel beendet: Remis (Brett voll).")
    else:
        st.metric("Am Zug", C.PLAYER_NAMES[state.player_to_move])
        st.metric("Geschätzter Wert", f"{result.value:.2f}" if abs(result.value) < 1000 else ("Sieg" if result.value > 0 else "Niederlage"))
        st.metric("Durchsuchte Knoten", _de(result.node_count))
        st.caption("★ = von der Suche gewählte Spalte (Schätzung, keine Garantie).")

        pdf_bytes = build_pdf(rows, cols, moves, eval_choice, state.player_to_move, result.value, result.node_count, best_col)
        st.download_button("📄 Analyse als PDF", data=pdf_bytes, file_name="selfplay_analyse.pdf", mime="application/pdf")

st.markdown("---")
st.subheader("🔬 Wie entwickeln sich die Gewichte über die Generationen?")
st.markdown(
    f"""
    Jede Generation spielt {_de(C.GAMES_PER_GENERATION)} Partien gegen sich selbst (mit {C.SELFPLAY_EPSILON:.0%}
    Zufallszug-Anteil für Vielfalt - reines "vernünftiges" Spiel führt auf diesem kleinen Brett fast immer
    zurück zum theoretischen Remis, siehe „Wo die Annahmen enden"), sammelt die besuchten Stellungen mit
    ihrem EXAKTEN Wert und passt die Gewichte neu an.
    """
)
st.plotly_chart(weight_evolution_figure(_history), use_container_width=True, key="weight_chart")

st.subheader("🔬 Schlägt Selbstspiel die Handgewichtung?")
labels = ["Untrainiert\n(nur Suche)", "Zufallspartien-Fit\n(Stück 4)", "Selbstspiel-Fit\n(dieses Stück)", "Handgewichtet\n(Stück 4)"]
values = [C.UNTRAINED_AGREEMENT, C.RANDOM_FIT_AGREEMENT, C.SELFPLAY_AGREEMENT, C.HAND_AGREEMENT]
st.plotly_chart(agreement_comparison_figure(labels, values), use_container_width=True, key="agreement_chart")
st.warning(
    f"**Ehrlicher, unerwarteter Befund:** Training hilft deutlich gegenüber gar keinem Training "
    f"({C.UNTRAINED_AGREEMENT:.1%} ungeschult vs. {C.SELFPLAY_AGREEMENT:.1%} trainiert) - aber "
    f"Selbstspiel- und Zufallspartien-Training landen hier auf demselben Wert "
    f"({C.SELFPLAY_AGREEMENT:.1%} beide). Erwartet war, dass die realistischeren Selbstspiel-Stellungen "
    f"einen Vorsprung bringen. Der Grund: beide Trainingsarten entdecken dieselbe "
    f"Ein-Merkmal-Lösung (nur Stufe 3 zählt, siehe Diagramm oben und evaluation-function-demo) - bei "
    f"nur 3 verfügbaren Merkmalen ändert die Datenquelle nichts mehr an der Zugrangfolge, sobald diese "
    f"Struktur einmal gefunden ist. Die Handgewichtung (die alle drei Stufen nutzt) bleibt weiterhin "
    f"vorn ({C.HAND_AGREEMENT:.1%})."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
    - **Kein echtes AlphaZero.** Die Baumsuche bleibt Alpha-Beta (nicht MCTS), die Bewertungsfunktion bleibt
      linear (nicht neuronal) - bewusst klein und nachvollziehbar, nicht die volle Architektur.
    - **Reines "vernünftiges" Selbstspiel (wenig Zufall) bringt auf diesem kleinen Brett kaum
      Trainingsvielfalt** - fast jede Partie landet beim theoretischen Remis, sobald beide Seiten nicht
      grob patzen (echter Fund beim Bau: mit nur 2 zufälligen Eröffnungszügen bestand der gesamte
      Trainingsdatensatz aus 100 % Remis-Stellungen). Der hohe Zufallsanteil (70 %) ist deshalb bewusst so
      hoch gewählt, nicht Standard für Selbstspiel im Allgemeinen.
    - **Die Gewichte konvergieren schon nach einer Generation** - weitere Generationen ändern hier kaum
      noch etwas (siehe Diagramm oben), auf einem größeren Brett wäre vermutlich mehr Entwicklung zu sehen.
    """
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
        Jede Generation $g$ erzeugt Trainingsdaten $D_g$ durch Selbstspiel mit den Gewichten $w_{g-1}$
        (Alpha-Beta-Suche der Tiefe $d$, $\epsilon$-greedy für Vielfalt), löst jede besuchte Stellung
        $s \in D_g$ exakt ($y_s$, Grundwahrheit aus dem Alpha-Beta+Tabelle-Löser) und passt die Gewichte neu an:

        $$
        w_g = \arg\min_{w \geq 0} \sum_{s \in D_g} \big(w^\top x(s) - y_s\big)^2
        $$

        mit $x(s)$ demselben Merkmalsvektor wie in `evaluation-function-demo`. Der Unterschied zu Stück 4
        ist NICHT die Zielfunktion (dieselbe kleinste-Quadrate-Anpassung), sondern WELCHE Stellungen
        $D_g$ enthält - aus dem eigenen, sich verbessernden Spiel statt aus reinem Zufall.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
