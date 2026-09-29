"""PDF-Export der aktuellen Positions-Analyse (fpdf2).

`multi_cell(w=0, ...)` lässt den Cursor per Default am RECHTEN statt am linken
Rand stehen - ohne `new_x=LMARGIN` würde ein zweiter `multi_cell`-Aufruf direkt
danach abstürzen (bekannter Bug aus minimax-demo). Keine Sonderzeichen wie
„…" in PDF-gebundenen Strings (bekannter Bug aus alpha-beta-demo).
"""

from __future__ import annotations

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from sp_constants import PLAYER_NAMES
from sp_evaluation import format_de_number
from sp_presets import EVAL_LABELS

_NEXT_LINE = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_pdf(rows: int, cols: int, moves: list[int], eval_choice: str, player_to_move: int, value: float, node_count: int, best_column: int) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Selbstspiel-Bewertung - Analyse der aktuellen Stellung", **_NEXT_LINE)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Brett: {rows} Zeilen x {cols} Spalten", **_NEXT_LINE)
    move_text = ", ".join(str(m) for m in moves) if moves else "keine (Startstellung)"
    pdf.cell(0, 8, f"Bisherige Züge (Spalten): {move_text}", **_NEXT_LINE)
    label = EVAL_LABELS[eval_choice].split(" - ")[0]
    pdf.cell(0, 8, f"Bewertungsfunktion: {label}", **_NEXT_LINE)
    pdf.cell(0, 8, f"Am Zug: {PLAYER_NAMES[player_to_move]}", **_NEXT_LINE)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Ergebnis der Suche", **_NEXT_LINE)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Geschätzter Wert: {format_de_number(value, 2)}", **_NEXT_LINE)
    pdf.cell(0, 8, f"Durchsuchte Knoten: {format_de_number(node_count)}", **_NEXT_LINE)
    pdf.cell(0, 8, f"Gewählte Spalte: {best_column}", **_NEXT_LINE)

    return bytes(pdf.output())
