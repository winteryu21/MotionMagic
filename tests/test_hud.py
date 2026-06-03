"""HUD layout tests."""

from __future__ import annotations

from src.game.settings import SCREEN_WIDTH
from src.game.ui.hud import (
    SKILL_PANEL_COL_GAP,
    SKILL_PANEL_COMBO_GAP,
    SKILL_PANEL_RIGHT_MARGIN,
    SKILL_PANEL_ROW_W,
    SKILL_PANEL_X,
    _skill_panel_combo_area_width,
    _skill_panel_layout,
)


def test_skill_panel_wraps_unlocked_spells_within_screen_width() -> None:
    """Unlocked spell list should not overflow past the right safe margin."""
    spell_count = 7
    max_combo_len = 3

    columns, _rows_per_col, col_w, combo_area_w = _skill_panel_layout(
        spell_count,
        max_combo_len,
    )

    right_edge = (
        SKILL_PANEL_X
        + (columns - 1) * col_w
        + SKILL_PANEL_ROW_W
        + SKILL_PANEL_COMBO_GAP
        + combo_area_w
    )
    assert columns == 2
    assert right_edge <= SCREEN_WIDTH - SKILL_PANEL_RIGHT_MARGIN


def test_skill_panel_combo_area_matches_gesture_steps() -> None:
    """Combo area should grow with combo length without adding trailing gap."""
    single_width = _skill_panel_combo_area_width(1)
    triple_width = _skill_panel_combo_area_width(3)

    assert triple_width > single_width
    assert triple_width < single_width * 3 + SKILL_PANEL_COL_GAP
