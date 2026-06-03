"""Tests for spell unlock progression."""

from __future__ import annotations

from src.game.scenes.unlock import UnlockScene
from src.game.settings import GESTURE_PAPER, GESTURE_ROCK
from src.game.systems.magic import MagicSystem


def test_unlock_scene_unlocks_all_locked_spells_in_order() -> None:
    """Unlock scene should include every initially locked spell."""
    magic = MagicSystem()
    scene = UnlockScene()

    assert scene.unlock_order == [
        "lightning",
        "explosion",
        "piercing_bullet",
        "meteor",
    ]

    for stage, spell_key in zip(range(2, 10, 2), scene.unlock_order):
        assert scene.should_open(stage, magic)
        spell = scene.next_unlock_spell(magic)
        assert spell is not None
        assert spell.key == spell_key
        magic.unlock_spell(spell)

    assert scene.next_unlock_spell(magic) is None
    assert not scene.should_open(10, magic)


def test_meteor_combo_can_be_recognized_after_unlock() -> None:
    """Meteor should remain castable after adding its three-part combo."""
    magic = MagicSystem()
    magic.unlock_spell_by_key("meteor")

    spell = magic.spell_for_combo([GESTURE_ROCK, GESTURE_PAPER, GESTURE_ROCK])

    assert spell is not None
    assert spell.key == "meteor"


def test_magic_system_unlock_all_spells_unlocks_every_locked_spell() -> None:
    """Debug unlock should unlock every spell exactly once."""
    magic = MagicSystem()
    initially_locked = len(magic.locked_spells())

    unlocked_count = magic.unlock_all_spells()

    assert unlocked_count == initially_locked
    assert magic.locked_spells() == []
    assert magic.unlock_all_spells() == 0
