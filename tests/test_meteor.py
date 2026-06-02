"""Tests for the meteor spell."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.game.entities.enemy import Enemy
from src.game.entities.player import Player
from src.game.entities.projectile import Meteor
from src.game.settings import METEOR_COUNT_PER_FIELD
from src.game.systems.magic import MagicSystem


def test_meteor_damages_enemies_only_after_impact() -> None:
    """Meteor should apply its explosion damage when it reaches the ground."""
    enemy = Enemy(x=100.0, y=200.0, hp=200.0, max_hp=200.0)
    meteor = Meteor(
        x=200.0,
        y=100.0,
        target_x=100.0,
        target_y=200.0,
        damage=50.0,
        explosion_radius=80.0,
        fall_speed=200.0,
    )

    meteor.update(0.5, [enemy])
    assert enemy.hp == pytest.approx(200.0)
    assert meteor.explosion is None
    assert meteor.x < 200.0
    assert meteor.y > 100.0

    meteor.update(0.5, [enemy])
    assert enemy.hp == pytest.approx(150.0)
    assert meteor.explosion is not None
    assert meteor.alive


def test_meteor_spell_creates_random_effects_in_both_fields() -> None:
    """Meteor cast should add random falling effects to each battlefield."""
    magic = MagicSystem()
    spell = magic.unlock_spell_by_key("meteor")
    assert spell is not None
    fields = [
        SimpleNamespace(index=0, effects=[]),
        SimpleNamespace(index=1, effects=[]),
    ]

    with patch(
        "src.game.systems.magic.random.randint",
        side_effect=[300, 400] * METEOR_COUNT_PER_FIELD * len(fields),
    ):
        magic.cast(
            spell,
            Player(mana=999.0),
            fields[0],
            (999, 999),
            fields=fields,
            ignore_requirements=True,
            consume_resources=False,
        )

    assert all(len(field.effects) == METEOR_COUNT_PER_FIELD for field in fields)
    assert all(
        isinstance(effect, Meteor)
        for field in fields
        for effect in field.effects
    )
    assert all(
        (effect.target_x, effect.target_y) == (300, 400)
        for field in fields
        for effect in field.effects
    )
