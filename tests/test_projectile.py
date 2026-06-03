"""Tests for game projectiles."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace

import pytest

if find_spec("pygame") is None:
    sys.modules["pygame"] = ModuleType("pygame")

from src.game.entities.enemy import Enemy
from src.game.entities.projectile import LightningStrike, MagicMissile
from src.game.systems.combat import CombatSystem


def test_magic_missile_tracks_target_and_damages_on_arrival() -> None:
    """Magic missile should follow the target's latest position."""
    target = Enemy(x=10.0, y=0.0, hp=30.0, max_hp=30.0)
    missile = MagicMissile(x=0.0, y=0.0, target=target, damage=7.0, speed=5.0)

    missile.update(1.0)
    assert (missile.x, missile.y) == pytest.approx((5.0, 0.0))

    target.x = 5.0
    target.y = 5.0
    missile.update(1.0)

    assert target.hp == pytest.approx(23.0)
    assert not missile.alive


def test_magic_missile_disappears_when_target_is_dead() -> None:
    """Magic missile should disappear if its target is already dead."""
    target = Enemy(x=10.0, y=0.0, hp=0.0, max_hp=30.0)
    missile = MagicMissile(x=0.0, y=0.0, target=target, damage=7.0, speed=5.0)

    missile.update(1.0)

    assert not missile.alive


def test_combat_system_does_not_apply_magic_missile_damage() -> None:
    """Magic missile damage should only be applied by its own update."""
    target = Enemy(x=0.0, y=0.0, hp=30.0, max_hp=30.0)
    missile = MagicMissile(x=0.0, y=0.0, target=target, damage=7.0, speed=5.0)
    field = SimpleNamespace(projectiles=[missile], enemies=[target], effects=[])

    CombatSystem.update_projectile_collisions(field)

    assert target.hp == pytest.approx(30.0)


def test_lightning_strike_reveals_chain_segments_over_time() -> None:
    """Chain lightning visual segments should appear one by one."""
    strike = LightningStrike(
        [(0, 0), (10, 0), (20, 0), (30, 0)],
        damage=10.0,
        segment_delay=0.06,
    )

    assert strike._visible_segment_count() == 1

    strike.update(0.06, [])
    assert strike._visible_segment_count() == 2

    strike.update(0.06, [])
    assert strike._visible_segment_count() == 3
