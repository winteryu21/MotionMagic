"""Tests for reward descriptions."""

from __future__ import annotations

from src.game.settings import METEOR_COUNT_PER_FIELD
from src.game.systems.magic import MagicSystem
from src.game.systems.reward import RewardSystem


def test_spell_descriptions_match_current_implementations() -> None:
    """Spell upgrade descriptions should describe their actual behavior."""
    magic = MagicSystem()
    rewards = RewardSystem()

    missile = magic.spells["magic_missile"]
    missile_description = rewards._spell_description(missile, missile.stat)
    assert "단일 대상 추적" in missile_description
    assert "관통" not in missile_description

    piercing = magic.spells["piercing_bullet"]
    piercing_description = rewards._spell_description(piercing, piercing.stat)
    assert f"관통 {piercing.stat.pierce_count}회" in piercing_description

    meteor = magic.spells["meteor"]
    meteor_description = rewards._spell_description(meteor, meteor.stat)
    assert f"양쪽 전장 랜덤 낙하 {METEOR_COUNT_PER_FIELD}개" in meteor_description

    chain = magic.spells["chain_lightning"]
    chain_description = rewards._spell_description(chain, chain.stat)
    assert f"체인 {chain.stat.chain_count}회" in chain_description


def test_chain_lightning_range_is_expanded_by_level() -> None:
    """Chain lightning should use the expanded chain search radius."""
    magic = MagicSystem()
    chain = magic.spells["chain_lightning"]

    assert chain.level_table[1].radius == 500
    assert chain.level_table[2].radius == 560
    assert chain.level_table[3].radius == 620
