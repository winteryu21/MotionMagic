"""스테이지 클리어 보상 시스템."""

from __future__ import annotations

from dataclasses import dataclass
import random

from src.game.entities.player import Player
from src.game.systems.magic import MagicSystem, Spell


@dataclass(slots=True)
class RewardOption:
    category: str
    key: str
    title: str
    description: str


class RewardSystem:
    """플레이어 스탯 또는 마법 레벨업 중 3개를 뽑고 1개를 적용함."""

    def make_options(self, player: Player, magic: MagicSystem, count: int = 3) -> list[RewardOption]:
        pool: list[RewardOption] = [
            RewardOption("player", "max_hp", "최대 체력 강화", "플레이어 최대 체력 +10\n현재 체력도 최대치까지 회복"),
            RewardOption("player", "max_mana", "최대 마나 강화", "플레이어 최대 마나 +15\n현재 마나도 최대치까지 충전"),
            RewardOption("player", "mana_recovery", "마나 회복 강화", "마나 충전량 +15%\nSpace 유지 시 더 빠르게 회복"),
            RewardOption("player", "cooldown", "시전 속도 강화", "전체 마법 쿨타임 5% 감소\n최대 40%까지 감소"),
        ]

        for spell_key, spell in magic.spells.items():
            if spell.is_unlocked() and spell.level < max(spell.level_table):
                next_stat = spell.level_table[spell.level + 1]
                pool.append(
                    RewardOption(
                        "spell",
                        spell_key,
                        f"{spell.name} Lv.{spell.level + 1}",
                        self._spell_description(spell, next_stat),
                    )
                )

        random.shuffle(pool)
        return pool[: min(count, len(pool))]

    def apply(self, option: RewardOption, player: Player, magic: MagicSystem) -> str:
        if option.category == "player":
            if option.key == "max_hp":
                player.max_hp += 10
                player.hp = player.max_hp
                return "최대 체력 +10 적용"
            if option.key == "max_mana":
                player.max_mana += 15
                player.mana = player.max_mana
                return "최대 마나 +15 적용"
            if option.key == "mana_recovery":
                player.mana_recovery_multiplier += 0.15
                return "마나 충전량 +15% 적용"
            if option.key == "cooldown":
                player.global_cooldown_multiplier = max(0.6, player.global_cooldown_multiplier - 0.05)
                return "전체 쿨타임 5% 감소 적용"

        spell = magic.spells[option.key]
        before = spell.level
        spell.level_up()
        return f"{spell.name} Lv.{before} → Lv.{spell.level} 적용"

    def _spell_description(self, spell: Spell, stat) -> str:
        lines = [
            f"피해량 {int(stat.damage)} / 마나 {int(stat.mana_cost)}",
            f"쿨타임 {stat.cooldown:.2f}s",
        ]
        if getattr(spell, "key", "") == "magic_missile":
            lines.append(f"관통 {stat.pierce_count}회 / 속도 {int(stat.projectile_speed)}")
        elif getattr(spell, "key", "") == "lightning":
            lines.append(f"체인 {stat.chain_count}회 / 사거리 {int(stat.radius)}")
            lines.append("상태이상: 스턴")
        else:
            lines.append(f"범위 {int(stat.radius)}")
            if stat.status_effect == "dot":
                lines.append("상태이상: 지속피해")
            elif stat.status_effect == "stun":
                lines.append("상태이상: 스턴")
        return "\n".join(lines)
