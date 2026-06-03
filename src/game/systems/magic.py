"""마법 시스템 — 개발용 q/w/e 조합으로 마법 발동."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from src.game.entities.player import Player
from src.game.entities.projectile import Explosion, LightningStrike, MagicMissile
from src.game.settings import GESTURE_PAPER, GESTURE_ROCK, GESTURE_SCISSORS, PLAYER_X, PLAYER_Y


@dataclass(frozen=True, slots=True)
class MagicLevelStat:
    cooldown: float
    damage: float
    mana_cost: float
    projectile_speed: float = 420.0
    radius: float = 80.0
    target_count: int = 1
    unlocked: bool = True
    status_effect: str | None = None


@dataclass(slots=True)
class Spell:
    name: str
    combo: tuple[str, str]
    level_table: dict[int, MagicLevelStat]
    level: int = 1
    unlocked: bool = True
    last_cast_time: float = field(default=-999.0)

    @property
    def stat(self) -> MagicLevelStat:
        return self.level_table[min(self.level, max(self.level_table))]

    def is_unlocked(self) -> bool:
        return self.unlocked and self.stat.unlocked

    def cooldown_duration(self, player: Player) -> float:
        return self.stat.cooldown * player.global_cooldown_multiplier

    def cooldown_remaining(self, player: Player, now: float) -> float:
        return max(0.0, self.cooldown_duration(player) - (now - self.last_cast_time))

    def can_cast(self, player: Player, now: float) -> bool:
        stat = self.stat
        return self.is_unlocked() and player.mana >= stat.mana_cost and self.cooldown_remaining(player, now) <= 0.0

    def level_up(self) -> None:
        self.level = min(self.level + 1, max(self.level_table))


class MagicSystem:
    def __init__(self) -> None:
        self.spells: dict[str, Spell] = {
            "magic_missile": Spell(
                name="매직 미사일",
                combo=(GESTURE_SCISSORS, GESTURE_ROCK),
                level_table={
                    1: MagicLevelStat(cooldown=0.45, damage=20, mana_cost=8, projectile_speed=480, target_count=1),
                    2: MagicLevelStat(cooldown=0.38, damage=26, mana_cost=9, projectile_speed=540, target_count=2),
                    3: MagicLevelStat(cooldown=0.32, damage=34, mana_cost=10, projectile_speed=600, target_count=3),
                },
            ),
            "explosion": Spell(
                name="폭발",
                combo=(GESTURE_ROCK, GESTURE_PAPER),
                level_table={
                    1: MagicLevelStat(cooldown=1.4, damage=30, mana_cost=18, radius=85, status_effect="dot"),
                    2: MagicLevelStat(cooldown=1.2, damage=42, mana_cost=21, radius=105, status_effect="dot"),
                    3: MagicLevelStat(cooldown=1.0, damage=55, mana_cost=25, radius=130, status_effect="dot"),
                },
            ),
            "lightning": Spell(
                name="낙뢰",
                combo=(GESTURE_PAPER, GESTURE_SCISSORS),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(cooldown=1.1, damage=24, mana_cost=16, radius=60, status_effect="stun"),
                    2: MagicLevelStat(cooldown=0.95, damage=34, mana_cost=19, radius=78, status_effect="stun"),
                    3: MagicLevelStat(cooldown=0.8, damage=46, mana_cost=23, radius=95, status_effect="stun"),
                },
            ),
        }

    def locked_spells(self) -> list[Spell]:
        return [spell for spell in self.spells.values() if not spell.is_unlocked()]

    def unlock_spell(self, spell: Spell) -> None:
        spell.unlocked = True
        spell.level = max(1, spell.level)

    def unlock_spell_by_key(self, spell_key: str) -> Spell | None:
        spell = self.spells.get(spell_key)
        if spell is None:
            return None
        self.unlock_spell(spell)
        return spell

    def spell_for_combo(self, combo: list[str]) -> Spell | None:
        if len(combo) < 2:
            return None
        pair = tuple(combo[-2:])
        for spell in self.spells.values():
            if spell.is_unlocked() and spell.combo == pair:
                return spell
        return None

    def cast_by_combo(self, combo: list[str], player: Player, field, aim_pos: tuple[int, int]) -> str:
        spell = self.spell_for_combo(combo)
        if spell is None:
            return "조합 없음"
        return self.cast(spell, player, field, aim_pos)

    def cast(
        self,
        spell: Spell,
        player: Player,
        field,
        aim_pos: tuple[int, int],
        *,
        ignore_requirements: bool = False,
        consume_resources: bool = True,
        origin_pos: tuple[int, int] | None = None,
    ) -> str:
        now = time.monotonic()
        if not ignore_requirements and not spell.can_cast(player, now):
            return "마나/쿨타임 부족"

        stat = spell.stat
        if consume_resources:
            if not player.spend_mana(stat.mana_cost):
                return "마나 부족"
            spell.last_cast_time = now

        cast_x, cast_y = origin_pos if origin_pos is not None else (PLAYER_X, PLAYER_Y)

        if spell.name == "매직 미사일":
            targets = sorted(
                [enemy for enemy in field.enemies if enemy.alive],
                key=lambda enemy: enemy.distance_to(aim_pos),
            )[: stat.target_count]
            for target in targets:
                field.projectiles.append(
                    MagicMissile(
                        x=cast_x,
                        y=cast_y,
                        target=target,
                        damage=stat.damage,
                        speed=stat.projectile_speed,
                    )
                )
            return f"{spell.name} Lv.{spell.level}"

        if spell.name == "폭발":
            field.effects.append(Explosion(aim_pos[0], aim_pos[1], stat.damage, stat.radius))
            return f"{spell.name} Lv.{spell.level}"

        if spell.name == "낙뢰":
            field.effects.append(LightningStrike(aim_pos[0], aim_pos[1], stat.damage, stat.radius))
            return f"{spell.name} Lv.{spell.level}"

        return "미구현 마법"

    def possible_spells(self, partial_combo: list[str]) -> list[Spell]:
        unlocked_spells = [spell for spell in self.spells.values() if spell.is_unlocked()]
        if not partial_combo:
            return unlocked_spells
        return [spell for spell in unlocked_spells if spell.combo[: len(partial_combo)] == tuple(partial_combo)]
