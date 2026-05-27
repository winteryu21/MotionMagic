"""마법 시스템 — 원본 화면/씬 구조에 feature-game-proto 전투 마법만 이식."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from src.game.entities.enemy import StatusEffect
from src.game.entities.player import Player
from src.game.entities.projectile import Explosion, Fireball, LightningStrike, MagicMissile
from src.game.settings import GESTURE_PAPER, GESTURE_ROCK, GESTURE_SCISSORS, PLAYER_X, PLAYER_Y


@dataclass(frozen=True, slots=True)
class MagicLevelStat:
    cooldown: float
    damage: float
    mana_cost: float
    projectile_speed: float = 420.0
    radius: float = 80.0
    target_count: int = 1
    pierce_count: int = 1
    chain_count: int = 1
    unlocked: bool = True
    status_effect: str | None = None


@dataclass(slots=True)
class Spell:
    key: str
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
                key="magic_missile",
                name="마탄",
                combo=(GESTURE_SCISSORS, GESTURE_ROCK),
                level_table={
                    1: MagicLevelStat(cooldown=0.55, damage=20, mana_cost=8, projectile_speed=560, pierce_count=2),
                    2: MagicLevelStat(cooldown=0.48, damage=26, mana_cost=9, projectile_speed=620, pierce_count=3),
                    3: MagicLevelStat(cooldown=0.40, damage=34, mana_cost=10, projectile_speed=700, pierce_count=4),
                },
            ),
            "fireball": Spell(
                key="fireball",
                name="화염구",
                combo=(GESTURE_ROCK, GESTURE_PAPER),
                level_table={
                    1: MagicLevelStat(cooldown=1.45, damage=34, mana_cost=18, projectile_speed=360, radius=90, status_effect="dot"),
                    2: MagicLevelStat(cooldown=1.25, damage=46, mana_cost=21, projectile_speed=390, radius=115, status_effect="dot"),
                    3: MagicLevelStat(cooldown=1.05, damage=60, mana_cost=25, projectile_speed=420, radius=145, status_effect="dot"),
                },
            ),
            # key는 기존 unlock/reward 흐름 호환을 위해 lightning 유지, 표시명과 동작은 체인 라이트닝으로 변경
            "lightning": Spell(
                key="lightning",
                name="체인 라이트닝",
                combo=(GESTURE_PAPER, GESTURE_SCISSORS),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(cooldown=1.15, damage=24, mana_cost=16, radius=350, chain_count=4, status_effect="stun"),
                    2: MagicLevelStat(cooldown=1.00, damage=32, mana_cost=19, radius=380, chain_count=5, status_effect="stun"),
                    3: MagicLevelStat(cooldown=0.85, damage=42, mana_cost=23, radius=420, chain_count=6, status_effect="stun"),
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

    def cast_by_combo(
        self,
        combo: list[str],
        player: Player,
        field,
        aim_pos: tuple[int, int],
        *,
        origin_pos: tuple[int, int] | None = None,
    ) -> str:
        spell = self.spell_for_combo(combo)
        if spell is None:
            return "조합 없음"
        return self.cast(spell, player, field, aim_pos, origin_pos=origin_pos)

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
        origin = (float(cast_x), float(cast_y))
        field_index = getattr(field, "index", 0)

        if spell.key == "magic_missile":
            field.projectiles.append(
                MagicMissile.toward(
                    origin=origin,
                    target=aim_pos,
                    damage=stat.damage,
                    speed=stat.projectile_speed,
                    field_index=field_index,
                    pierce_limit=stat.pierce_count,
                )
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "fireball":
            field.projectiles.append(
                Fireball.toward(
                    origin=origin,
                    target=aim_pos,
                    damage=stat.damage,
                    speed=stat.projectile_speed,
                    field_index=field_index,
                    explosion_radius=stat.radius,
                )
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "lightning":
            hit_path = self._cast_chain_lightning(
                start_pos=(int(cast_x), int(cast_y)),
                field=field,
                chain_limit=stat.chain_count,
                chain_range=stat.radius,
                damage=stat.damage,
            )
            field.effects.append(LightningStrike(hit_path, damage=stat.damage))
            return f"{spell.name} Lv.{spell.level}"

        return "미구현 마법"

    def _cast_chain_lightning(
        self,
        start_pos: tuple[int, int],
        field,
        chain_limit: int,
        chain_range: float,
        damage: float,
    ) -> list[tuple[int, int]]:
        enemies = [enemy for enemy in field.enemies if enemy.alive]
        path: list[tuple[int, int]] = [start_pos]
        current_pos = (float(start_pos[0]), float(start_pos[1]))
        hit: list = []

        for _ in range(chain_limit):
            closest = None
            closest_dist = float("inf")
            for enemy in enemies:
                if enemy in hit:
                    continue
                dist = math.hypot(enemy.x - current_pos[0], enemy.y - current_pos[1])
                if dist <= chain_range and dist < closest_dist:
                    closest = enemy
                    closest_dist = dist
            if closest is None:
                break
            closest.take_damage(damage)
            closest.apply_status(StatusEffect("stun", duration=0.65))
            hit.append(closest)
            current_pos = (closest.x, closest.y)
            path.append((int(closest.x), int(closest.y)))

        if len(path) == 1:
            target_x = start_pos[0] + (260 if getattr(field, "index", 0) == 0 else -260)
            path.append((target_x, start_pos[1]))
        return path

    def possible_spells(self, partial_combo: list[str]) -> list[Spell]:
        unlocked_spells = [spell for spell in self.spells.values() if spell.is_unlocked()]
        if not partial_combo:
            return unlocked_spells
        return [spell for spell in unlocked_spells if spell.combo[: len(partial_combo)] == tuple(partial_combo)]
