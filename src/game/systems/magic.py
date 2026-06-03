"""마법 시스템 — 원본 화면/씬 구조에 feature-game-proto 전투 마법만 이식."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from src.game.entities.player import Player
from src.game.entities.projectile import (
    Explosion,
    Fireball,
    LightningStrike,
    MagicMissile,
    Meteor,
    PiercingBullet,
)
from src.game.settings import (
    CHAIN_LIGHTNING_SEGMENT_DELAY,
    COLOR_LIGHTNING,
    GESTURE_PAPER,
    GESTURE_ROCK,
    GESTURE_SCISSORS,
    METEOR_COUNT_PER_FIELD,
    METEOR_START_X_OFFSET,
    METEOR_START_Y,
    METEOR_TARGET_MARGIN_X,
    METEOR_TARGET_MARGIN_Y,
    PLAYER_X,
    PLAYER_Y,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


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
        return (
            self.is_unlocked()
            and player.mana >= stat.mana_cost
            and self.cooldown_remaining(player, now) <= 0.0
        )

    def level_up(self) -> None:
        self.level = min(self.level + 1, max(self.level_table))


class MagicSystem:
    def __init__(self) -> None:
        self.spells: dict[str, Spell] = {
            # ── 1티어 ──
            "magic_missile": Spell(
                key="magic_missile",
                name="매직 미사일",
                combo=(GESTURE_SCISSORS,),
                level_table={
                    1: MagicLevelStat(
                        cooldown=0.45,
                        damage=20,
                        mana_cost=8,
                        projectile_speed=480,
                        pierce_count=1,
                    ),
                    2: MagicLevelStat(
                        cooldown=0.38,
                        damage=26,
                        mana_cost=9,
                        projectile_speed=540,
                        pierce_count=2,
                    ),
                    3: MagicLevelStat(
                        cooldown=0.32,
                        damage=34,
                        mana_cost=10,
                        projectile_speed=600,
                        pierce_count=3,
                    ),
                },
            ),
            "fireball": Spell(
                key="fireball",
                name="화염구",
                combo=(GESTURE_ROCK,),
                level_table={
                    1: MagicLevelStat(
                        cooldown=3.0,
                        damage=40,
                        mana_cost=30,
                        projectile_speed=350,
                        radius=120,
                    ),
                    2: MagicLevelStat(
                        cooldown=2.5,
                        damage=48,
                        mana_cost=35,
                        projectile_speed=400,
                        radius=140,
                    ),
                    3: MagicLevelStat(
                        cooldown=2.0,
                        damage=58,
                        mana_cost=40,
                        projectile_speed=450,
                        radius=160,
                    ),
                },
            ),
            "chain_lightning": Spell(
                key="chain_lightning",
                name="체인 라이트닝",
                combo=(GESTURE_PAPER,),
                level_table={
                    1: MagicLevelStat(
                        cooldown=5.0,
                        damage=25,
                        mana_cost=45,
                        radius=500,
                        chain_count=4,
                    ),
                    2: MagicLevelStat(
                        cooldown=4.5,
                        damage=30,
                        mana_cost=50,
                        radius=560,
                        chain_count=5,
                    ),
                    3: MagicLevelStat(
                        cooldown=4.0,
                        damage=36,
                        mana_cost=55,
                        radius=620,
                        chain_count=6,
                    ),
                },
            ),
            # ── 2티어 ──
            "lightning": Spell(
                key="lightning",
                name="낙뢰",
                combo=(GESTURE_SCISSORS, GESTURE_ROCK),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(
                        cooldown=1.1,
                        damage=24,
                        mana_cost=16,
                        radius=60,
                        status_effect="stun",
                    ),
                    2: MagicLevelStat(
                        cooldown=0.95,
                        damage=34,
                        mana_cost=19,
                        radius=78,
                        status_effect="stun",
                    ),
                    3: MagicLevelStat(
                        cooldown=0.8,
                        damage=46,
                        mana_cost=23,
                        radius=95,
                        status_effect="stun",
                    ),
                },
            ),
            "explosion": Spell(
                key="explosion",
                name="폭발",
                combo=(GESTURE_ROCK, GESTURE_PAPER),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(
                        cooldown=1.4,
                        damage=30,
                        mana_cost=18,
                        radius=85,
                        status_effect="dot",
                    ),
                    2: MagicLevelStat(
                        cooldown=1.2,
                        damage=42,
                        mana_cost=21,
                        radius=105,
                        status_effect="dot",
                    ),
                    3: MagicLevelStat(
                        cooldown=1.0,
                        damage=55,
                        mana_cost=25,
                        radius=130,
                        status_effect="dot",
                    ),
                },
            ),
            "piercing_bullet": Spell(
                key="piercing_bullet",
                name="마탄",
                combo=(GESTURE_PAPER, GESTURE_SCISSORS),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(
                        cooldown=1.0,
                        damage=20,
                        mana_cost=15,
                        projectile_speed=600,
                        pierce_count=3,
                    ),
                    2: MagicLevelStat(
                        cooldown=0.8,
                        damage=24,
                        mana_cost=18,
                        projectile_speed=660,
                        pierce_count=4,
                    ),
                    3: MagicLevelStat(
                        cooldown=0.6,
                        damage=28,
                        mana_cost=21,
                        projectile_speed=720,
                        pierce_count=5,
                    ),
                },
            ),
            # ── 3티어 ──
            "meteor": Spell(
                key="meteor",
                name="메테오",
                combo=(GESTURE_ROCK, GESTURE_PAPER, GESTURE_ROCK),
                unlocked=False,
                level_table={
                    1: MagicLevelStat(
                        cooldown=8.0,
                        damage=120,
                        mana_cost=70,
                        radius=200,
                        status_effect="dot",
                    ),
                    2: MagicLevelStat(
                        cooldown=7.0,
                        damage=150,
                        mana_cost=80,
                        radius=230,
                        status_effect="dot",
                    ),
                    3: MagicLevelStat(
                        cooldown=6.0,
                        damage=180,
                        mana_cost=90,
                        radius=260,
                        status_effect="dot",
                    ),
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
        if not combo:
            return None
        # 가장 긴 접미사부터 매칭 시도
        for i in range(len(combo)):
            suffix = tuple(combo[i:])
            for spell in self.spells.values():
                if spell.is_unlocked() and spell.combo == suffix:
                    return spell
        return None

    def cast_by_combo(
        self,
        combo: list[str],
        player: Player,
        field,
        aim_pos: tuple[int, int],
        *,
        fields: list | None = None,
        origin_pos: tuple[int, int] | None = None,
    ) -> str:
        spell = self.spell_for_combo(combo)
        if spell is None:
            return "조합 없음"
        return self.cast(
            spell,
            player,
            field,
            aim_pos,
            fields=fields,
            origin_pos=origin_pos,
        )

    def cast(
        self,
        spell: Spell,
        player: Player,
        field,
        aim_pos: tuple[int, int],
        *,
        ignore_requirements: bool = False,
        consume_resources: bool = True,
        fields: list | None = None,
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
            enemies = getattr(field, "enemies", [])
            target_enemy = None
            if enemies:
                target_enemy = min(
                    [e for e in enemies if e.alive],
                    key=lambda e: math.hypot(e.x - aim_pos[0], e.y - aim_pos[1]),
                    default=None,
                )

            if target_enemy is not None:
                field.projectiles.append(
                    MagicMissile(
                        x=origin[0],
                        y=origin[1],
                        target=target_enemy,
                        damage=stat.damage,
                        speed=stat.projectile_speed,
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

        if spell.key == "chain_lightning":
            hit_path = self._cast_chain_lightning(
                start_pos=(int(cast_x), int(cast_y)),
                field=field,
                chain_limit=stat.chain_count,
                chain_range=stat.radius,
                damage=stat.damage,
            )
            field.effects.append(
                LightningStrike(
                    hit_path,
                    damage=stat.damage,
                    segment_delay=CHAIN_LIGHTNING_SEGMENT_DELAY,
                )
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "lightning":
            field.effects.append(LightningStrike(aim_pos[0], aim_pos[1], stat.damage))
            # 낙뢰는 좁은 범위 폭발 피해와 함께 스턴(기절) 부여
            field.effects.append(
                Explosion(
                    aim_pos[0],
                    aim_pos[1],
                    stat.damage * 0.5,
                    stat.radius,
                    duration=0.2,
                    status_effect="stun",
                    color=COLOR_LIGHTNING,
                )
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "explosion":
            field.effects.append(
                Explosion(aim_pos[0], aim_pos[1], stat.damage, stat.radius)
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "piercing_bullet":
            field.projectiles.append(
                PiercingBullet.toward(
                    origin=origin,
                    target=aim_pos,
                    damage=stat.damage,
                    speed=stat.projectile_speed,
                    field_index=field_index,
                    pierce_limit=stat.pierce_count,
                )
            )
            return f"{spell.name} Lv.{spell.level}"

        if spell.key == "meteor":
            target_fields = fields if fields is not None else [field]
            for target_field in target_fields:
                for _ in range(METEOR_COUNT_PER_FIELD):
                    target_x = random.randint(
                        METEOR_TARGET_MARGIN_X,
                        SCREEN_WIDTH - METEOR_TARGET_MARGIN_X,
                    )
                    target_y = random.randint(
                        METEOR_TARGET_MARGIN_Y,
                        SCREEN_HEIGHT - METEOR_TARGET_MARGIN_Y,
                    )
                    target_field.effects.append(
                        Meteor(
                            x=target_x + METEOR_START_X_OFFSET,
                            y=METEOR_START_Y,
                            target_x=target_x,
                            target_y=target_y,
                            damage=stat.damage,
                            explosion_radius=stat.radius,
                        )
                    )
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
            hit.append(closest)
            current_pos = (closest.x, closest.y)
            path.append((int(closest.x), int(closest.y)))

        if len(path) == 1:
            target_x = start_pos[0] + (260 if getattr(field, "index", 0) == 0 else -260)
            path.append((target_x, start_pos[1]))
        return path

    def possible_spells(self, partial_combo: list[str]) -> list[Spell]:
        unlocked_spells = [
            spell for spell in self.spells.values() if spell.is_unlocked()
        ]
        if not partial_combo:
            return unlocked_spells
        return [
            spell
            for spell in unlocked_spells
            if spell.combo[: len(partial_combo)] == tuple(partial_combo)
        ]
