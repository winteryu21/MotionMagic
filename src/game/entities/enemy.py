"""적 유닛과 적 종류별 기본 스탯."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pygame

from src.game.settings import COLOR_ENEMY, COLOR_ENEMY_FAST, COLOR_ENEMY_TANK, COLOR_HUD_HP, COLOR_STUN


@dataclass(frozen=True, slots=True)
class EnemyStat:
    """적 종류별 고정 스탯."""

    key: str
    display_name: str
    hp: float
    speed: float
    siege_damage: float
    size: int
    defense_rate: float = 0.0


ENEMY_STATS: dict[str, EnemyStat] = {
    "normal": EnemyStat(
        key="normal",
        display_name="일반 적",
        hp=40.0,
        speed=62.0,
        siege_damage=8.0,
        size=28,
        defense_rate=0.0,
    ),
    "fast": EnemyStat(
        key="fast",
        display_name="빠른 적",
        hp=24.0,
        speed=108.0,
        siege_damage=6.0,
        size=22,
        defense_rate=0.0,
    ),
    "tank": EnemyStat(
        key="tank",
        display_name="탱커 적",
        hp=95.0,
        speed=36.0,
        siege_damage=16.0,
        size=42,
        defense_rate=0.10,
    ),
}


@dataclass(slots=True)
class StatusEffect:
    kind: str
    duration: float
    tick_damage: float = 0.0
    tick_interval: float = 1.0
    tick_timer: float = 0.0


@dataclass(slots=True)
class Enemy:
    x: float
    y: float
    hp: float = 35.0
    max_hp: float = 35.0
    speed: float = 65.0
    siege_damage: float = 8.0
    size: int = 28
    enemy_type: str = "normal"
    defense_rate: float = 0.0
    field_index: int = 0
    advance_direction: int = -1
    status_effects: list[StatusEffect] = field(default_factory=list)
    reached_base: bool = False

    @classmethod
    def from_type(
        cls,
        enemy_type: str,
        x: float,
        y: float,
        field_index: int = 0,
        advance_direction: int = -1,
    ) -> "Enemy":
        stat = ENEMY_STATS[enemy_type]
        return cls(
            x=x,
            y=y,
            hp=stat.hp,
            max_hp=stat.hp,
            speed=stat.speed,
            siege_damage=stat.siege_damage,
            size=stat.size,
            enemy_type=stat.key,
            defense_rate=stat.defense_rate,
            field_index=field_index,
            advance_direction=advance_direction,
        )

    @property
    def alive(self) -> bool:
        return self.hp > 0 and not self.reached_base

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.size / 2),
            int(self.y - self.size / 2),
            self.size,
            self.size,
        )

    def distance_to(self, pos: tuple[float, float]) -> float:
        return math.hypot(self.x - pos[0], self.y - pos[1])

    def take_damage(self, amount: float) -> None:
        reduced = amount * max(0.0, 1.0 - self.defense_rate)
        self.hp -= reduced

    def apply_status(self, effect: StatusEffect) -> None:
        self.status_effects.append(effect)

    def update(self, dt: float, base_line_x: float) -> None:
        stunned = False
        for effect in list(self.status_effects):
            effect.duration -= dt
            if effect.kind == "stun":
                stunned = True
            elif effect.kind == "dot":
                effect.tick_timer -= dt
                if effect.tick_timer <= 0:
                    self.take_damage(effect.tick_damage)
                    effect.tick_timer = effect.tick_interval
            if effect.duration <= 0:
                self.status_effects.remove(effect)

        if not stunned:
            self.x += self.speed * self.advance_direction * dt

        if self.advance_direction < 0 and self.x <= base_line_x:
            self.reached_base = True
        elif self.advance_direction > 0 and self.x >= base_line_x:
            self.reached_base = True

    def draw(self, surface: pygame.Surface, active: bool = True) -> None:
        if self.enemy_type == "fast":
            color = COLOR_ENEMY_FAST
        elif self.enemy_type == "tank":
            color = COLOR_ENEMY_TANK
        else:
            color = COLOR_ENEMY
        if not active:
            color = tuple(max(0, c // 2) for c in color)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.size // 2)

        if any(effect.kind == "stun" for effect in self.status_effects):
            pygame.draw.circle(surface, COLOR_STUN, (int(self.x), int(self.y)), self.size // 2 + 4, 2)

        bar_w = self.size + 12
        bar_h = 5
        ratio = max(0.0, min(1.0, self.hp / self.max_hp))
        bg = pygame.Rect(int(self.x - bar_w / 2), int(self.y + self.size / 2 + 4), bar_w, bar_h)
        fg = pygame.Rect(bg.x, bg.y, int(bar_w * ratio), bar_h)
        pygame.draw.rect(surface, (60, 60, 70), bg)
        pygame.draw.rect(surface, COLOR_HUD_HP, fg)
