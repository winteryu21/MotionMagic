"""투사체 및 마법 오브젝트.

원본 BattleScene/Hud 구조(field.enemies, field.projectiles, field.effects)는 유지하고,
feature-game-proto의 마탄/화염구/체인 라이트닝 전투 로직만 리스트 기반으로 이식한 파일.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pygame

from src.game.entities.enemy import Enemy, StatusEffect
from src.game.settings import COLOR_EXPLOSION, COLOR_LIGHTNING, COLOR_PROJECTILE, SCREEN_HEIGHT, SCREEN_WIDTH


@dataclass(slots=True)
class Projectile:
    """속도 벡터 기반 투사체 기본 클래스."""

    x: float
    y: float
    speed_x: float
    speed_y: float
    damage: float
    field_index: int = 0
    radius: int = 7
    alive: bool = True

    @property
    def rect(self) -> pygame.Rect:
        size = self.radius * 2
        return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius), size, size)

    def update(self, dt: float) -> None:
        self.x += self.speed_x * dt
        self.y += self.speed_y * dt
        if self.x < -40 or self.x > SCREEN_WIDTH + 40 or self.y < -40 or self.y > SCREEN_HEIGHT + 40:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, COLOR_PROJECTILE, (int(self.x), int(self.y)), self.radius)


@dataclass(slots=True)
class MagicMissile(Projectile):
    """관통 마탄.

    기존 원본의 MagicMissile 이름은 유지하되, feature-game-proto의 PiercingBullet처럼
    방향 벡터로 날아가고 여러 적을 관통하도록 동작한다.
    """

    pierce_limit: int = 3
    hit_enemy_ids: set[int] = field(default_factory=set)

    @classmethod
    def toward(
        cls,
        origin: tuple[float, float],
        target: tuple[float, float],
        damage: float,
        speed: float,
        field_index: int = 0,
        pierce_limit: int = 3,
    ) -> "MagicMissile":
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        length = math.hypot(dx, dy)
        if length < 1.0:
            dx = 1.0 if field_index == 0 else -1.0
            dy = 0.0
            length = 1.0
        return cls(
            x=origin[0],
            y=origin[1],
            speed_x=dx / length * speed,
            speed_y=dy / length * speed,
            damage=damage,
            field_index=field_index,
            radius=6,
            pierce_limit=pierce_limit,
        )

    def can_hit(self, enemy: Enemy) -> bool:
        return id(enemy) not in self.hit_enemy_ids and len(self.hit_enemy_ids) < self.pierce_limit

    def register_hit(self, enemy: Enemy) -> None:
        self.hit_enemy_ids.add(id(enemy))
        if len(self.hit_enemy_ids) >= self.pierce_limit:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.ellipse(surface, (155, 89, 182), pygame.Rect(int(self.x - 10), int(self.y - 5), 20, 10))
        pygame.draw.ellipse(surface, (255, 255, 255), pygame.Rect(int(self.x - 6), int(self.y - 3), 12, 6))


@dataclass(slots=True)
class Fireball(Projectile):
    """화염구. 적과 충돌하면 폭발 효과를 생성한다."""

    explosion_radius: float = 95.0

    @classmethod
    def toward(
        cls,
        origin: tuple[float, float],
        target: tuple[float, float],
        damage: float,
        speed: float,
        field_index: int = 0,
        explosion_radius: float = 95.0,
    ) -> "Fireball":
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        length = math.hypot(dx, dy)
        if length < 1.0:
            dx = 1.0 if field_index == 0 else -1.0
            dy = 0.0
            length = 1.0
        return cls(
            x=origin[0],
            y=origin[1],
            speed_x=dx / length * speed,
            speed_y=dy / length * speed,
            damage=damage,
            field_index=field_index,
            radius=15,
            explosion_radius=explosion_radius,
        )

    def draw(self, surface: pygame.Surface) -> None:
        center = (int(self.x), int(self.y))
        pygame.draw.circle(surface, (230, 126, 34), center, self.radius)
        pygame.draw.circle(surface, (241, 196, 15), center, max(4, self.radius - 5))
        pygame.draw.circle(surface, (255, 255, 255), center, max(2, self.radius - 10))


@dataclass(slots=True)
class Explosion:
    x: float
    y: float
    damage: float
    radius: float
    duration: float = 0.35
    age: float = 0.0
    applied: bool = False
    alive: bool = True

    def update(self, dt: float, enemies: list[Enemy]) -> None:
        if not self.applied:
            for enemy in enemies:
                if enemy.alive and enemy.distance_to((self.x, self.y)) <= self.radius:
                    enemy.take_damage(self.damage)
                    enemy.apply_status(StatusEffect("dot", duration=2.0, tick_damage=max(2.0, self.damage * 0.08), tick_interval=0.5))
            self.applied = True
        self.age += dt
        self.alive = self.age < self.duration

    def draw(self, surface: pygame.Surface) -> None:
        t = min(1.0, self.age / self.duration)
        r = max(6, int(self.radius * (0.25 + 0.75 * t)))
        pygame.draw.circle(surface, COLOR_EXPLOSION, (int(self.x), int(self.y)), r, 3)


@dataclass(slots=True)
class LightningStrike:
    """체인 라이트닝 시각 효과 및 즉시 타격 처리."""

    path: list[tuple[int, int]]
    damage: float
    duration: float = 0.22
    age: float = 0.0
    applied: bool = True
    alive: bool = True

    # 기존 UnlockScene 호환용: LightningStrike(x, y, damage, radius) 형태도 허용하기 위해 __init__ 직접 정의
    def __init__(self, x_or_path, y: float | None = None, damage: float = 0.0, radius: float = 0.0, duration: float = 0.22) -> None:
        if isinstance(x_or_path, list):
            self.path = x_or_path
            self.damage = damage
        else:
            x = int(x_or_path)
            yy = int(y if y is not None else 0)
            self.path = [(x, 70), (x, yy)]
            self.damage = damage
        self.duration = duration
        self.age = 0.0
        self.applied = True
        self.alive = True

    def update(self, dt: float, enemies: list[Enemy]) -> None:
        self.age += dt
        self.alive = self.age < self.duration

    def draw(self, surface: pygame.Surface) -> None:
        if len(self.path) < 2:
            return
        for start, end in zip(self.path, self.path[1:]):
            pygame.draw.line(surface, COLOR_LIGHTNING, start, end, 5)
            pygame.draw.circle(surface, COLOR_LIGHTNING, end, 14, 2)
