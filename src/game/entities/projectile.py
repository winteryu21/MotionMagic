"""투사체 및 마법 오브젝트."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pygame

from src.game.entities.enemy import Enemy, StatusEffect
from src.game.settings import COLOR_EXPLOSION, COLOR_LIGHTNING, COLOR_PROJECTILE


@dataclass(slots=True)
class MagicMissile:
    x: float
    y: float
    target: Enemy
    damage: float
    speed: float
    radius: int = 6
    alive: bool = True

    def update(self, dt: float) -> None:
        if not self.target.alive:
            self.alive = False
            return
        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = max(1.0, math.hypot(dx, dy))
        step = self.speed * dt
        if step >= dist:
            self.target.take_damage(self.damage)
            self.alive = False
            return
        self.x += dx / dist * step
        self.y += dy / dist * step

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, COLOR_PROJECTILE, (int(self.x), int(self.y)), self.radius)


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
                    enemy.apply_status(StatusEffect("dot", duration=2.0, tick_damage=3.0, tick_interval=0.5))
            self.applied = True
        self.age += dt
        self.alive = self.age < self.duration

    def draw(self, surface: pygame.Surface) -> None:
        t = min(1.0, self.age / self.duration)
        r = max(4, int(self.radius * t))
        pygame.draw.circle(surface, COLOR_EXPLOSION, (int(self.x), int(self.y)), r, 3)


@dataclass(slots=True)
class LightningStrike:
    x: float
    y: float
    damage: float
    radius: float
    duration: float = 0.22
    age: float = 0.0
    applied: bool = False
    alive: bool = True

    def update(self, dt: float, enemies: list[Enemy]) -> None:
        if not self.applied:
            for enemy in enemies:
                if enemy.alive and enemy.distance_to((self.x, self.y)) <= self.radius:
                    enemy.take_damage(self.damage)
                    enemy.apply_status(StatusEffect("stun", duration=0.8))
            self.applied = True
        self.age += dt
        self.alive = self.age < self.duration

    def draw(self, surface: pygame.Surface) -> None:
        top = (int(self.x), 70)
        bottom = (int(self.x), int(self.y))
        pygame.draw.line(surface, COLOR_LIGHTNING, top, bottom, 5)
        pygame.draw.circle(surface, COLOR_LIGHTNING, bottom, int(self.radius), 2)
