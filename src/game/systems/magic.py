"""마법 시스템 — 제스처 조합 및 단축키로 마법 발동."""

from __future__ import annotations

import math
import random
from typing import Optional

import pygame

from src.game.entities.projectile import Projectile
from src.game.game_manager import GameManager
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.systems.spell_data import SPELL_REGISTRY

_VIEW_HEIGHT: int = SCREEN_HEIGHT


class PiercingBullet(Projectile):
    """관통 마탄 투사체. 적을 뚫고 지나가며 다수의 적에게 피해를 준다."""

    def __init__(
        self,
        x: float,
        y: float,
        speed_x: float,
        speed_y: float,
        damage: float,
        field_id: int,
        pierce_limit: int = 3,
    ) -> None:
        image = pygame.Surface((20, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(image, (155, 89, 182), (0, 0, 20, 10))
        pygame.draw.ellipse(image, (255, 255, 255), (4, 2, 12, 6))

        super().__init__(x, y, speed_x, speed_y, damage, field_id, image=image)
        self.pierce_limit = pierce_limit
        self.hit_enemies: set[pygame.sprite.Sprite] = set()

    def can_hit(self, enemy: pygame.sprite.Sprite) -> bool:
        return enemy not in self.hit_enemies and len(self.hit_enemies) < self.pierce_limit


class Fireball(Projectile):
    """화염구 투사체. 충돌 시 폭발하여 범위 내 모든 적에게 AoE 피해를 입힌다."""

    def __init__(
        self,
        x: float,
        y: float,
        speed_x: float,
        speed_y: float,
        damage: float,
        field_id: int,
        explosion_radius: float = 1200.0,
    ) -> None:
        image = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(image, (230, 126, 34), (16, 16), 16)
        pygame.draw.circle(image, (241, 196, 15), (16, 16), 10)
        pygame.draw.circle(image, (255, 255, 255), (16, 16), 5)

        super().__init__(x, y, speed_x, speed_y, damage, field_id, image=image)
        self.explosion_radius = explosion_radius


class MagicSystem:
    """마법 발동 및 자원 소모 제어 클래스."""

    def __init__(self, scene: pygame.sprite.Sprite | any) -> None:
        self.scene = scene
        self.game_manager = GameManager()

    def cast_spell(
        self,
        spell_name: str,
        target_pos: Optional[tuple[float, float]] = None,
    ) -> bool:
        """마법을 캐스팅하고 투사체를 스폰하거나 즉시 타격 효과를 실행한다.

        Args:
            spell_name: 발동할 마법명.
            target_pos: 서브-Surface 기준 정규화 조준 좌표 (0.0~1.0). None이면 고정 방향.

        Returns:
            발동 성공 여부.
        """
        spell = SPELL_REGISTRY.get(spell_name)
        if spell is None or not spell.unlocked:
            return False

        if not self.game_manager.is_spell_ready(spell_name):
            return False
        if self.game_manager.mp < spell.mp_cost:
            return False

        self.game_manager.mp -= spell.mp_cost
        self.game_manager.start_spell_cooldown(spell_name, spell.cooldown)

        active_field = self.game_manager.active_field
        player = self.scene.player_left if active_field == 0 else self.scene.player_right
        start_pos = player.rect.center

        dir_x, dir_y = self._calc_direction(start_pos, target_pos, active_field)

        if spell_name == "piercing_bullet":
            proj = PiercingBullet(
                x=float(start_pos[0]),
                y=float(start_pos[1]),
                speed_x=spell.speed * dir_x,
                speed_y=spell.speed * dir_y,
                damage=spell.damage,
                field_id=active_field,
                pierce_limit=spell.pierce_count,
            )
            self.scene.all_sprites.add(proj)
            self.scene.projectiles.add(proj)

        elif spell_name == "fireball":
            proj = Fireball(
                x=float(start_pos[0]),
                y=float(start_pos[1]),
                speed_x=spell.speed * dir_x,
                speed_y=spell.speed * dir_y,
                damage=spell.damage,
                field_id=active_field,
                explosion_radius=spell.aoe_radius,
            )
            self.scene.all_sprites.add(proj)
            self.scene.projectiles.add(proj)

        elif spell_name == "chain_lightning":
            self._cast_chain_lightning(
                start_pos, active_field,
                chain_limit=spell.chain_count,
                damage=spell.damage,
            )

        return True

    # ── 헬퍼 ──────────────────────────────────────────────────────────────

    def _calc_direction(
        self,
        start_pos: tuple[int, int],
        target_pos: Optional[tuple[float, float]],
        active_field: int,
    ) -> tuple[float, float]:
        """정규화 조준 좌표를 방향 벡터로 변환한다."""
        if target_pos is not None:
            target_px = target_pos[0] * SCREEN_WIDTH
            target_py = target_pos[1] * _VIEW_HEIGHT
            dx = target_px - float(start_pos[0])
            dy = target_py - float(start_pos[1])
            length = math.hypot(dx, dy)
            if length >= 1.0:
                return dx / length, dy / length
        return (1.0 if active_field == 0 else -1.0), 0.0

    def _cast_chain_lightning(
        self,
        start_pos: tuple[int, int],
        field_id: int,
        chain_limit: int,
        damage: float,
    ) -> None:
        """체인 라이트닝 즉시 체이닝 타격 연산."""
        enemies = [
            sprite for sprite in self.scene.all_sprites
            if hasattr(sprite, "field_id")
            and sprite.field_id == field_id
            and sprite not in self.scene.players
            and hasattr(sprite, "take_damage")
        ]

        if not enemies:
            target_x = SCREEN_WIDTH if field_id == 0 else 0
            self.scene.add_lightning_effect([start_pos, (target_x, start_pos[1])], field_id)
            return

        chain_range = 350.0
        chain_path: list[tuple[int, int]] = [start_pos]
        current_pos = (float(start_pos[0]), float(start_pos[1]))
        hit_enemies: list = []

        for _ in range(chain_limit):
            closest_enemy = None
            min_dist = float("inf")

            for enemy in enemies:
                if enemy in hit_enemies:
                    continue
                dx = enemy.rect.centerx - current_pos[0]
                dy = enemy.rect.centery - current_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist and dist <= chain_range:
                    min_dist = dist
                    closest_enemy = enemy

            if closest_enemy:
                closest_enemy.take_damage(damage)
                hit_enemies.append(closest_enemy)
                current_pos = (float(closest_enemy.rect.centerx), float(closest_enemy.rect.centery))
                chain_path.append(closest_enemy.rect.center)
            else:
                break

        self.scene.add_lightning_effect(chain_path, field_id)
