"""전투 시스템 — 투사체 이동 후 적 충돌 판정만 담당."""

from __future__ import annotations

import math

from src.game.entities.enemy import Enemy
from src.game.entities.projectile import Explosion, Fireball, MagicMissile, Projectile


class CombatSystem:
    """원본 BattleField 리스트 구조에 맞춘 충돌 처리기."""

    @staticmethod
    def update_projectile_collisions(field) -> None:
        """field.projectiles와 field.enemies 사이 충돌을 검사한다.

        - MagicMissile: 관통 마탄. 이미 맞은 적은 다시 맞지 않음.
        - Fireball: 첫 충돌 시 Explosion effect 생성 후 소멸.
        - Projectile: 첫 충돌 적에게 피해 후 소멸.
        """
        for projectile in list(field.projectiles):
            if not projectile.alive:
                continue
            for enemy in list(field.enemies):
                if not enemy.alive:
                    continue
                if not CombatSystem._collides(projectile, enemy):
                    continue

                if isinstance(projectile, Fireball):
                    field.effects.append(
                        Explosion(
                            x=projectile.x,
                            y=projectile.y,
                            damage=projectile.damage,
                            radius=projectile.explosion_radius,
                        )
                    )
                    projectile.alive = False
                    break

                if isinstance(projectile, MagicMissile):
                    if projectile.can_hit(enemy):
                        enemy.take_damage(projectile.damage)
                        projectile.register_hit(enemy)
                    if not projectile.alive:
                        break
                    continue

                enemy.take_damage(projectile.damage)
                projectile.alive = False
                break

    @staticmethod
    def _collides(projectile: Projectile, enemy: Enemy) -> bool:
        hit_radius = projectile.radius + enemy.size / 2
        return math.hypot(projectile.x - enemy.x, projectile.y - enemy.y) <= hit_radius
