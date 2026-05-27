"""전투 시스템 — 제스처 마법의 이중 구조 및 충돌 판정."""

from __future__ import annotations

import math
import pygame

from src.game.systems.magic import Fireball, PiercingBullet


class CombatSystem:
    """게임 내 투사체와 적 간의 충돌 및 데미지 처리를 전담하는 클래스."""

    def __init__(self, scene: pygame.sprite.Sprite | any) -> None:
        """전투 시스템을 초기화한다.

        Args:
            scene: 전투 씬(BattleScene) 객체 참조.
        """
        self.scene = scene

    def update(self) -> None:
        """투사체와 적 간의 충돌 검사를 수행하고 피해 및 소멸을 처리한다."""
        # 1. 투사체 그룹과 적 그룹 간의 AABB 사각형 충돌 검사
        collisions = pygame.sprite.groupcollide(
            self.scene.projectiles,
            self.scene.enemies,
            False,
            False
        )

        for proj, hit_list in collisions.items():
            # 동일한 전장(field_id)에 속한 경우만 유효 충돌로 필터링
            valid_hits = [enemy for enemy in hit_list if enemy.field_id == proj.field_id]
            if not valid_hits:
                continue

            if isinstance(proj, Fireball):
                # 화염구: 어떤 적이든 접촉 시 즉시 폭발(AoE 피해)하고 소멸
                self._explode_fireball(proj)
                proj.kill()

            elif isinstance(proj, PiercingBullet):
                # 관통 마탄: 이미 타격한 적은 건너뛰며, 관통 제한 횟수까지 타격 후 소멸
                for enemy in valid_hits:
                    if proj.can_hit(enemy):
                        enemy.take_damage(proj.damage)
                        proj.hit_enemies.add(enemy)
                        
                        # 관통 한도 도달 시 소멸
                        if len(proj.hit_enemies) >= proj.pierce_limit:
                            proj.kill()
                            break

            else:
                # 기본 투사체: 접촉한 첫 번째 적에게 데미지를 가하고 즉시 소멸
                first_enemy = valid_hits[0]
                first_enemy.take_damage(proj.damage)
                proj.kill()

    def _explode_fireball(self, fireball: Fireball) -> None:
        """화염구 폭발 중심 좌표 기준으로 범위 내의 모든 적에게 광역 데미지를 가한다.

        Args:
            fireball: 폭발한 Fireball 인스턴스.
        """
        center_x = float(fireball.rect.centerx)
        center_y = float(fireball.rect.centery)
        radius = fireball.explosion_radius
        field_id = fireball.field_id

        # 해당 전장의 적들 중 폭발 반경 이내인 경우 데미지 적용
        for enemy in self.scene.enemies:
            if enemy.field_id == field_id:
                dx = enemy.rect.centerx - center_x
                dy = enemy.rect.centery - center_y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance <= radius:
                    enemy.take_damage(fireball.damage)
