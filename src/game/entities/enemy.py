"""적 유닛 엔티티."""

from __future__ import annotations

import pygame

from src.game.settings import SCREEN_WIDTH

from src.game.game_manager import GameManager

# 기본 적 상수
ENEMY_SIZE = 64
HP_BAR_HEIGHT = 6
BODY_HEIGHT = 48


class Enemy(pygame.sprite.Sprite):
    """적 캐릭터 클래스.

    지정된 전장에 스폰되어 플레이어 방향으로 이동하며,
    체력이 모두 닳으면 소멸하고 플레이어에게 도달하면 피해를 준다.
    """

    def __init__(
        self,
        x: float,
        y: float,
        field_id: int,
        hp: float = 50.0,
        speed: float = 120.0,
        damage: float = 10.0,
        defense_rate: float = 0.0,
    ) -> None:
        """적 유닛을 초기화한다.

        Args:
            x: 초기 X 좌표.
            y: 초기 Y 좌표.
            field_id: 소속된 전장 ID.
            hp: 적 체력.
            speed: 이동 속도.
            damage: 플레이어에게 가하는 피해량.
            defense_rate: 받는 피해 감소율 (0.0 ~ 1.0).
        """
        super().__init__()
        self.field_id = field_id
        self.max_hp = hp
        self.hp = hp
        self.speed = speed
        self.damage = damage
        self.defense_rate = defense_rate
        self.game_manager = GameManager()

        # Surface 준비 (투명 배경 지원)
        self.image = pygame.Surface((ENEMY_SIZE, ENEMY_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.center = (int(x), int(y))

        self.pos_x = float(self.rect.centerx)
        self.pos_y = float(self.rect.centery)

        # 초기 그래픽 렌더링
        self._redraw()

    def _redraw(self) -> None:
        """적의 현재 상태(체력 등)에 맞춰 이미지를 다시 그린다."""
        self.image.fill((0, 0, 0, 0))  # 투명화 클리어

        # 1. 적 바디 그리기 (붉은색 계열)
        body_y = ENEMY_SIZE - BODY_HEIGHT
        pygame.draw.rect(
            self.image,
            (231, 76, 60),  # Crimson Red
            (8, body_y, ENEMY_SIZE - 16, BODY_HEIGHT),
            border_radius=6
        )
        pygame.draw.rect(
            self.image,
            (192, 57, 43),  # Darker Red border
            (8, body_y, ENEMY_SIZE - 16, BODY_HEIGHT),
            width=2,
            border_radius=6
        )

        # 2. 체력 바 그리기 (HP 잔여량 비율 적용)
        hp_ratio = max(0.0, min(1.0, self.hp / self.max_hp))
        bar_y = 2
        # 체력 바 배경 (어두운 빨강)
        pygame.draw.rect(
            self.image,
            (120, 20, 20),
            (4, bar_y, ENEMY_SIZE - 8, HP_BAR_HEIGHT),
            border_radius=2
        )
        # 현재 체력 (초록색)
        if hp_ratio > 0:
            pygame.draw.rect(
                self.image,
                (46, 204, 113),
                (4, bar_y, int((ENEMY_SIZE - 8) * hp_ratio), HP_BAR_HEIGHT),
                border_radius=2
            )

    def update(self, dt: float) -> None:
        """이동 업데이트 및 플레이어 충전선 도달 검사.

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
        """
        # 1. 이동 제어 (비대칭 전장 구조 반영)
        # Field 0: 플레이어가 왼쪽에 있으므로 오른쪽에서 왼쪽으로 이동 (x 감소)
        # Field 1: 플레이어가 오른쪽에 있으므로 왼쪽에서 오른쪽으로 이동 (x 증가)
        if self.field_id == 0:
            self.pos_x -= self.speed * dt
            # Field 0: 오른쪽에서 왼쪽으로 이동 → 플레이어 기지선 x=200 도달 시 공격
            if self.pos_x <= 200.0:
                self._attack_player()
        elif self.field_id == 1:
            self.pos_x += self.speed * dt
            # Field 1: 왼쪽에서 오른쪽으로 이동 → 플레이어 기지선 x=SCREEN_WIDTH-200 도달 시 공격
            if self.pos_x >= float(SCREEN_WIDTH - 200):
                self._attack_player()

        # 좌표 반영
        self.rect.centerx = int(self.pos_x)
        self._redraw()

    def _attack_player(self) -> None:
        """플레이어에게 직접 데미지를 가하고 본인은 소멸한다."""
        self.game_manager.take_damage(self.damage)
        self.kill()

    def take_damage(self, amount: float) -> None:
        """피해를 입고 체력이 0이 되면 소멸한다.

        Args:
            amount: 받을 데미지 양 (방어율 적용 전).
        """
        actual = amount * (1.0 - self.defense_rate)
        self.hp = max(0.0, self.hp - actual)
        if self.hp <= 0.0:
            self.game_manager.score += 10
            self.kill()
