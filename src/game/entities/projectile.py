"""투사체 엔티티."""

from __future__ import annotations

import pygame

from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH


class Projectile(pygame.sprite.Sprite):
    """투사체 기본 클래스.

    마법 투사체의 공통 동작(속도에 따른 이동, 화면 이탈 시 자동 소멸 등)을 관리한다.
    """

    def __init__(
        self,
        x: float,
        y: float,
        speed_x: float,
        speed_y: float,
        damage: float,
        field_id: int,
        image: pygame.Surface | None = None
    ) -> None:
        """투사체를 초기화한다.

        Args:
            x: 초기 X 좌표.
            y: 초기 Y 좌표.
            speed_x: X축 이동 속도 (초당 픽셀).
            speed_y: Y축 이동 속도 (초당 픽셀).
            damage: 가하는 피해량.
            field_id: 소속된 전장 ID.
            image: 커스텀 투사체 이미지. None일 경우 기본 노란색 구체 생성.
        """
        super().__init__()
        self.field_id = field_id
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.damage = damage

        # 이미지 설정
        if image is not None:
            self.image = image
        else:
            # 기본 모양: 16x16 크기의 노란색 빛나는 원
            self.image = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(self.image, (241, 196, 15), (8, 8), 8)  # 황금빛 구체
            pygame.draw.circle(self.image, (255, 255, 255), (8, 8), 5)  # 중앙 코어

        self.rect = self.image.get_rect()
        self.rect.center = (int(x), int(y))

        # 정밀한 소수점 물리 좌표 연산을 위해 실수형 위치 추적
        self.pos_x = float(self.rect.centerx)
        self.pos_y = float(self.rect.centery)

    def update(self, dt: float) -> None:
        """프레임 업데이트. 위치를 갱신하고 화면 이탈 여부를 검사한다.

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
        """
        # 1. 위치 이동
        self.pos_x += self.speed_x * dt
        self.pos_y += self.speed_y * dt

        # rect 반영
        self.rect.centerx = int(self.pos_x)
        self.rect.centery = int(self.pos_y)

        # 2. 화면 이탈 검사 (전체 화면 기준)
        if (self.pos_x < -32.0 or
                self.pos_x > float(SCREEN_WIDTH + 32) or
                self.pos_y < -32.0 or
                self.pos_y > float(SCREEN_HEIGHT + 32)):
            self.kill()
