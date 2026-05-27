"""플레이어 캐릭터 엔티티."""

from __future__ import annotations

import pygame

from src.game.settings import TILE_SIZE


class Player(pygame.sprite.Sprite):
    """플레이어 캐릭터 클래스.

    이 캐릭터는 움직이지 않으며, 화면 내 지정된 위치(예: 왼쪽 혹은 오른쪽)에 고정되어 존재한다.
    하얀색 사각형 형태를 띤다.
    """

    def __init__(self, x: float, y: float, field_id: int) -> None:
        """플레이어 캐릭터를 초기화한다.

        Args:
            x: 초기 X 좌표 (중앙 기준).
            y: 초기 Y 좌표 (중앙 기준).
            field_id: 소속된 전장 ID.
        """
        super().__init__()
        self.field_id = field_id

        # 하얀색 사각형 외관 설정
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self.image.fill((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.center = (int(x), int(y))

    def update(self, dt: float) -> None:
        """프레임 업데이트. 플레이어는 고정되어 있어 아무 동작도 하지 않는다.

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
        """
        pass
