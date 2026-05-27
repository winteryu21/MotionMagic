"""플레이어 캐릭터 관련 기능 테스트."""

from __future__ import annotations

import pygame

from src.game.entities.player import Player
from src.game.settings import PLAYER_SIZE


def test_player_initialization() -> None:
    """플레이어가 올바른 초기 위치와 크기로 생성되는지 테스트한다."""
    player = Player(150.0, 400.0)
    assert player.rect.x == 150
    assert player.rect.y == 400
    assert player.rect.width == PLAYER_SIZE
    assert player.rect.height == PLAYER_SIZE


def test_player_remains_still() -> None:
    """플레이어가 update 호출 이후에도 위치가 변경되지 않는지 테스트한다."""
    player = Player(150.0, 400.0)
    player.update(0.016)  # dt = 0.016 (약 60 FPS)
    assert player.rect.x == 150
    assert player.rect.y == 400


def test_player_draw() -> None:
    """플레이어가 Surface에 정상적으로 그려지는지 테스트한다."""
    # pygame display 초기화 없이도 Surface는 생성하여 그릴 수 있습니다.
    surface = pygame.Surface((300, 300))
    player = Player(10.0, 10.0)
    player.draw(surface)

    # 지정된 영역이 하얀색(255, 255, 255)으로 채워졌는지 픽셀 색상으로 확인합니다.
    pixel_color = surface.get_at((15, 15))
    assert pixel_color == (255, 255, 255, 255)
