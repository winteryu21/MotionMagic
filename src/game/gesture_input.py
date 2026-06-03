"""제스처 좌표를 게임 입력 좌표로 변환하는 유틸리티."""

from __future__ import annotations

from src.bridge.gesture_event import GestureEvent
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH


def screen_pos_from_gesture_event(event: GestureEvent) -> tuple[int, int]:
    """정규화된 제스처 좌표를 현재 게임 화면 좌표로 변환한다.

    Args:
        event: bridge 계층에서 전달된 제스처 이벤트.

    Returns:
        화면 영역 안으로 clamp된 ``(x, y)`` 좌표.
    """
    x = round(event.aim_x * SCREEN_WIDTH)
    y = round(event.aim_y * SCREEN_HEIGHT)
    return (
        max(0, min(SCREEN_WIDTH - 1, x)),
        max(0, min(SCREEN_HEIGHT - 1, y)),
    )
