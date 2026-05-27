"""제스처 → 게임 이벤트 변환."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GestureLabel = Literal["fist", "palm", "point", "scissors"]


@dataclass(frozen=True)
class GestureEvent:
    """AI 시스템이 인식한 제스처를 게임에 전달하는 이벤트.

    Attributes:
        gesture: 인식된 제스처 라벨.
        confidence: CNN 모델의 예측 확신도 (0.0 ~ 1.0).
        aim_x: 에임 X 좌표 (화면 기준, 0.0 ~ 1.0).
        aim_y: 에임 Y 좌표 (화면 기준, 0.0 ~ 1.0).
    """

    gesture: GestureLabel
    confidence: float
    aim_x: float
    aim_y: float


# Pygame 사용자 정의 이벤트 타입 정의
import pygame

GESTURE_EVENT = pygame.USEREVENT + 1


def post_gesture_event(gesture_event: GestureEvent) -> None:
    """인식된 제스처 이벤트를 Pygame의 이벤트 큐에 게시한다.

    이 함수는 백그라운드 스레드에서 안전하게 호출하여 메인 스레드의 Pygame 루프로
    이벤트를 전달하는 데 사용된다.

    Args:
        gesture_event: 게시할 GestureEvent 객체.
    """
    if pygame.get_init():
        event = pygame.event.Event(GESTURE_EVENT, {"gesture_event": gesture_event})
        pygame.event.post(event)
