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
