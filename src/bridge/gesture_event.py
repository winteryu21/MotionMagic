"""제스처 → 게임 이벤트 변환."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GestureLabel = Literal[
    "rock",
    "paper",
    "scissors",
    "trigger",
    "clasp",
    "sonaldo",
    "aim",
    "fire",
]
GestureEventKind = Literal["stack", "aim", "fire", "special"]
GestureChannel = Literal["left", "right", "both"]


@dataclass(frozen=True)
class GestureEvent:
    """AI 시스템이 인식한 제스처를 게임에 전달하는 이벤트.

    Attributes:
        gesture: 인식된 제스처 또는 액션 라벨.
        confidence: 예측 또는 handedness 확신도 (0.0 ~ 1.0).
        aim_x: 에임 X 좌표 (화면 기준, 0.0 ~ 1.0).
        aim_y: 에임 Y 좌표 (화면 기준, 0.0 ~ 1.0).
        kind: 스택, 조준, 발사, 특수 모드 중 이벤트 종류.
        channel: 왼손, 오른손, 양손 중 이벤트 입력 채널.
        active: 해당 채널이 활성 상태인지 여부.
    """

    gesture: GestureLabel
    confidence: float
    aim_x: float
    aim_y: float
    kind: GestureEventKind = "stack"
    channel: GestureChannel = "left"
    active: bool = True
