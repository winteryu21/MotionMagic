"""손 채널 모드 상태를 게임 이벤트로 변환하는 bridge 파이프라인."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.ai.aim_tracker import (
    AimAnchor,
    EmaAimTracker,
    tracking_landmark_ids_for_anchor,
)
from src.ai.gesture_modes import (
    DEFAULT_PINCH_CLOSE_DELTA,
    DEFAULT_PINCH_CLOSE_VELOCITY,
    DEFAULT_PINCH_CLOSED_THRESHOLD,
    DEFAULT_PINCH_OPEN_THRESHOLD,
    AimModeTracker,
    BoolDebouncer,
    FireUpdate,
    HandObservation,
    PinchFireDetector,
    SpecialGesture,
    SpecialGestureDebouncer,
    StackGestureDebouncer,
    assign_hands,
    classify_special_gesture,
    classify_stack_gesture,
    compute_finger_states,
    is_aim_pose,
    normalized_pinch_distance,
)
from src.bridge.gesture_event import GestureEvent

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_AIM_SENSITIVITY = 3.0
DEFAULT_EMA_ALPHA = 0.3


@dataclass(frozen=True)
class GestureModePipelineConfig:
    """제스처 모드 bridge 파이프라인 설정.

    Attributes:
        frame_width: 조준 좌표를 계산할 입력 프레임 너비.
        frame_height: 조준 좌표를 계산할 입력 프레임 높이.
        ema_alpha: 조준점 EMA 현재 입력 반영 비율.
        aim_sensitivity: 화면 중심 기준 조준 감도.
        aim_sensitivity_x: X축 조준 감도. ``None``이면 ``aim_sensitivity`` 사용.
        aim_sensitivity_y: Y축 조준 감도. ``None``이면 ``aim_sensitivity`` 사용.
        aim_center_x: 화면 중앙으로 매핑할 입력 X 좌표.
        aim_center_y: 화면 중앙으로 매핑할 입력 Y 좌표.
        aim_anchor: 조준점 anchor. ``"index"``는 검지 끝, ``"pinch"``는
            엄지/검지 중심을 사용한다.
        mode_stable_seconds: 조준/양손 채널 안정화 지연 시간.
        mode_grace_seconds: 짧은 추적 끊김 유예 시간.
        stack_stable_seconds: 왼손 스택 제스처 안정화 지연 시간.
        stack_cooldown_seconds: 스택 이벤트 최소 간격.
        pinch_open_threshold: 재장전으로 볼 엄지-검지 거리.
        pinch_closed_threshold: 발사로 볼 엄지-검지 닫힘 거리.
        pinch_close_velocity: 발사로 볼 닫힘 속도.
        pinch_close_delta: 발사로 볼 프레임 간 닫힘 변화량.
    """

    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    ema_alpha: float = DEFAULT_EMA_ALPHA
    aim_sensitivity: float = DEFAULT_AIM_SENSITIVITY
    aim_sensitivity_x: float | None = None
    aim_sensitivity_y: float | None = None
    aim_center_x: float = 0.5
    aim_center_y: float = 0.5
    aim_anchor: AimAnchor = "index"
    mode_stable_seconds: float = 0.15
    mode_grace_seconds: float = 0.20
    stack_stable_seconds: float = 0.18
    stack_cooldown_seconds: float = 0.35
    pinch_open_threshold: float = DEFAULT_PINCH_OPEN_THRESHOLD
    pinch_closed_threshold: float = DEFAULT_PINCH_CLOSED_THRESHOLD
    pinch_close_velocity: float = DEFAULT_PINCH_CLOSE_VELOCITY
    pinch_close_delta: float = DEFAULT_PINCH_CLOSE_DELTA


class GestureModePipeline:
    """왼손/오른손 손 채널 상태를 ``GestureEvent`` 목록으로 변환한다."""

    def __init__(self, config: GestureModePipelineConfig | None = None) -> None:
        """파이프라인을 초기화한다.

        Args:
            config: 파이프라인 설정. ``None``이면 기본값을 사용한다.
        """
        self._config = config or GestureModePipelineConfig()
        self._left_stack = StackGestureDebouncer(
            stable_seconds=self._config.stack_stable_seconds,
            grace_seconds=self._config.mode_grace_seconds,
            cooldown_seconds=self._config.stack_cooldown_seconds,
        )
        self._aim_mode = AimModeTracker(
            stable_seconds=self._config.mode_stable_seconds,
            grace_seconds=self._config.mode_grace_seconds,
        )
        self._two_hand_gate = BoolDebouncer(
            stable_seconds=self._config.mode_stable_seconds,
            grace_seconds=self._config.mode_grace_seconds,
        )
        self._special_gesture = SpecialGestureDebouncer(
            stable_seconds=self._config.mode_stable_seconds,
            grace_seconds=self._config.mode_grace_seconds,
        )
        self._aim_tracker = EmaAimTracker(
            game_width=self._config.frame_width,
            game_height=self._config.frame_height,
            alpha=self._config.ema_alpha,
            sensitivity=self._config.aim_sensitivity,
            sensitivity_x=self._config.aim_sensitivity_x,
            sensitivity_y=self._config.aim_sensitivity_y,
            center_x=self._config.aim_center_x,
            center_y=self._config.aim_center_y,
            tracking_landmark_ids=tracking_landmark_ids_for_anchor(
                self._config.aim_anchor
            ),
        )
        self._fire_detector = PinchFireDetector(
            open_threshold=self._config.pinch_open_threshold,
            closed_threshold=self._config.pinch_closed_threshold,
            close_velocity=self._config.pinch_close_velocity,
            close_delta=self._config.pinch_close_delta,
        )
        self._last_aim: tuple[float, float] | None = None
        self._two_hand_active = False

    @property
    def two_hand_active(self) -> bool:
        """양손 채널이 안정화되어 있는지 반환한다."""
        return self._two_hand_active

    def update(
        self,
        observations: list[HandObservation],
        timestamp: float | None = None,
    ) -> list[GestureEvent]:
        """한 프레임의 손 관측값을 게임 이벤트로 변환한다.

        Args:
            observations: MediaPipe에서 얻은 손 관측값 목록.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            이번 프레임에 발생한 ``GestureEvent`` 목록.
        """
        if timestamp is None:
            timestamp = time.time()

        left, right = assign_hands(observations)
        events: list[GestureEvent] = []

        special_candidate = self._special_candidate(left, right)
        special_update = self._special_gesture.update(
            special_candidate,
            timestamp=timestamp,
        )
        self._two_hand_active = self._two_hand_gate.update(
            left is not None and right is not None,
            timestamp=timestamp,
        )

        if special_update.emitted is not None:
            events.append(
                self._build_special_event(special_update.emitted, left, right)
            )
            return events

        if special_candidate is not None and special_update.stable is not None:
            events.append(self._build_special_event(special_update.stable, left, right))
            return events

        if special_candidate is not None or special_update.stable is not None:
            return events

        stack_event = self._update_left_stack(left, timestamp)
        if stack_event is not None:
            events.append(stack_event)

        aim_x: float | None = None
        aim_y: float | None = None
        pinch_distance: float | None = None
        right_score = 0.0

        if right is not None:
            right_score = right.score
            right_states = compute_finger_states(right.landmarks)
            right_candidate = is_aim_pose(right_states)
            pinch_distance = normalized_pinch_distance(right.landmarks)
        else:
            right_candidate = False

        aim_update = self._aim_mode.update(right_candidate, timestamp=timestamp)
        if right is not None and aim_update.active:
            aim_x, aim_y = self._aim_tracker.update(
                right.landmarks,
                timestamp=timestamp,
            )
            self._last_aim = (aim_x, aim_y)
            norm_x, norm_y = self._normalize_aim(aim_x, aim_y)
            events.append(
                GestureEvent(
                    gesture="aim",
                    confidence=right_score,
                    aim_x=norm_x,
                    aim_y=norm_y,
                    kind="aim",
                    channel="right",
                    active=True,
                )
            )

        fire_update = self._fire_detector.update(
            aim_active=aim_update.active,
            pinch_distance=pinch_distance,
            aim_x=aim_x,
            aim_y=aim_y,
            timestamp=timestamp,
        )
        fire_event = self._build_fire_event(fire_update, right_score)
        if fire_event is not None:
            events.append(fire_event)

        return events

    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        self._left_stack.reset()
        self._aim_mode.reset()
        self._two_hand_gate.reset()
        self._special_gesture.reset()
        self._aim_tracker.reset()
        self._fire_detector.reset()
        self._last_aim = None
        self._two_hand_active = False

    def _special_candidate(
        self,
        left: HandObservation | None,
        right: HandObservation | None,
    ) -> SpecialGesture | None:
        if left is None or right is None:
            return None
        return classify_special_gesture(left.landmarks, right.landmarks)

    def _build_special_event(
        self,
        gesture: SpecialGesture,
        left: HandObservation | None,
        right: HandObservation | None,
    ) -> GestureEvent:
        confidence = 0.0
        if left is not None and right is not None:
            confidence = min(left.score, right.score)
        aim_x, aim_y = self._last_normalized_aim()
        return GestureEvent(
            gesture=gesture,
            confidence=confidence,
            aim_x=aim_x,
            aim_y=aim_y,
            kind="special",
            channel="both",
            active=True,
        )

    def _update_left_stack(
        self,
        left: HandObservation | None,
        timestamp: float,
    ) -> GestureEvent | None:
        candidate = None
        confidence = 0.0
        if left is not None:
            confidence = left.score
            left_states = compute_finger_states(left.landmarks)
            candidate = classify_stack_gesture(left_states)

        stack_update = self._left_stack.update(candidate, timestamp=timestamp)
        if stack_update.emitted is None:
            return None

        aim_x, aim_y = self._last_normalized_aim()
        return GestureEvent(
            gesture=stack_update.emitted,
            confidence=confidence,
            aim_x=aim_x,
            aim_y=aim_y,
            kind="stack",
            channel="left",
            active=True,
        )

    def _build_fire_event(
        self,
        fire_update: FireUpdate,
        confidence: float,
    ) -> GestureEvent | None:
        if not fire_update.fired or fire_update.fire_x is None:
            return None

        fire_y = fire_update.fire_y if fire_update.fire_y is not None else 0.0
        norm_x, norm_y = self._normalize_aim(fire_update.fire_x, fire_y)
        return GestureEvent(
            gesture="fire",
            confidence=confidence,
            aim_x=norm_x,
            aim_y=norm_y,
            kind="fire",
            channel="right",
            active=True,
        )

    def _last_normalized_aim(self) -> tuple[float, float]:
        if self._last_aim is None:
            return 0.5, 0.5
        return self._normalize_aim(self._last_aim[0], self._last_aim[1])

    def _normalize_aim(self, aim_x: float, aim_y: float) -> tuple[float, float]:
        norm_x = aim_x / max(float(self._config.frame_width), 1.0)
        norm_y = aim_y / max(float(self._config.frame_height), 1.0)
        return max(0.0, min(norm_x, 1.0)), max(0.0, min(norm_y, 1.0))
