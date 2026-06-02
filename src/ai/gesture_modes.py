"""왼손/오른손/양손 제스처 모드 상태 머신."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

HandLabel = Literal["Left", "Right"]
StackGesture = Literal["rock", "paper", "scissors"]

THUMB = 0
INDEX = 1
MIDDLE = 2
RING = 3
PINKY = 4

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
WRIST = 0
MIDDLE_MCP = 9

DEFAULT_MODE_STABLE_SECONDS = 0.15
DEFAULT_MODE_GRACE_SECONDS = 0.20
DEFAULT_STACK_STABLE_SECONDS = 0.18
DEFAULT_STACK_COOLDOWN_SECONDS = 0.35
DEFAULT_PINCH_OPEN_THRESHOLD = 0.70
DEFAULT_PINCH_CLOSED_THRESHOLD = 0.55
DEFAULT_PINCH_CLOSE_VELOCITY = 0.60
DEFAULT_PINCH_CLOSE_DELTA = 0.12
DEFAULT_FIRE_COOLDOWN_SECONDS = 0.30
DEFAULT_AIM_HISTORY_SECONDS = 0.45
DEFAULT_PRE_FIRE_SECONDS = 0.12


@dataclass(frozen=True)
class HandObservation:
    """한 손의 MediaPipe 관측값.

    Attributes:
        label: MediaPipe handedness 라벨.
        landmarks: ``(21, 3)`` 원시 랜드마크 좌표.
        score: handedness 또는 detection 신뢰도.
    """

    label: HandLabel
    landmarks: np.ndarray
    score: float = 1.0


@dataclass(frozen=True)
class StackUpdate:
    """왼손 스택 모드 갱신 결과.

    Attributes:
        candidate: 이번 프레임 후보 제스처.
        stable: 안정화된 제스처.
        emitted: 이번 프레임에 스택에 추가할 제스처.
    """

    candidate: StackGesture | None
    stable: StackGesture | None
    emitted: StackGesture | None


@dataclass(frozen=True)
class AimUpdate:
    """오른손 조준 모드 갱신 결과.

    Attributes:
        candidate: 이번 프레임이 조준 후보인지 여부.
        active: 안정화/유예를 거친 조준 활성 상태.
    """

    candidate: bool
    active: bool


@dataclass(frozen=True)
class FireUpdate:
    """오른손 핀치 발사 갱신 결과.

    Attributes:
        fired: 이번 프레임에 발사됐는지 여부.
        fire_x: 발사 시 고정된 X 좌표.
        fire_y: 발사 시 고정된 Y 좌표.
        pinch_distance: 손 크기 기준 엄지-검지 거리.
        pinch_velocity: 엄지-검지 거리 변화 속도. 닫히면 음수.
        armed: 다시 발사 가능한 상태인지 여부.
    """

    fired: bool
    fire_x: float | None
    fire_y: float | None
    pinch_distance: float
    pinch_velocity: float
    armed: bool


def compute_finger_states(landmarks: np.ndarray) -> np.ndarray:
    """손가락 펼침 상태를 원시 랜드마크에서 계산한다.

    엄지는 검지 MCP와의 상대 거리, 나머지 손가락은 손목 기준
    tip/PIP 거리 비교를 사용한다.

    Args:
        landmarks: ``(21, 3)`` 원시 랜드마크 좌표.

    Returns:
        ``[엄지, 검지, 중지, 약지, 새끼]`` 펼침 상태.
    """
    wrist = landmarks[WRIST]
    states = []

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]
    thumb_tip_dist = np.linalg.norm(thumb_tip - index_mcp)
    thumb_ip_dist = np.linalg.norm(thumb_ip - index_mcp)
    states.append(1.0 if thumb_tip_dist > thumb_ip_dist * 1.6 else 0.0)

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        tip_dist = np.linalg.norm(landmarks[tip] - wrist)
        pip_dist = np.linalg.norm(landmarks[pip] - wrist)
        states.append(1.0 if tip_dist > pip_dist else 0.0)

    return np.array(states, dtype=np.float32)


def classify_stack_gesture(finger_states: np.ndarray) -> StackGesture | None:
    """왼손 스택용 손가락 상태를 제스처로 변환한다.

    엄지는 시점 변화에 약하므로 스택 제스처 판정에서 제외한다.

    Args:
        finger_states: ``(5,)`` 손가락 펼침 상태.

    Returns:
        ``rock``, ``paper``, ``scissors`` 또는 ``None``.
    """
    index = int(finger_states[INDEX])
    middle = int(finger_states[MIDDLE])
    ring = int(finger_states[RING])
    pinky = int(finger_states[PINKY])
    fingers = (index, middle, ring, pinky)

    if fingers == (0, 0, 0, 0):
        return "rock"
    if fingers == (1, 1, 1, 1):
        return "paper"
    if fingers == (1, 1, 0, 0):
        return "scissors"
    return None


def is_aim_pose(finger_states: np.ndarray) -> bool:
    """오른손 조준 포즈 여부를 판정한다.

    검지가 펴지고 약지/새끼가 접힌 상태를 조준 후보로 본다.
    엄지와 중지는 정면 카메라에서 펼침 판정이 흔들리기 쉬워
    조준 활성 조건에서는 제외한다. 실제 조준점은 계속 엄지/검지
    끝 중심점을 사용한다.

    Args:
        finger_states: ``(5,)`` 손가락 펼침 상태.

    Returns:
        조준 포즈 여부.
    """
    return bool(
        finger_states[INDEX] == 1.0
        and finger_states[RING] == 0.0
        and finger_states[PINKY] == 0.0
    )


def aim_point(landmarks: np.ndarray) -> tuple[float, float]:
    """엄지/검지 끝 중심점을 반환한다.

    Args:
        landmarks: ``(21, 3)`` 원시 랜드마크 좌표.

    Returns:
        MediaPipe 정규화 좌표계의 ``(x, y)``.
    """
    center = (landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) * 0.5
    return float(center[0]), float(center[1])


def normalized_pinch_distance(landmarks: np.ndarray) -> float:
    """손 크기 기준 엄지-검지 끝 거리.

    Args:
        landmarks: ``(21, 3)`` 원시 랜드마크 좌표.

    Returns:
        손목-중지 MCP 거리에 대한 엄지-검지 거리 비율.
    """
    hand_scale = float(np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_MCP]))
    if hand_scale < 1e-6:
        return 0.0

    pinch = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]))
    return pinch / hand_scale


def assign_hands(
    observations: list[HandObservation],
) -> tuple[HandObservation | None, HandObservation | None]:
    """관측 손 목록을 왼손/오른손 채널로 분리한다.

    같은 handedness가 여러 개면 신뢰도가 높은 손을 사용한다.

    Args:
        observations: 손 관측 목록.

    Returns:
        ``(left, right)`` 튜플.
    """
    left: HandObservation | None = None
    right: HandObservation | None = None

    for observation in sorted(observations, key=lambda item: item.score, reverse=True):
        if observation.label == "Left" and left is None:
            left = observation
        elif observation.label == "Right" and right is None:
            right = observation

    return left, right


class BoolDebouncer:
    """bool 후보 값을 안정화하고 짧은 끊김에는 유예를 둔다."""

    def __init__(
        self,
        stable_seconds: float = DEFAULT_MODE_STABLE_SECONDS,
        grace_seconds: float = DEFAULT_MODE_GRACE_SECONDS,
    ) -> None:
        self._stable_seconds = stable_seconds
        self._grace_seconds = grace_seconds
        self._candidate_started_at: float | None = None
        self._last_seen_at: float | None = None
        self._active = False

    def update(self, candidate: bool, timestamp: float | None = None) -> bool:
        """후보 값을 갱신하고 활성 상태를 반환한다.

        Args:
            candidate: 이번 프레임 후보 여부.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            안정화/유예를 거친 활성 상태.
        """
        if timestamp is None:
            timestamp = time.time()

        if candidate:
            self._last_seen_at = timestamp
            if self._candidate_started_at is None:
                self._candidate_started_at = timestamp
            if timestamp - self._candidate_started_at >= self._stable_seconds:
                self._active = True
        else:
            self._candidate_started_at = None
            if self._last_seen_at is None:
                self._active = False
            elif timestamp - self._last_seen_at > self._grace_seconds:
                self._active = False

        return self._active

    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        self._candidate_started_at = None
        self._last_seen_at = None
        self._active = False


class StackGestureDebouncer:
    """왼손 스택 제스처를 안정화하고 1회 이벤트로 변환한다."""

    def __init__(
        self,
        stable_seconds: float = DEFAULT_STACK_STABLE_SECONDS,
        grace_seconds: float = DEFAULT_MODE_GRACE_SECONDS,
        cooldown_seconds: float = DEFAULT_STACK_COOLDOWN_SECONDS,
    ) -> None:
        self._stable_seconds = stable_seconds
        self._grace_seconds = grace_seconds
        self._cooldown_seconds = cooldown_seconds

        self._candidate: StackGesture | None = None
        self._candidate_started_at: float | None = None
        self._last_seen_at: float | None = None
        self._stable: StackGesture | None = None
        self._last_emitted: StackGesture | None = None
        self._last_emitted_at: float | None = None

    def update(
        self,
        candidate: StackGesture | None,
        timestamp: float | None = None,
    ) -> StackUpdate:
        """스택 제스처 후보를 갱신한다.

        Args:
            candidate: 이번 프레임 후보 제스처.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            안정화/이벤트 결과.
        """
        if timestamp is None:
            timestamp = time.time()

        if candidate is None:
            self._candidate = None
            self._candidate_started_at = None
            if self._last_seen_at is None:
                self._stable = None
            elif timestamp - self._last_seen_at > self._grace_seconds:
                self._stable = None
                self._last_emitted = None
            return StackUpdate(candidate, self._stable, None)

        self._last_seen_at = timestamp
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_started_at = timestamp
            return StackUpdate(candidate, self._stable, None)

        if self._candidate_started_at is None:
            self._candidate_started_at = timestamp

        emitted = None
        if timestamp - self._candidate_started_at >= self._stable_seconds:
            self._stable = candidate
            if self._can_emit(candidate, timestamp):
                emitted = candidate
                self._last_emitted = candidate
                self._last_emitted_at = timestamp

        return StackUpdate(candidate, self._stable, emitted)

    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        self._candidate = None
        self._candidate_started_at = None
        self._last_seen_at = None
        self._stable = None
        self._last_emitted = None
        self._last_emitted_at = None

    def _can_emit(self, candidate: StackGesture, timestamp: float) -> bool:
        if candidate == self._last_emitted:
            return False
        if self._last_emitted_at is None:
            return True
        return timestamp - self._last_emitted_at >= self._cooldown_seconds


class AimModeTracker:
    """오른손 조준 모드 후보를 안정화한다."""

    def __init__(
        self,
        stable_seconds: float = DEFAULT_MODE_STABLE_SECONDS,
        grace_seconds: float = DEFAULT_MODE_GRACE_SECONDS,
    ) -> None:
        self._debouncer = BoolDebouncer(stable_seconds, grace_seconds)

    def update(
        self,
        candidate: bool,
        timestamp: float | None = None,
    ) -> AimUpdate:
        """조준 후보를 갱신한다.

        Args:
            candidate: 이번 프레임 조준 후보 여부.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            조준 모드 갱신 결과.
        """
        active = self._debouncer.update(candidate, timestamp)
        return AimUpdate(candidate, active)

    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        self._debouncer.reset()


class PinchFireDetector:
    """오른손 엄지-검지 핀치 발사 상태 머신.

    Args:
        open_threshold: 다시 발사 가능 상태로 볼 엄지-검지 거리.
        closed_threshold: 발사 후보로 볼 엄지-검지 닫힘 거리.
        close_velocity: 발사 후보로 볼 닫힘 속도.
        close_delta: 발사 후보로 볼 프레임 간 닫힘 변화량.
        cooldown_seconds: 발사 후 최소 대기 시간.
        aim_history_seconds: 발사 좌표를 되돌아볼 조준 기록 유지 시간.
        pre_fire_seconds: 발사 순간보다 몇 초 전 조준점을 사용할지.
    """

    def __init__(
        self,
        open_threshold: float = DEFAULT_PINCH_OPEN_THRESHOLD,
        closed_threshold: float = DEFAULT_PINCH_CLOSED_THRESHOLD,
        close_velocity: float = DEFAULT_PINCH_CLOSE_VELOCITY,
        close_delta: float = DEFAULT_PINCH_CLOSE_DELTA,
        cooldown_seconds: float = DEFAULT_FIRE_COOLDOWN_SECONDS,
        aim_history_seconds: float = DEFAULT_AIM_HISTORY_SECONDS,
        pre_fire_seconds: float = DEFAULT_PRE_FIRE_SECONDS,
    ) -> None:
        self._open_threshold = open_threshold
        self._closed_threshold = closed_threshold
        self._close_velocity = close_velocity
        self._close_delta = close_delta
        self._cooldown_seconds = cooldown_seconds
        self._aim_history_seconds = aim_history_seconds
        self._pre_fire_seconds = pre_fire_seconds

        self._aim_history: deque[tuple[float, float, float]] = deque()
        self._last_distance: float | None = None
        self._last_distance_at: float | None = None
        self._last_fire_at: float | None = None
        self._armed = True

    def update(
        self,
        aim_active: bool,
        pinch_distance: float | None,
        aim_x: float | None,
        aim_y: float | None,
        timestamp: float | None = None,
    ) -> FireUpdate:
        """핀치 발사 상태를 갱신한다.

        Args:
            aim_active: 조준 모드 활성 여부.
            pinch_distance: 손 크기 기준 엄지-검지 거리.
            aim_x: 현재 조준 X 좌표.
            aim_y: 현재 조준 Y 좌표.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            핀치 발사 갱신 결과.
        """
        if timestamp is None:
            timestamp = time.time()

        if aim_active and aim_x is not None and aim_y is not None:
            self._append_aim(timestamp, aim_x, aim_y)

        if not aim_active or pinch_distance is None:
            self._last_distance = None
            self._last_distance_at = None
            display_distance = 0.0 if pinch_distance is None else pinch_distance
            return FireUpdate(False, None, None, display_distance, 0.0, self._armed)

        previous_distance = self._last_distance
        velocity = self._pinch_velocity(pinch_distance, timestamp)
        distance_delta = 0.0
        if previous_distance is not None:
            distance_delta = previous_distance - pinch_distance

        self._last_distance = pinch_distance
        self._last_distance_at = timestamp

        if pinch_distance >= self._open_threshold:
            self._armed = True

        fired = self._should_fire(
            pinch_distance,
            velocity,
            distance_delta,
            timestamp,
        )
        if not fired:
            return FireUpdate(False, None, None, pinch_distance, velocity, self._armed)

        fire_x, fire_y = self._aim_before(timestamp - self._pre_fire_seconds)
        self._armed = False
        self._last_fire_at = timestamp
        return FireUpdate(True, fire_x, fire_y, pinch_distance, velocity, self._armed)

    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        self._aim_history.clear()
        self._last_distance = None
        self._last_distance_at = None
        self._last_fire_at = None
        self._armed = True

    def _append_aim(self, timestamp: float, aim_x: float, aim_y: float) -> None:
        self._aim_history.append((timestamp, aim_x, aim_y))
        min_time = timestamp - self._aim_history_seconds
        while self._aim_history and self._aim_history[0][0] < min_time:
            self._aim_history.popleft()

    def _pinch_velocity(self, pinch_distance: float, timestamp: float) -> float:
        if self._last_distance is None or self._last_distance_at is None:
            return 0.0

        elapsed = max(timestamp - self._last_distance_at, 1e-6)
        return (pinch_distance - self._last_distance) / elapsed

    def _should_fire(
        self,
        pinch_distance: float,
        velocity: float,
        distance_delta: float,
        timestamp: float,
    ) -> bool:
        if not self._armed:
            return False
        if pinch_distance > self._closed_threshold:
            return False
        closes_fast = velocity <= -self._close_velocity
        closes_far_enough = distance_delta >= self._close_delta
        if not closes_fast and not closes_far_enough:
            return False
        if self._last_fire_at is None:
            return True
        return timestamp - self._last_fire_at >= self._cooldown_seconds

    def _aim_before(self, target_time: float) -> tuple[float | None, float | None]:
        if not self._aim_history:
            return None, None

        selected = self._aim_history[0]
        for item in self._aim_history:
            if item[0] > target_time:
                break
            selected = item
        return selected[1], selected[2]
