"""트리거 제스처 기반 사격 판정 상태 머신."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

DEFAULT_AIM_HISTORY_SECONDS = 0.45
DEFAULT_PRE_FIRE_SECONDS = 0.10
DEFAULT_FLICK_WINDOW_SECONDS = 0.18
DEFAULT_MIN_TRIGGER_HOLD_SECONDS = 0.12
DEFAULT_MIN_FLICK_DISTANCE = 0.045
DEFAULT_MIN_UPWARD_VELOCITY = 0.85
DEFAULT_MIN_FRAME_UPWARD_VELOCITY = 0.012
DEFAULT_MIN_FRAME_UPWARD_ACCELERATION = 0.008
DEFAULT_REARM_DOWNWARD_VELOCITY = 0.20
DEFAULT_COOLDOWN_SECONDS = 0.35


@dataclass(frozen=True)
class ShotUpdate:
    """사격 판정 갱신 결과.

    Attributes:
        fired: 이번 프레임에 발사됐는지 여부.
        aim_x: 현재 EMA 조준 X 좌표.
        aim_y: 현재 EMA 조준 Y 좌표.
        fire_x: 발사 시 고정된 X 좌표. 발사하지 않으면 ``None``.
        fire_y: 발사 시 고정된 Y 좌표. 발사하지 않으면 ``None``.
        wrist_velocity_y: 손목 Y축 속도. 위로 튕기면 음수.
        upward_velocity: 프레임 기반 위쪽 속도. 위로 튕기면 양수.
        upward_acceleration: 프레임 기반 위쪽 가속도.
        is_trigger: 현재 trigger 상태인지 여부.
    """

    fired: bool
    aim_x: float
    aim_y: float
    fire_x: float | None
    fire_y: float | None
    wrist_velocity_y: float
    is_trigger: bool
    upward_velocity: float = 0.0
    upward_acceleration: float = 0.0


class TriggerShotDetector:
    """trigger 상태에서 손목 위쪽 튕김만 발사로 판정한다.

    MediaPipe 화면 좌표계에서 Y는 아래로 갈수록 커진다. 따라서 손목을
    위로 튕기면 Y값이 빠르게 감소하며, 이 클래스는 trigger 유지 시간,
    위쪽 이동량, 위쪽 속도, 재장전 상태를 함께 확인한다.

    Args:
        aim_history_seconds: 발사 좌표를 되돌아볼 조준 기록 유지 시간.
        pre_fire_seconds: 발사 순간보다 몇 초 전 조준점을 사용할지.
        flick_window_seconds: 손목 튕김을 평가할 시간 창.
        min_trigger_hold_seconds: 발사 전 trigger가 유지되어야 하는 시간.
        min_flick_distance: 위쪽 튕김 최소 이동량.
        min_upward_velocity: 위쪽 튕김 최소 속도.
        min_frame_upward_velocity: 프레임 기반 위쪽 속도 임계값.
        min_frame_upward_acceleration: 프레임 기반 위쪽 가속도 임계값.
        rearm_downward_velocity: 재장전으로 인정할 아래쪽 이동 속도.
        cooldown_seconds: 발사 후 최소 대기 시간.
    """

    def __init__(
        self,
        aim_history_seconds: float = DEFAULT_AIM_HISTORY_SECONDS,
        pre_fire_seconds: float = DEFAULT_PRE_FIRE_SECONDS,
        flick_window_seconds: float = DEFAULT_FLICK_WINDOW_SECONDS,
        min_trigger_hold_seconds: float = DEFAULT_MIN_TRIGGER_HOLD_SECONDS,
        min_flick_distance: float = DEFAULT_MIN_FLICK_DISTANCE,
        min_upward_velocity: float = DEFAULT_MIN_UPWARD_VELOCITY,
        min_frame_upward_velocity: float = DEFAULT_MIN_FRAME_UPWARD_VELOCITY,
        min_frame_upward_acceleration: float = DEFAULT_MIN_FRAME_UPWARD_ACCELERATION,
        rearm_downward_velocity: float = DEFAULT_REARM_DOWNWARD_VELOCITY,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._aim_history_seconds = aim_history_seconds
        self._pre_fire_seconds = pre_fire_seconds
        self._flick_window_seconds = flick_window_seconds
        self._min_trigger_hold_seconds = min_trigger_hold_seconds
        self._min_flick_distance = min_flick_distance
        self._min_upward_velocity = min_upward_velocity
        self._min_frame_upward_velocity = min_frame_upward_velocity
        self._min_frame_upward_acceleration = min_frame_upward_acceleration
        self._rearm_downward_velocity = rearm_downward_velocity
        self._cooldown_seconds = cooldown_seconds

        self._aim_history: deque[tuple[float, float, float]] = deque()
        self._wrist_history: deque[tuple[float, float]] = deque()
        self._trigger_started_at: float | None = None
        self._last_shot_at: float | None = None
        self._armed = True

    def update(
        self,
        is_trigger: bool,
        wrist_y: float,
        aim_x: float,
        aim_y: float,
        timestamp: float | None = None,
    ) -> ShotUpdate:
        """사격 상태를 한 프레임 갱신한다.

        Args:
            is_trigger: 현재 손 모양이 trigger인지 여부.
            wrist_y: MediaPipe 원시 손목 Y 좌표.
            aim_x: 현재 EMA 조준 X 좌표.
            aim_y: 현재 EMA 조준 Y 좌표.
            timestamp: 현재 시간. ``None``이면 자동 측정.

        Returns:
            이번 프레임 사격 판정 결과.
        """
        if timestamp is None:
            timestamp = time.time()

        self._append_aim(timestamp, aim_x, aim_y)

        if not is_trigger:
            self._reset_trigger_state()
            return ShotUpdate(False, aim_x, aim_y, None, None, 0.0, False)

        if self._trigger_started_at is None:
            self._trigger_started_at = timestamp
            self._wrist_history.clear()
            self._armed = True

        self._append_wrist(timestamp, wrist_y)
        wrist_velocity_y, upward_distance = self._measure_upward_flick(
            timestamp,
            wrist_y,
        )
        upward_velocity, upward_acceleration = self._measure_frame_flick()

        if not self._armed:
            if wrist_velocity_y >= self._rearm_downward_velocity:
                self._armed = True
                self._wrist_history.clear()
                self._append_wrist(timestamp, wrist_y)
            return ShotUpdate(
                False,
                aim_x,
                aim_y,
                None,
                None,
                wrist_velocity_y,
                True,
                upward_velocity,
                upward_acceleration,
            )

        fired = self._should_fire(
            timestamp,
            wrist_velocity_y,
            upward_distance,
            upward_velocity,
            upward_acceleration,
        )
        if not fired:
            return ShotUpdate(
                False,
                aim_x,
                aim_y,
                None,
                None,
                wrist_velocity_y,
                True,
                upward_velocity,
                upward_acceleration,
            )

        fire_x, fire_y = self._aim_before(timestamp - self._pre_fire_seconds)
        self._last_shot_at = timestamp
        self._armed = False
        return ShotUpdate(
            True,
            aim_x,
            aim_y,
            fire_x,
            fire_y,
            wrist_velocity_y,
            True,
            upward_velocity,
            upward_acceleration,
        )

    def reset(self) -> None:
        """모든 내부 상태를 초기화한다."""
        self._aim_history.clear()
        self._reset_trigger_state()
        self._last_shot_at = None

    def _append_aim(self, timestamp: float, aim_x: float, aim_y: float) -> None:
        self._aim_history.append((timestamp, aim_x, aim_y))
        self._prune_aim_history(timestamp)

    def _append_wrist(self, timestamp: float, wrist_y: float) -> None:
        self._wrist_history.append((timestamp, wrist_y))
        min_time = timestamp - self._flick_window_seconds
        while self._wrist_history and self._wrist_history[0][0] < min_time:
            self._wrist_history.popleft()

    def _prune_aim_history(self, timestamp: float) -> None:
        min_time = timestamp - self._aim_history_seconds
        while self._aim_history and self._aim_history[0][0] < min_time:
            self._aim_history.popleft()

    def _reset_trigger_state(self) -> None:
        self._wrist_history.clear()
        self._trigger_started_at = None
        self._armed = True

    def _measure_upward_flick(
        self,
        timestamp: float,
        wrist_y: float,
    ) -> tuple[float, float]:
        if not self._wrist_history:
            return 0.0, 0.0

        baseline_time, baseline_y = max(
            self._wrist_history,
            key=lambda item: item[1],
        )
        elapsed = max(timestamp - baseline_time, 1e-6)
        upward_distance = baseline_y - wrist_y
        wrist_velocity_y = (wrist_y - baseline_y) / elapsed
        return wrist_velocity_y, upward_distance

    def _measure_frame_flick(self) -> tuple[float, float]:
        if len(self._wrist_history) < 3:
            return 0.0, 0.0

        values = list(self._wrist_history)
        upward_velocity = values[-3][1] - values[-1][1]
        if len(values) >= 4:
            previous_velocity = values[-4][1] - values[-2][1]
        else:
            previous_velocity = 0.0

        upward_acceleration = upward_velocity - previous_velocity
        return upward_velocity, upward_acceleration

    def _should_fire(
        self,
        timestamp: float,
        wrist_velocity_y: float,
        upward_distance: float,
        upward_velocity: float,
        upward_acceleration: float,
    ) -> bool:
        if self._trigger_started_at is None:
            return False

        trigger_elapsed = timestamp - self._trigger_started_at
        if trigger_elapsed < self._min_trigger_hold_seconds:
            return False

        if self._last_shot_at is not None:
            if timestamp - self._last_shot_at < self._cooldown_seconds:
                return False

        time_based_flick = (
            upward_distance >= self._min_flick_distance
            and -wrist_velocity_y >= self._min_upward_velocity
        )
        frame_based_flick = (
            upward_velocity >= self._min_frame_upward_velocity
            and upward_acceleration >= self._min_frame_upward_acceleration
        )
        return time_based_flick or frame_based_flick

    def _aim_before(self, target_time: float) -> tuple[float, float]:
        if not self._aim_history:
            return 0.0, 0.0

        selected = self._aim_history[0]
        for item in self._aim_history:
            if item[0] > target_time:
                break
            selected = item
        return selected[1], selected[2]
