"""조준선 추적기.

손의 특정 랜드마크 좌표를 게임 해상도(1920×1080)에 맵핑하고,
스무딩 필터로 손떨림을 부드럽게 보정한다.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

GAME_WIDTH = 1920
GAME_HEIGHT = 1080
TRACKING_LANDMARK_ID = 8  # 검지 끝
AimAnchor = Literal["index", "pinch"]
INDEX_AIM_LANDMARK_IDS = (8,)
PINCH_AIM_LANDMARK_IDS = (4, 8)

# 1€ Filter 기본 파라미터
DEFAULT_MIN_CUTOFF = 1.0
DEFAULT_BETA = 0.007
DEFAULT_D_CUTOFF = 1.0
DEFAULT_EMA_ALPHA = 0.3


def tracking_landmark_ids_for_anchor(anchor: AimAnchor) -> tuple[int, ...]:
    """조준 anchor에 해당하는 랜드마크 ID 목록을 반환한다.

    Args:
        anchor: ``"index"`` 또는 ``"pinch"``.

    Returns:
        조준점 평균 계산에 사용할 랜드마크 ID 튜플.
    """
    if anchor == "index":
        return INDEX_AIM_LANDMARK_IDS
    return PINCH_AIM_LANDMARK_IDS


def aim_anchor_point(landmarks: np.ndarray, anchor: AimAnchor) -> tuple[float, float]:
    """조준 anchor의 원시 정규화 좌표를 반환한다.

    Args:
        landmarks: ``(21, 2)`` 또는 ``(21, 3)`` 원시 랜드마크 좌표.
        anchor: ``"index"`` 또는 ``"pinch"``.

    Returns:
        MediaPipe 정규화 좌표계의 ``(x, y)``.
    """
    landmark_ids = tracking_landmark_ids_for_anchor(anchor)
    points = landmarks[list(landmark_ids)]
    point = points.mean(axis=0)
    return float(point[0]), float(point[1])


# ---------------------------------------------------------------------------
# 1€ Filter 구현
# ---------------------------------------------------------------------------


class OneEuroFilter:
    """원-유로 필터 (1€ Filter).

    느린 움직임에는 강한 필터링(흔들림 방지),
    빠른 움직임에는 약한 필터링(지연 최소화)을 적용한다.

    Args:
        min_cutoff: 최소 컷오프 주파수. 값이 클수록 필터링이 약해짐.
        beta: 속도 의존 컷오프 가중치. 값이 클수록 빠른 움직임 추적이 향상.
        d_cutoff: 미분 신호의 컷오프 주파수.
    """

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        """지수 이동 평균의 스무딩 팩터 계산.

        Args:
            t_e: 타임스텝 간격.
            cutoff: 컷오프 주파수.

        Returns:
            스무딩 팩터 α (0~1).
        """
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    @staticmethod
    def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
        """지수 이동 평균.

        Args:
            a: 스무딩 팩터.
            x: 현재 값.
            x_prev: 이전 값.

        Returns:
            보정된 값.
        """
        return a * x + (1.0 - a) * x_prev

    def __call__(self, x: float, t: float | None = None) -> float:
        """필터 적용.

        Args:
            x: 현재 원시 값.
            t: 현재 타임스탬프. ``None``이면 자동 측정.

        Returns:
            필터링된 값.
        """
        if t is None:
            t = time.time()

        if self._t_prev is None:
            # 첫 번째 호출: 초기화
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            t_e = 1e-6

        # 미분 신호 추정
        a_d = self._smoothing_factor(t_e, self._d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # 속도에 따른 적응형 컷오프
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)

        # 메인 필터링
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev)

        # 상태 업데이트
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self) -> None:
        """필터 상태 초기화."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


# ---------------------------------------------------------------------------
# 조준선 추적기
# ---------------------------------------------------------------------------


class AimTracker:
    """1€ Filter 기반 조준선 추적기.

    검지 끝(Landmark 8) 좌표를 게임 해상도에 맵핑하고,
    X/Y 각각에 독립적인 1€ 필터를 적용하여 부드럽게 추적한다.

    Args:
        game_width: 게임 화면 너비 (기본 1920).
        game_height: 게임 화면 높이 (기본 1080).
        min_cutoff: 1€ 필터 최소 컷오프 주파수.
        beta: 1€ 필터 속도 가중치.
    """

    def __init__(
        self,
        game_width: int = GAME_WIDTH,
        game_height: int = GAME_HEIGHT,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self._game_width = game_width
        self._game_height = game_height
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def update(
        self,
        landmarks: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[float, float]:
        """조준점 갱신.

        Args:
            landmarks: ``(21, 2)`` 또는 ``(21, 3)`` 원시 랜드마크 좌표.
                x, y 값은 0.0~1.0 범위 (MediaPipe 정규화 좌표).
            timestamp: 현재 타임스탬프. ``None``이면 자동 측정.

        Returns:
            ``(aim_x, aim_y)`` 게임 해상도 기준 조준점 좌표.
        """
        tip = landmarks[TRACKING_LANDMARK_ID]  # (2,)
        raw_x, raw_y = float(tip[0]), float(tip[1])

        # 1€ 필터 적용
        smooth_x = self._filter_x(raw_x, timestamp)
        smooth_y = self._filter_y(raw_y, timestamp)

        # 게임 해상도로 맵핑 (0.0~1.0 → 0~1920, 0~1080)
        aim_x = smooth_x * self._game_width
        aim_y = smooth_y * self._game_height

        # 범위 클램핑
        aim_x = max(0.0, min(aim_x, float(self._game_width)))
        aim_y = max(0.0, min(aim_y, float(self._game_height)))

        return aim_x, aim_y

    def reset(self) -> None:
        """필터 상태 초기화."""
        self._filter_x.reset()
        self._filter_y.reset()


class EmaFilter:
    """지수가중이동평균(EMA) 필터.

    Args:
        alpha: 현재 입력 반영 비율. 1에 가까울수록 반응이 빠르고,
            0에 가까울수록 더 부드럽다.
    """

    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA) -> None:
        self._alpha = alpha
        self._value: float | None = None

    def __call__(self, x: float) -> float:
        """필터 적용.

        Args:
            x: 현재 원시 값.

        Returns:
            필터링된 값.
        """
        if self._value is None:
            self._value = x
            return x

        self._value = self._alpha * x + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        """필터 상태 초기화."""
        self._value = None


class EmaAimTracker:
    """EMA 기반 조준선 추적기.

    검지 끝(Landmark 8) 좌표를 게임 해상도에 맵핑하고,
    X/Y 각각에 독립적인 EMA 필터를 적용한다.

    Args:
        game_width: 게임 화면 너비.
        game_height: 게임 화면 높이.
        alpha: EMA 현재 입력 반영 비율.
        sensitivity: X/Y 공통 조준 감도.
        sensitivity_x: X축 조준 감도. ``None``이면 ``sensitivity``를 사용한다.
        sensitivity_y: Y축 조준 감도. ``None``이면 ``sensitivity``를 사용한다.
        center_x: 화면 중앙으로 매핑할 입력 X 좌표.
        center_y: 화면 중앙으로 매핑할 입력 Y 좌표.
        tracking_landmark_ids: 조준점 계산에 사용할 랜드마크 ID 목록.
    """

    def __init__(
        self,
        game_width: int = GAME_WIDTH,
        game_height: int = GAME_HEIGHT,
        alpha: float = DEFAULT_EMA_ALPHA,
        sensitivity: float = 1.0,
        sensitivity_x: float | None = None,
        sensitivity_y: float | None = None,
        center_x: float = 0.5,
        center_y: float = 0.5,
        tracking_landmark_ids: tuple[int, ...] = (TRACKING_LANDMARK_ID,),
    ) -> None:
        self._game_width = game_width
        self._game_height = game_height
        self._sensitivity_x = sensitivity if sensitivity_x is None else sensitivity_x
        self._sensitivity_y = sensitivity if sensitivity_y is None else sensitivity_y
        self._center_x = center_x
        self._center_y = center_y
        self._tracking_landmark_ids = tracking_landmark_ids
        self._filter_x = EmaFilter(alpha=alpha)
        self._filter_y = EmaFilter(alpha=alpha)

    def update(
        self,
        landmarks: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[float, float]:
        """조준점 갱신.

        Args:
            landmarks: ``(21, 2)`` 또는 ``(21, 3)`` 원시 랜드마크 좌표.
            timestamp: 현재 타임스탬프. EMA에서는 사용하지 않는다.

        Returns:
            ``(aim_x, aim_y)`` 게임 해상도 기준 조준점 좌표.
        """
        _ = timestamp
        points = landmarks[list(self._tracking_landmark_ids)]
        point = points.mean(axis=0)
        raw_x = 0.5 + (float(point[0]) - self._center_x) * self._sensitivity_x
        raw_y = 0.5 + (float(point[1]) - self._center_y) * self._sensitivity_y

        smooth_x = self._filter_x(raw_x)
        smooth_y = self._filter_y(raw_y)

        aim_x = smooth_x * self._game_width
        aim_y = smooth_y * self._game_height

        aim_x = max(0.0, min(aim_x, float(self._game_width)))
        aim_y = max(0.0, min(aim_y, float(self._game_height)))

        return aim_x, aim_y

    def reset(self) -> None:
        """필터 상태 초기화."""
        self._filter_x.reset()
        self._filter_y.reset()
