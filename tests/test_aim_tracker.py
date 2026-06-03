"""aim_tracker 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.aim_tracker import (
    EmaAimTracker,
    aim_anchor_point,
    tracking_landmark_ids_for_anchor,
)


def _landmarks_with_tip(x: float, y: float) -> np.ndarray:
    """검지 끝 좌표만 설정한 테스트 랜드마크 생성."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[8] = [x, y, 0.0]
    return landmarks


def test_ema_aim_tracker_maps_first_position_directly() -> None:
    """첫 입력은 스무딩 없이 화면 좌표로 매핑."""
    tracker = EmaAimTracker(game_width=640, game_height=480, alpha=0.5)

    aim_x, aim_y = tracker.update(_landmarks_with_tip(0.25, 0.5))

    assert aim_x == 160.0
    assert aim_y == 240.0


def test_ema_aim_tracker_smooths_next_position() -> None:
    """두 번째 입력부터 EMA로 보간."""
    tracker = EmaAimTracker(game_width=640, game_height=480, alpha=0.5)

    tracker.update(_landmarks_with_tip(0.25, 0.5))
    aim_x, aim_y = tracker.update(_landmarks_with_tip(0.75, 0.25))

    assert aim_x == 320.0
    assert aim_y == 180.0


def test_ema_aim_tracker_can_average_two_fingertips_with_sensitivity() -> None:
    """검지/중지 끝 중간점을 감도 보정해 조준점으로 사용."""
    tracker = EmaAimTracker(
        game_width=640,
        game_height=480,
        alpha=1.0,
        sensitivity=2.0,
        tracking_landmark_ids=(8, 12),
    )
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[8] = [0.60, 0.40, 0.0]
    landmarks[12] = [0.70, 0.50, 0.0]

    aim_x, aim_y = tracker.update(landmarks)

    assert aim_x == pytest.approx(512.0)
    assert aim_y == pytest.approx(192.0)


def test_ema_aim_tracker_uses_axis_sensitivity_and_center() -> None:
    """X/Y 감도와 입력 중심점을 독립적으로 보정한다."""
    tracker = EmaAimTracker(
        game_width=640,
        game_height=480,
        alpha=1.0,
        sensitivity=1.0,
        sensitivity_x=4.0,
        sensitivity_y=2.0,
        center_x=0.60,
        center_y=0.40,
    )

    center_x, center_y = tracker.update(_landmarks_with_tip(0.60, 0.40))
    tracker.reset()
    aim_x, aim_y = tracker.update(_landmarks_with_tip(0.65, 0.45))

    assert center_x == pytest.approx(320.0)
    assert center_y == pytest.approx(240.0)
    assert aim_x == pytest.approx(448.0)
    assert aim_y == pytest.approx(288.0)


def test_ema_aim_tracker_clamps_after_axis_sensitivity() -> None:
    """큰 감도 보정 후에도 화면 범위를 벗어나지 않는다."""
    tracker = EmaAimTracker(
        game_width=640,
        game_height=480,
        alpha=1.0,
        sensitivity_x=8.0,
        sensitivity_y=8.0,
    )

    aim_x, aim_y = tracker.update(_landmarks_with_tip(0.90, 0.05))

    assert aim_x == 640.0
    assert aim_y == 0.0


def test_aim_anchor_helpers_select_index_or_pinch_center() -> None:
    """조준 anchor에 따라 검지 끝 또는 엄지/검지 중심을 반환한다."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[4] = [0.20, 0.40, 0.0]
    landmarks[8] = [0.60, 0.20, 0.0]

    assert tracking_landmark_ids_for_anchor("index") == (8,)
    assert tracking_landmark_ids_for_anchor("pinch") == (4, 8)
    assert aim_anchor_point(landmarks, "index") == pytest.approx((0.60, 0.20))
    assert aim_anchor_point(landmarks, "pinch") == pytest.approx((0.40, 0.30))
