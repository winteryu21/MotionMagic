"""gesture_modes 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.gesture_modes import (
    AimModeTracker,
    HandObservation,
    PinchFireDetector,
    SpecialGestureDebouncer,
    StackGestureDebouncer,
    aim_point,
    assign_hands,
    classify_special_gesture,
    classify_stack_gesture,
    compute_finger_states,
    is_aim_pose,
    normalized_pinch_distance,
)


def _base_landmarks() -> np.ndarray:
    """테스트용 기본 손 랜드마크를 생성한다."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[3] = [0.20, -0.10, 0.0]
    landmarks[4] = [0.45, -0.18, 0.0]
    landmarks[5] = [0.16, -0.18, 0.0]
    landmarks[6] = [0.18, -0.35, 0.0]
    landmarks[8] = [0.20, -0.70, 0.0]
    landmarks[9] = [0.00, -0.30, 0.0]
    landmarks[10] = [0.00, -0.45, 0.0]
    landmarks[12] = [0.00, -0.75, 0.0]
    landmarks[13] = [-0.16, -0.18, 0.0]
    landmarks[14] = [-0.18, -0.30, 0.0]
    landmarks[16] = [-0.18, -0.55, 0.0]
    landmarks[17] = [-0.30, -0.12, 0.0]
    landmarks[18] = [-0.32, -0.24, 0.0]
    landmarks[20] = [-0.32, -0.45, 0.0]
    return landmarks


def _fold_finger(landmarks: np.ndarray, tip: int, folded_y: float) -> None:
    landmarks[tip] = [landmarks[tip][0], folded_y, 0.0]


def _rock_landmarks() -> np.ndarray:
    landmarks = _base_landmarks()
    _fold_finger(landmarks, 8, -0.25)
    _fold_finger(landmarks, 12, -0.25)
    _fold_finger(landmarks, 16, -0.20)
    _fold_finger(landmarks, 20, -0.16)
    return landmarks


def _scissors_landmarks() -> np.ndarray:
    landmarks = _base_landmarks()
    _fold_finger(landmarks, 16, -0.20)
    _fold_finger(landmarks, 20, -0.16)
    return landmarks


def _aim_landmarks(pinch_distance: float = 1.0) -> np.ndarray:
    landmarks = _rock_landmarks()
    landmarks[4] = [0.20 + pinch_distance * 0.30, -0.70, 0.0]
    landmarks[8] = [0.20, -0.70, 0.0]
    return landmarks


def _shifted(landmarks: np.ndarray, dx: float) -> np.ndarray:
    shifted = landmarks.copy()
    shifted[:, 0] += dx
    return shifted


def test_classifies_left_stack_gestures() -> None:
    """rock/scissors/paper 후보를 손가락 상태에서 구분한다."""
    assert classify_stack_gesture(compute_finger_states(_rock_landmarks())) == "rock"
    assert (
        classify_stack_gesture(compute_finger_states(_scissors_landmarks()))
        == "scissors"
    )
    assert classify_stack_gesture(compute_finger_states(_base_landmarks())) == "paper"


def test_aim_pose_requires_index_and_folded_ring_pinky() -> None:
    """오른손 조준 후보는 검지가 펴지고 약지/새끼가 접힌 상태다."""
    states = compute_finger_states(_aim_landmarks())

    assert is_aim_pose(states)
    assert not is_aim_pose(compute_finger_states(_base_landmarks()))


def test_aim_pose_tolerates_unstable_thumb_and_middle_state() -> None:
    """엄지/중지 펼침 판정이 흔들려도 조준 후보를 유지한다."""
    states = compute_finger_states(_aim_landmarks())
    states[0] = 0.0
    states[2] = 1.0

    assert is_aim_pose(states)


def test_aim_point_uses_thumb_index_center() -> None:
    """조준점은 엄지/검지 끝 중심점이다."""
    landmarks = _aim_landmarks()

    x, y = aim_point(landmarks)

    assert x == pytest.approx((landmarks[4][0] + landmarks[8][0]) * 0.5)
    assert y == pytest.approx((landmarks[4][1] + landmarks[8][1]) * 0.5)


def test_normalized_pinch_distance_scales_by_hand_size() -> None:
    """핀치 거리는 손 크기 기준으로 정규화한다."""
    landmarks = _aim_landmarks()

    distance = normalized_pinch_distance(landmarks)

    assert distance > 0.0


def test_assign_hands_keeps_best_left_and_right() -> None:
    """같은 handedness가 여러 개면 점수가 높은 손을 채널에 배정한다."""
    left_low = HandObservation("Left", _base_landmarks(), score=0.2)
    left_high = HandObservation("Left", _base_landmarks(), score=0.9)
    right = HandObservation("Right", _base_landmarks(), score=0.8)

    left, assigned_right = assign_hands([left_low, right, left_high])

    assert left is left_high
    assert assigned_right is right


def test_classifies_clasp_when_open_hands_are_close() -> None:
    """합장은 두 열린 손의 손끝들이 가까운 양손 제스처다."""
    left = _shifted(_base_landmarks(), -0.08)
    right = _shifted(_base_landmarks(), 0.08)

    assert classify_special_gesture(left, right) == "clasp"


def test_rejects_clasp_when_hands_are_far_apart() -> None:
    """열린 손이라도 거리가 멀면 합장으로 보지 않는다."""
    left = _shifted(_base_landmarks(), -1.00)
    right = _shifted(_base_landmarks(), 1.00)

    assert classify_special_gesture(left, right) is None


def test_classifies_sonaldo_from_two_open_aim_poses() -> None:
    """손흥민 시그니처는 양손 엄지/검지가 열린 조준형 포즈다."""
    left = _shifted(_aim_landmarks(pinch_distance=1.0), -0.08)
    right = _shifted(_aim_landmarks(pinch_distance=1.0), 0.08)

    assert classify_special_gesture(left, right) == "sonaldo"


def test_special_debouncer_emits_once_after_stable_delay() -> None:
    """양손 특수 제스처는 안정화 후 1회만 emit된다."""
    debouncer = SpecialGestureDebouncer(stable_seconds=0.10, cooldown_seconds=0.30)

    assert debouncer.update("clasp", timestamp=0.00).emitted is None
    assert debouncer.update("clasp", timestamp=0.05).emitted is None
    assert debouncer.update("clasp", timestamp=0.11).emitted == "clasp"
    assert debouncer.update("clasp", timestamp=0.20).emitted is None


def test_stack_debouncer_emits_once_after_stable_delay() -> None:
    """왼손 스택 제스처는 안정화 후 1회만 emit된다."""
    debouncer = StackGestureDebouncer(stable_seconds=0.10, cooldown_seconds=0.30)

    assert debouncer.update("rock", timestamp=0.00).emitted is None
    assert debouncer.update("rock", timestamp=0.05).emitted is None
    assert debouncer.update("rock", timestamp=0.11).emitted == "rock"
    assert debouncer.update("rock", timestamp=0.30).emitted is None


def test_stack_debouncer_grace_prevents_tracking_dropout_repeat() -> None:
    """짧은 tracking 끊김은 같은 제스처 중복 emit으로 이어지지 않는다."""
    debouncer = StackGestureDebouncer(
        stable_seconds=0.10,
        grace_seconds=0.20,
        cooldown_seconds=0.30,
    )

    assert debouncer.update("rock", timestamp=0.00).emitted is None
    assert debouncer.update("rock", timestamp=0.12).emitted == "rock"
    assert debouncer.update(None, timestamp=0.18).emitted is None
    assert debouncer.update("rock", timestamp=0.24).emitted is None


def test_aim_mode_tracker_has_delay_and_grace() -> None:
    """오른손 조준 모드는 안정화 지연과 끊김 유예를 가진다."""
    tracker = AimModeTracker(stable_seconds=0.10, grace_seconds=0.20)

    assert not tracker.update(True, timestamp=0.00).active
    assert tracker.update(True, timestamp=0.11).active
    assert tracker.update(False, timestamp=0.20).active
    assert not tracker.update(False, timestamp=0.40).active


def test_pinch_fire_only_fires_while_aim_active() -> None:
    """핀치 발사는 조준 모드가 active일 때만 동작한다."""
    detector = PinchFireDetector(
        open_threshold=0.70,
        closed_threshold=0.30,
        close_velocity=1.0,
    )

    detector.update(False, 0.80, 100.0, 100.0, timestamp=0.00)
    update = detector.update(False, 0.20, 120.0, 120.0, timestamp=0.10)

    assert not update.fired


def test_pinch_fire_reports_distance_even_when_aim_inactive() -> None:
    """조준 비활성 상태에서도 진단용 핀치 거리는 보존한다."""
    detector = PinchFireDetector()

    update = detector.update(False, 0.42, 120.0, 120.0, timestamp=0.10)

    assert not update.fired
    assert update.pinch_distance == 0.42
    assert update.pinch_velocity == 0.0


def test_pinch_fire_uses_previous_aim_position() -> None:
    """핀치 발사는 닫히기 직전 조준 좌표로 1회 발생한다."""
    detector = PinchFireDetector(
        open_threshold=0.70,
        closed_threshold=0.30,
        close_velocity=1.0,
        pre_fire_seconds=0.08,
    )

    detector.update(True, 0.90, 100.0, 110.0, timestamp=0.00)
    detector.update(True, 0.80, 130.0, 140.0, timestamp=0.08)
    update = detector.update(True, 0.20, 300.0, 310.0, timestamp=0.16)

    assert update.fired
    assert update.fire_x == 130.0
    assert update.fire_y == 140.0
    assert not update.armed


def test_pinch_fire_uses_distance_delta_when_velocity_is_soft() -> None:
    """속도값이 애매해도 프레임 간 닫힘 변화가 충분하면 발사된다."""
    detector = PinchFireDetector(
        open_threshold=0.70,
        closed_threshold=0.55,
        close_velocity=10.0,
        close_delta=0.12,
    )

    detector.update(True, 0.70, 100.0, 100.0, timestamp=0.00)
    update = detector.update(True, 0.52, 120.0, 120.0, timestamp=0.10)

    assert update.fired


def test_pinch_fire_ignores_static_closed_pinch() -> None:
    """닫힌 상태로 가만히 있는 손은 발사로 보지 않는다."""
    detector = PinchFireDetector(
        open_threshold=0.70,
        closed_threshold=0.55,
        close_velocity=0.60,
        close_delta=0.12,
    )

    detector.update(True, 0.52, 100.0, 100.0, timestamp=0.00)
    update = detector.update(True, 0.51, 120.0, 120.0, timestamp=0.10)

    assert not update.fired


def test_pinch_fire_rearms_after_reopen() -> None:
    """핀치는 다시 충분히 벌어진 뒤에만 재발사된다."""
    detector = PinchFireDetector(
        open_threshold=0.70,
        closed_threshold=0.30,
        close_velocity=1.0,
        cooldown_seconds=0.10,
    )

    detector.update(True, 0.90, 100.0, 100.0, timestamp=0.00)
    first = detector.update(True, 0.20, 120.0, 120.0, timestamp=0.10)
    second = detector.update(True, 0.10, 140.0, 140.0, timestamp=0.30)
    detector.update(True, 0.85, 160.0, 160.0, timestamp=0.40)
    third = detector.update(True, 0.20, 180.0, 180.0, timestamp=0.52)

    assert first.fired
    assert not second.fired
    assert third.fired
