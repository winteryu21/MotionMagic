"""gesture_mode_pipeline 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.gesture_modes import HandObservation
from src.bridge.camera_thread import CameraThread
from src.bridge.gesture_mode_pipeline import (
    GestureModePipeline,
    GestureModePipelineConfig,
)


def _base_landmarks() -> np.ndarray:
    """테스트용 기본 손 랜드마크를 생성한다."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.50, 0.80, 0.0]
    landmarks[3] = [0.53, 0.70, 0.0]
    landmarks[4] = [0.42, 0.62, 0.0]
    landmarks[5] = [0.55, 0.68, 0.0]
    landmarks[6] = [0.56, 0.50, 0.0]
    landmarks[8] = [0.56, 0.25, 0.0]
    landmarks[9] = [0.50, 0.66, 0.0]
    landmarks[10] = [0.50, 0.46, 0.0]
    landmarks[12] = [0.50, 0.20, 0.0]
    landmarks[13] = [0.45, 0.68, 0.0]
    landmarks[14] = [0.44, 0.50, 0.0]
    landmarks[16] = [0.44, 0.28, 0.0]
    landmarks[17] = [0.40, 0.70, 0.0]
    landmarks[18] = [0.39, 0.54, 0.0]
    landmarks[20] = [0.39, 0.34, 0.0]
    return landmarks


def _fold_finger(landmarks: np.ndarray, tip: int, folded_y: float) -> None:
    landmarks[tip] = [float(landmarks[tip][0]), folded_y, 0.0]


def _rock_landmarks() -> np.ndarray:
    landmarks = _base_landmarks()
    _fold_finger(landmarks, 8, 0.62)
    _fold_finger(landmarks, 12, 0.62)
    _fold_finger(landmarks, 16, 0.62)
    _fold_finger(landmarks, 20, 0.62)
    return landmarks


def _aim_landmarks(closed: bool = False) -> np.ndarray:
    landmarks = _rock_landmarks()
    landmarks[8] = [0.55, 0.35, 0.0]
    landmarks[4] = [0.54, 0.36, 0.0] if closed else [0.35, 0.45, 0.0]
    return landmarks


def _pipeline() -> GestureModePipeline:
    """테스트용 즉시 안정화 파이프라인을 생성한다."""
    return GestureModePipeline(
        GestureModePipelineConfig(
            frame_width=640,
            frame_height=480,
            ema_alpha=1.0,
            aim_sensitivity=1.0,
            mode_stable_seconds=0.0,
            stack_stable_seconds=0.0,
            stack_cooldown_seconds=0.0,
        )
    )


def test_camera_thread_swaps_handedness_by_default_for_mirrored_frame() -> None:
    """미러링된 카메라 입력은 기본적으로 좌우 handedness를 보정한다."""
    thread = CameraThread(lambda event: None)

    assert thread._effective_handedness("Left") == "Right"
    assert thread._effective_handedness("Right") == "Left"


def test_pipeline_emits_left_stack_event_after_stabilization() -> None:
    """왼손 rock 후보는 안정화 후 stack 이벤트가 된다."""
    pipeline = _pipeline()
    left = HandObservation("Left", _rock_landmarks(), score=0.9)

    assert pipeline.update([left], timestamp=0.00) == []
    events = pipeline.update([left], timestamp=0.01)

    assert len(events) == 1
    assert events[0].gesture == "rock"
    assert events[0].kind == "stack"
    assert events[0].channel == "left"


def test_pipeline_emits_right_aim_event_only_from_right_hand() -> None:
    """오른손 조준 포즈는 aim 이벤트로 변환된다."""
    pipeline = _pipeline()
    right = HandObservation("Right", _aim_landmarks(), score=0.8)

    events = pipeline.update([right], timestamp=0.00)

    assert len(events) == 1
    assert events[0].gesture == "aim"
    assert events[0].kind == "aim"
    assert events[0].channel == "right"
    assert events[0].aim_x == pytest.approx(0.55)
    assert events[0].aim_y == pytest.approx(0.35)


def test_pipeline_can_use_pinch_aim_anchor() -> None:
    """설정으로 예전 엄지/검지 중심 조준 anchor를 사용할 수 있다."""
    pipeline = GestureModePipeline(
        GestureModePipelineConfig(
            frame_width=640,
            frame_height=480,
            ema_alpha=1.0,
            aim_sensitivity=1.0,
            aim_anchor="pinch",
            mode_stable_seconds=0.0,
        )
    )
    right = HandObservation("Right", _aim_landmarks(), score=0.8)

    events = pipeline.update([right], timestamp=0.00)

    assert len(events) == 1
    assert events[0].gesture == "aim"
    assert events[0].aim_x == pytest.approx(0.45)
    assert events[0].aim_y == pytest.approx(0.40)


def test_pipeline_applies_aim_mapping_calibration() -> None:
    """파이프라인 설정의 조준 중심/축 감도를 aim 이벤트에 반영한다."""
    pipeline = GestureModePipeline(
        GestureModePipelineConfig(
            frame_width=640,
            frame_height=480,
            ema_alpha=1.0,
            aim_sensitivity=1.0,
            aim_sensitivity_x=4.0,
            aim_sensitivity_y=2.0,
            aim_center_x=0.55,
            aim_center_y=0.35,
            mode_stable_seconds=0.0,
        )
    )
    right = HandObservation("Right", _aim_landmarks(), score=0.8)

    events = pipeline.update([right], timestamp=0.00)

    assert len(events) == 1
    assert events[0].kind == "aim"
    assert events[0].aim_x == pytest.approx(0.50)
    assert events[0].aim_y == pytest.approx(0.50)


def test_pipeline_emits_fire_event_from_previous_aim_position() -> None:
    """핀치 발사는 닫히기 직전 조준 좌표로 fire 이벤트를 발생시킨다."""
    pipeline = _pipeline()
    open_right = HandObservation("Right", _aim_landmarks(closed=False), score=0.8)
    closed_right = HandObservation("Right", _aim_landmarks(closed=True), score=0.8)

    pipeline.update([open_right], timestamp=0.00)
    events = pipeline.update([closed_right], timestamp=0.10)

    fire_events = [event for event in events if event.kind == "fire"]
    assert len(fire_events) == 1
    assert fire_events[0].gesture == "fire"
    assert fire_events[0].channel == "right"
    assert fire_events[0].aim_x == pytest.approx(0.55)
    assert fire_events[0].aim_y == pytest.approx(0.35)


def test_pipeline_tracks_two_hand_gate_without_emitting_special_event() -> None:
    """양손 채널은 gate로만 추적하고 특수 제스처 이벤트는 아직 내보내지 않는다."""
    pipeline = _pipeline()
    left = HandObservation("Left", _rock_landmarks(), score=0.9)
    right = HandObservation("Right", _aim_landmarks(), score=0.8)

    events = pipeline.update([left, right], timestamp=0.00)

    assert pipeline.two_hand_active
    assert all(event.kind != "special" for event in events)
