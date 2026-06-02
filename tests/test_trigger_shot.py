"""trigger_shot 모듈 테스트."""

from __future__ import annotations

from src.ai.trigger_shot import TriggerShotDetector


def _detector() -> TriggerShotDetector:
    """테스트용 민감도를 고정한 detector 생성."""
    return TriggerShotDetector(
        pre_fire_seconds=0.10,
        min_trigger_hold_seconds=0.12,
        min_flick_distance=0.04,
        min_upward_velocity=0.8,
        cooldown_seconds=0.35,
    )


def test_does_not_fire_without_trigger() -> None:
    """trigger 상태가 아니면 손목을 위로 튕겨도 발사하지 않는다."""
    detector = _detector()

    detector.update(False, wrist_y=0.60, aim_x=100.0, aim_y=100.0, timestamp=0.0)
    update = detector.update(
        False,
        wrist_y=0.48,
        aim_x=200.0,
        aim_y=200.0,
        timestamp=0.12,
    )

    assert not update.fired


def test_does_not_fire_before_trigger_hold_time() -> None:
    """trigger 진입 직후 튕김은 발사로 인정하지 않는다."""
    detector = _detector()

    detector.update(True, wrist_y=0.60, aim_x=100.0, aim_y=100.0, timestamp=0.0)
    update = detector.update(
        True,
        wrist_y=0.50,
        aim_x=200.0,
        aim_y=200.0,
        timestamp=0.05,
    )

    assert not update.fired


def test_fires_from_previous_aim_position_on_upward_flick() -> None:
    """trigger 유지 후 위로 튕기면 직전 조준 위치로 발사한다."""
    detector = _detector()

    detector.update(True, wrist_y=0.60, aim_x=100.0, aim_y=110.0, timestamp=0.00)
    detector.update(True, wrist_y=0.60, aim_x=120.0, aim_y=130.0, timestamp=0.12)
    update = detector.update(
        True,
        wrist_y=0.49,
        aim_x=400.0,
        aim_y=300.0,
        timestamp=0.24,
    )

    assert update.fired
    assert update.fire_x == 120.0
    assert update.fire_y == 130.0
    assert update.aim_x == 400.0
    assert update.aim_y == 300.0


def test_fires_on_frame_based_acceleration_spike() -> None:
    """프로토타입 방식의 프레임 기반 가속도 스파이크도 발사로 인정."""
    detector = TriggerShotDetector(
        pre_fire_seconds=0.10,
        min_trigger_hold_seconds=0.05,
        min_flick_distance=1.0,
        min_upward_velocity=100.0,
        min_frame_upward_velocity=0.012,
        min_frame_upward_acceleration=0.008,
        cooldown_seconds=0.35,
    )

    detector.update(True, wrist_y=0.600, aim_x=100.0, aim_y=100.0, timestamp=0.00)
    detector.update(True, wrist_y=0.598, aim_x=120.0, aim_y=100.0, timestamp=0.04)
    detector.update(True, wrist_y=0.596, aim_x=140.0, aim_y=100.0, timestamp=0.08)
    update = detector.update(
        True,
        wrist_y=0.580,
        aim_x=300.0,
        aim_y=220.0,
        timestamp=0.12,
    )

    assert update.fired
    assert update.upward_velocity > 0.012
    assert update.upward_acceleration > 0.008


def test_does_not_fire_twice_without_rearm() -> None:
    """한 번 발사된 튕김은 재장전 전까지 중복 발사하지 않는다."""
    detector = _detector()

    detector.update(True, wrist_y=0.60, aim_x=100.0, aim_y=100.0, timestamp=0.00)
    detector.update(True, wrist_y=0.60, aim_x=120.0, aim_y=100.0, timestamp=0.12)
    first = detector.update(
        True,
        wrist_y=0.49,
        aim_x=200.0,
        aim_y=100.0,
        timestamp=0.24,
    )
    second = detector.update(
        True,
        wrist_y=0.47,
        aim_x=250.0,
        aim_y=100.0,
        timestamp=0.70,
    )

    assert first.fired
    assert not second.fired


def test_rearms_after_trigger_release() -> None:
    """trigger를 놓았다가 다시 잡으면 다음 튕김을 발사한다."""
    detector = _detector()

    detector.update(True, wrist_y=0.60, aim_x=100.0, aim_y=100.0, timestamp=0.00)
    detector.update(True, wrist_y=0.60, aim_x=120.0, aim_y=100.0, timestamp=0.12)
    first = detector.update(
        True,
        wrist_y=0.49,
        aim_x=200.0,
        aim_y=100.0,
        timestamp=0.24,
    )
    detector.update(False, wrist_y=0.58, aim_x=220.0, aim_y=120.0, timestamp=0.50)
    detector.update(True, wrist_y=0.60, aim_x=240.0, aim_y=140.0, timestamp=0.70)
    detector.update(True, wrist_y=0.60, aim_x=260.0, aim_y=160.0, timestamp=0.84)
    second = detector.update(
        True,
        wrist_y=0.49,
        aim_x=500.0,
        aim_y=300.0,
        timestamp=0.96,
    )

    assert first.fired
    assert second.fired
    assert second.fire_x == 260.0
    assert second.fire_y == 160.0
