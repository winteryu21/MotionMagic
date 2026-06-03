"""07 — 왼손/오른손/양손 제스처 모드 진단 테스트.

MediaPipe Hand Landmarker만 사용해 세 입력 채널을 분리해서 검증한다.

사용법:
    python scripts/07_gesture_mode_test.py
    python scripts/07_gesture_mode_test.py --no-swap-handedness
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import cast

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision import (
    RunningMode as VisionRunningMode,
)

from src.ai.aim_tracker import (
    AimAnchor,
    EmaAimTracker,
    aim_anchor_point,
    tracking_landmark_ids_for_anchor,
)
from src.ai.collector import HAND_LANDMARKER_MODEL, _draw_landmarks
from src.ai.gesture_modes import (
    DEFAULT_PINCH_CLOSE_DELTA,
    DEFAULT_PINCH_CLOSE_VELOCITY,
    DEFAULT_PINCH_CLOSED_THRESHOLD,
    DEFAULT_PINCH_OPEN_THRESHOLD,
    DEFAULT_PRE_FIRE_SECONDS,
    AimModeTracker,
    BoolDebouncer,
    FireUpdate,
    HandLabel,
    HandObservation,
    PinchFireDetector,
    SpecialGestureDebouncer,
    StackGestureDebouncer,
    assign_hands,
    classify_special_gesture,
    classify_stack_gesture,
    compute_finger_states,
    is_aim_pose,
    normalized_pinch_distance,
)
from src.ai.preprocessor import extract_landmarks

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
UI_PANEL_WIDTH = 560
SHOT_MARKER_SECONDS = 0.65
DEFAULT_AIM_SENSITIVITY = 3.0
LEFT_HAND_COLOR = (80, 220, 120)
RIGHT_HAND_COLOR = (0, 220, 255)
UNKNOWN_HAND_COLOR = (160, 160, 160)
SPECIAL_MODE_COLOR = (255, 180, 80)
PINCH_OPEN_COLOR = (80, 220, 120)
PINCH_CLOSED_COLOR = (0, 0, 255)
RAW_AIM_COLOR = (255, 80, 255)


def _swap_label(label: str) -> str:
    """좌우 handedness 라벨을 뒤집는다.

    Args:
        label: MediaPipe handedness 라벨.

    Returns:
        뒤집힌 라벨. 알 수 없는 라벨은 그대로 반환한다.
    """
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def _hand_color(label: str | None) -> tuple[int, int, int]:
    """손 라벨별 표시 색상을 반환한다.

    Args:
        label: handedness 라벨.

    Returns:
        BGR 색상.
    """
    if label == "Left":
        return LEFT_HAND_COLOR
    if label == "Right":
        return RIGHT_HAND_COLOR
    return UNKNOWN_HAND_COLOR


def _landmark_dicts(hand_landmarks: object) -> list[dict[str, float]]:
    """MediaPipe 랜드마크 객체를 딕셔너리 리스트로 변환한다.

    Args:
        hand_landmarks: MediaPipe hand landmarks.

    Returns:
        ``[{"x": ..., "y": ..., "z": ...}, ...]`` 형태의 좌표.
    """
    return [
        {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks
    ]


def _extract_handedness(result: object, hand_index: int) -> tuple[str | None, float]:
    """MediaPipe 결과에서 handedness 라벨과 점수를 추출한다.

    Args:
        result: HandLandmarker 결과.
        hand_index: 손 인덱스.

    Returns:
        ``(label, score)`` 튜플.
    """
    handedness = getattr(result, "handedness", None)
    if not handedness or hand_index >= len(handedness) or not handedness[hand_index]:
        return None, 0.0

    category = handedness[hand_index][0]
    return str(category.category_name), float(category.score)


def _read_observations(
    result: object,
    swap_handedness: bool,
) -> tuple[list[HandObservation], list[tuple[list[dict[str, float]], str | None]]]:
    """MediaPipe 결과를 손 채널 관측값으로 변환한다.

    Args:
        result: HandLandmarker 결과.
        swap_handedness: 좌우 라벨 보정 여부.

    Returns:
        ``(observations, draw_items)`` 튜플.
    """
    observations: list[HandObservation] = []
    draw_items: list[tuple[list[dict[str, float]], str | None]] = []
    hand_landmarks = getattr(result, "hand_landmarks", None)
    if not hand_landmarks:
        return observations, draw_items

    for index, landmarks in enumerate(hand_landmarks):
        raw_label, score = _extract_handedness(result, index)
        effective_label = (
            _swap_label(raw_label) if swap_handedness and raw_label else raw_label
        )
        landmark_dicts = _landmark_dicts(landmarks)
        draw_items.append((landmark_dicts, effective_label))

        if effective_label not in {"Left", "Right"}:
            continue

        observations.append(
            HandObservation(
                label=cast(HandLabel, effective_label),
                landmarks=extract_landmarks(landmark_dicts),
                score=score,
            )
        )

    return observations, draw_items


def _draw_hand_label(
    frame: np.ndarray,
    landmarks: list[dict[str, float]],
    label: str | None,
    score: float | None = None,
) -> None:
    """손목 근처에 채널 라벨을 표시한다.

    Args:
        frame: BGR 이미지.
        landmarks: 랜드마크 딕셔너리 리스트.
        label: handedness 라벨.
        score: handedness 신뢰도.
    """
    if not landmarks:
        return

    height, width = frame.shape[:2]
    wrist = landmarks[0]
    x = int(wrist["x"] * width)
    y = int(wrist["y"] * height)
    text = label or "Unknown"
    if score is not None:
        text = f"{text} {score:.2f}"

    cv2.putText(
        frame,
        text,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        _hand_color(label),
        2,
    )


def _draw_shots(
    frame: np.ndarray,
    shots: deque[tuple[float, float, float]],
    now: float,
) -> None:
    """최근 핀치 발사 위치를 화면에 표시한다.

    Args:
        frame: BGR 이미지.
        shots: ``(timestamp, x, y)`` 발사 위치 큐.
        now: 현재 시간.
    """
    while shots and now - shots[0][0] > SHOT_MARKER_SECONDS:
        shots.popleft()

    for shot_time, x, y in shots:
        age = now - shot_time
        radius = int(18 + age * 42)
        center = (int(x), int(y))
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
        cv2.drawMarker(frame, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 2)


def _draw_right_aim_debug(
    frame: np.ndarray,
    landmarks: np.ndarray,
    is_candidate: bool,
    is_active: bool,
    aim_anchor: AimAnchor,
) -> None:
    """오른손 조준 anchor와 엄지/검지 핀치 상태를 크게 표시한다.

    Args:
        frame: BGR 이미지.
        landmarks: ``(21, 3)`` 오른손 원시 좌표.
        is_candidate: 이번 프레임이 조준 후보인지 여부.
        is_active: 안정화된 조준 활성 여부.
        aim_anchor: 조준점 anchor.
    """
    height, width = frame.shape[:2]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    thumb_pt = (int(thumb_tip[0] * width), int(thumb_tip[1] * height))
    index_pt = (int(index_tip[0] * width), int(index_tip[1] * height))
    raw_aim_x, raw_aim_y = aim_anchor_point(landmarks, aim_anchor)
    raw_aim = (int(raw_aim_x * width), int(raw_aim_y * height))

    color = RIGHT_HAND_COLOR if is_active else RAW_AIM_COLOR
    if not is_candidate:
        color = UNKNOWN_HAND_COLOR

    cv2.line(frame, thumb_pt, index_pt, color, 3)
    cv2.circle(frame, thumb_pt, 8, color, 2)
    cv2.circle(frame, index_pt, 8, color, 2)
    cv2.circle(frame, raw_aim, 10, color, 2)
    cv2.drawMarker(frame, raw_aim, color, cv2.MARKER_CROSS, 24, 2)


def _draw_status(
    panel: np.ndarray,
    stack_text: str,
    left_text: str,
    right_text: str,
    fire_update: FireUpdate | None,
    special_active: bool,
    special_text: str,
    shot_count: int,
    fps: float,
    pinch_closed_threshold: float,
    aim_active: bool,
    has_right_hand: bool,
    pre_fire_seconds: float,
) -> None:
    """현재 모드 상태를 별도 UI 패널에 표시한다.

    Args:
        panel: BGR 패널 이미지.
        stack_text: 왼손 스택 문자열.
        left_text: 왼손 상태 문자열.
        right_text: 오른손 상태 문자열.
        fire_update: 최근 발사 상태.
        special_active: 양손 채널 안정화 여부.
        special_text: 양손 특수 후보/안정화 상태 문자열.
        shot_count: 누적 발사 수.
        fps: 평균 FPS.
        pinch_closed_threshold: 발사로 볼 엄지-검지 닫힘 거리.
        aim_active: 조준 모드 활성 여부.
        has_right_hand: 오른손 채널 감지 여부.
        pre_fire_seconds: 발사 좌표를 되돌아볼 조준 기록 시간.
    """
    panel[:] = (0, 0, 0)

    pinch = 0.0 if fire_update is None else fire_update.pinch_distance
    pinch_velocity = 0.0 if fire_update is None else fire_update.pinch_velocity
    armed = "Y" if fire_update is None or fire_update.armed else "N"
    pinch_state = "aim-off"
    if aim_active:
        pinch_state = "ready"
    elif not has_right_hand:
        pinch_state = "no-right"
    special_gate_text = "ON" if special_active else "OFF"
    special_display = f"{special_gate_text} {special_text}"
    special_color = SPECIAL_MODE_COLOR if special_active else (130, 130, 130)

    lines = [
        ("Left stack", left_text, LEFT_HAND_COLOR),
        ("Stack", stack_text or "-", (230, 230, 230)),
        ("Right aim", right_text, RIGHT_HAND_COLOR),
        (
            "Pinch",
            (
                f"dist={pinch:.2f}/{pinch_closed_threshold:.2f} "
                f"vel={pinch_velocity:+.2f} armed={armed} {pinch_state}"
            ),
            (230, 230, 230),
        ),
        ("Two-hand", special_display, special_color),
        ("Fire aim", f"-{pre_fire_seconds:.2f}s history", (230, 230, 230)),
        ("Shots/FPS", f"{shot_count} / {fps:.0f}", (230, 230, 230)),
        ("Keys", "q/ESC quit, r reset", (180, 180, 180)),
    ]

    cv2.putText(
        panel,
        "MotionMagic AI Debug",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    y = 68
    for label, value, color in lines:
        cv2.putText(
            panel,
            f"{label}: {value}",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2 if label in {"Left stack", "Right aim", "Two-hand"} else 1,
        )
        y += 30


def _finger_text(finger_states: np.ndarray | None) -> str:
    """손가락 상태를 짧은 문자열로 변환한다.

    Args:
        finger_states: ``(5,)`` 펼침 상태.

    Returns:
        ``TIMP`` 순서의 상태 문자열.
    """
    if finger_states is None:
        return "-----"
    return "".join("O" if state == 1.0 else "X" for state in finger_states)


def run_test(
    camera_id: int = 0,
    ema_alpha: float = 0.3,
    aim_sensitivity: float = DEFAULT_AIM_SENSITIVITY,
    aim_sensitivity_x: float | None = None,
    aim_sensitivity_y: float | None = None,
    aim_center_x: float = 0.5,
    aim_center_y: float = 0.5,
    aim_anchor: AimAnchor = "index",
    swap_handedness: bool = True,
    right_only: bool = False,
    pinch_open_threshold: float = DEFAULT_PINCH_OPEN_THRESHOLD,
    pinch_closed_threshold: float = DEFAULT_PINCH_CLOSED_THRESHOLD,
    pinch_close_velocity: float = DEFAULT_PINCH_CLOSE_VELOCITY,
    pinch_close_delta: float = DEFAULT_PINCH_CLOSE_DELTA,
) -> None:
    """세 제스처 모드 진단 테스트를 실행한다.

    Args:
        camera_id: 카메라 디바이스 ID.
        ema_alpha: 오른손 조준점 EMA 현재 입력 반영 비율.
        aim_sensitivity: 화면 중심 기준 조준 감도.
        aim_sensitivity_x: X축 조준 감도.
        aim_sensitivity_y: Y축 조준 감도.
        aim_center_x: 화면 중앙으로 매핑할 입력 X 좌표.
        aim_center_y: 화면 중앙으로 매핑할 입력 Y 좌표.
        aim_anchor: 조준점 anchor.
        swap_handedness: MediaPipe handedness 좌우 라벨 보정 여부.
        right_only: 오른손 조준/발사만 집중해서 볼지 여부.
        pinch_open_threshold: 재장전으로 볼 엄지-검지 거리.
        pinch_closed_threshold: 발사로 볼 엄지-검지 닫힘 거리.
        pinch_close_velocity: 발사로 볼 닫힘 속도.
        pinch_close_delta: 발사로 볼 프레임 간 닫힘 변화량.
    """
    if not HAND_LANDMARKER_MODEL.exists():
        print(f"HandLandmarker 모델 없음: {HAND_LANDMARKER_MODEL}")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"카메라(ID={camera_id})를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    left_stack = StackGestureDebouncer()
    aim_mode = AimModeTracker()
    two_hand_gate = BoolDebouncer()
    special_debouncer = SpecialGestureDebouncer()
    aim_tracker = EmaAimTracker(
        game_width=CAMERA_WIDTH,
        game_height=CAMERA_HEIGHT,
        alpha=ema_alpha,
        sensitivity=aim_sensitivity,
        sensitivity_x=aim_sensitivity_x,
        sensitivity_y=aim_sensitivity_y,
        center_x=aim_center_x,
        center_y=aim_center_y,
        tracking_landmark_ids=tracking_landmark_ids_for_anchor(aim_anchor),
    )
    fire_detector = PinchFireDetector(
        open_threshold=pinch_open_threshold,
        closed_threshold=pinch_closed_threshold,
        close_velocity=pinch_close_velocity,
        close_delta=pinch_close_delta,
    )

    stack: list[str] = []
    shots: deque[tuple[float, float, float]] = deque()
    fps_history: deque[float] = deque(maxlen=30)
    shot_count = 0
    last_aim: tuple[float, float] | None = None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("\n제스처 모드 진단 테스트 시작. 종료: q 또는 ESC, 리셋: r")
    print(
        "왼손: rock/paper/scissors 스택 | " "오른손: 검지 조준 후보 + 선택 anchor 조준"
    )
    print(
        "핀치 발사 기준: "
        f"closed<={pinch_closed_threshold:.2f}, "
        f"velocity<={-pinch_close_velocity:.2f}, "
        f"delta>={pinch_close_delta:.2f}"
    )
    resolved_sensitivity_x = (
        aim_sensitivity if aim_sensitivity_x is None else aim_sensitivity_x
    )
    resolved_sensitivity_y = (
        aim_sensitivity if aim_sensitivity_y is None else aim_sensitivity_y
    )
    print(
        "조준 매핑: "
        f"anchor={aim_anchor}, "
        f"sx={resolved_sensitivity_x:.2f}, "
        f"sy={resolved_sensitivity_y:.2f}, "
        f"center=({aim_center_x:.2f}, {aim_center_y:.2f})"
    )
    if right_only:
        print("오른손 조준/핀치 발사 집중 모드: ON")
    if swap_handedness:
        print("MediaPipe handedness 좌우 라벨 보정: ON")
    else:
        print("MediaPipe handedness 좌우 라벨 보정: OFF")

    started_at = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame_started_at = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_started_at - started_at) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            observations, draw_items = _read_observations(result, swap_handedness)
            left, right = assign_hands(observations)

            for landmark_dicts, label in draw_items:
                _draw_landmarks(frame, landmark_dicts, color=_hand_color(label))
                _draw_hand_label(frame, landmark_dicts, label)

            left_states: np.ndarray | None = None
            right_states: np.ndarray | None = None
            left_candidate = None
            right_candidate = False
            fire_update: FireUpdate | None = None
            special_candidate = None

            if not right_only and left is not None:
                left_states = compute_finger_states(left.landmarks)
                left_candidate = classify_stack_gesture(left_states)
            stack_update = left_stack.update(left_candidate, timestamp=frame_started_at)
            if stack_update.emitted is not None:
                stack.append(stack_update.emitted)
                print(f"STACK: {stack_update.emitted} -> {' '.join(stack)}")

            aim_x: float | None = None
            aim_y: float | None = None
            pinch_distance: float | None = None
            if right is not None:
                right_states = compute_finger_states(right.landmarks)
                right_candidate = is_aim_pose(right_states)
                pinch_distance = normalized_pinch_distance(right.landmarks)

            aim_update = aim_mode.update(right_candidate, timestamp=frame_started_at)
            if right is not None:
                _draw_right_aim_debug(
                    frame,
                    right.landmarks,
                    is_candidate=right_candidate,
                    is_active=aim_update.active,
                    aim_anchor=aim_anchor,
                )

            if right is not None and aim_update.active:
                aim_x, aim_y = aim_tracker.update(
                    right.landmarks,
                    timestamp=frame_started_at,
                )
                last_aim = (aim_x, aim_y)
                cv2.drawMarker(
                    frame,
                    (int(aim_x), int(aim_y)),
                    RIGHT_HAND_COLOR,
                    cv2.MARKER_CROSS,
                    44,
                    3,
                )
                cv2.circle(
                    frame,
                    (int(aim_x), int(aim_y)),
                    24,
                    RIGHT_HAND_COLOR,
                    3,
                )
            elif last_aim is not None:
                cv2.drawMarker(
                    frame,
                    (int(last_aim[0]), int(last_aim[1])),
                    UNKNOWN_HAND_COLOR,
                    cv2.MARKER_CROSS,
                    20,
                    1,
                )

            fire_update = fire_detector.update(
                aim_active=aim_update.active,
                pinch_distance=pinch_distance,
                aim_x=aim_x,
                aim_y=aim_y,
                timestamp=frame_started_at,
            )
            if fire_update.fired and fire_update.fire_x is not None:
                shot_count += 1
                shots.append(
                    (
                        frame_started_at,
                        fire_update.fire_x,
                        fire_update.fire_y or 0.0,
                    )
                )
                print(
                    "PINCH FIRE "
                    f"#{shot_count}: x={fire_update.fire_x:.1f}, "
                    f"y={(fire_update.fire_y or 0.0):.1f}, "
                    f"pinch={fire_update.pinch_distance:.2f}, "
                    f"velocity={fire_update.pinch_velocity:.2f}"
                )

            special_active = two_hand_gate.update(
                left is not None and right is not None,
                timestamp=frame_started_at,
            )
            if left is not None and right is not None:
                special_candidate = classify_special_gesture(
                    left.landmarks,
                    right.landmarks,
                )
            special_update = special_debouncer.update(
                special_candidate,
                timestamp=frame_started_at,
            )
            if special_update.emitted is not None:
                print(f"SPECIAL: {special_update.emitted}")

            now = time.time()
            elapsed = now - frame_started_at
            fps_history.append(1.0 / max(elapsed, 1e-6))
            avg_fps = sum(fps_history) / len(fps_history)

            left_text = (
                f"cand={left_candidate or '-'} stable={stack_update.stable or '-'} "
                f"fingers={_finger_text(left_states)}"
            )
            if right_only:
                left_text = "disabled"
            right_text = (
                f"cand={'Y' if right_candidate else 'N'} "
                f"active={'Y' if aim_update.active else 'N'} "
                f"fingers={_finger_text(right_states)}"
            )

            _draw_shots(frame, shots, now)
            ui_panel = np.zeros((frame.shape[0], UI_PANEL_WIDTH, 3), dtype=np.uint8)
            _draw_status(
                ui_panel,
                stack_text=" ".join(stack),
                left_text=left_text,
                right_text=right_text,
                fire_update=fire_update,
                special_active=special_active,
                special_text=(
                    f"cand={special_candidate or '-'} "
                    f"stable={special_update.stable or '-'}"
                ),
                shot_count=shot_count,
                fps=avg_fps,
                pinch_closed_threshold=pinch_closed_threshold,
                aim_active=aim_update.active,
                has_right_hand=right is not None,
                pre_fire_seconds=DEFAULT_PRE_FIRE_SECONDS,
            )

            debug_view = np.hstack((frame, ui_panel))
            cv2.imshow("MotionMagic Gesture Mode Test", debug_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                left_stack.reset()
                aim_mode.reset()
                two_hand_gate.reset()
                special_debouncer.reset()
                aim_tracker.reset()
                fire_detector.reset()
                stack.clear()
                shots.clear()
                shot_count = 0
                last_aim = None
                print("리셋 완료")

    cap.release()
    cv2.destroyAllWindows()
    print("제스처 모드 진단 테스트 종료.")


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="MotionMagic 제스처 모드 진단 테스트")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 디바이스 ID (기본: 0)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="오른손 조준점 EMA 현재 입력 반영 비율 (기본: 0.3)",
    )
    parser.add_argument(
        "--aim-sensitivity",
        type=float,
        default=DEFAULT_AIM_SENSITIVITY,
        help="화면 중심 기준 조준 감도 (기본: 3.0)",
    )
    parser.add_argument(
        "--aim-sensitivity-x",
        type=float,
        default=None,
        help="X축 조준 감도. 미지정 시 --aim-sensitivity 사용",
    )
    parser.add_argument(
        "--aim-sensitivity-y",
        type=float,
        default=None,
        help="Y축 조준 감도. 미지정 시 --aim-sensitivity 사용",
    )
    parser.add_argument(
        "--aim-center-x",
        type=float,
        default=0.5,
        help="화면 중앙으로 매핑할 입력 X 좌표 (기본: 0.5)",
    )
    parser.add_argument(
        "--aim-center-y",
        type=float,
        default=0.5,
        help="화면 중앙으로 매핑할 입력 Y 좌표 (기본: 0.5)",
    )
    parser.add_argument(
        "--aim-anchor",
        choices=("index", "pinch"),
        default="index",
        help="조준점 기준: 검지 끝(index) 또는 엄지/검지 중심(pinch)",
    )
    parser.set_defaults(swap_handedness=True)
    parser.add_argument(
        "--no-swap-handedness",
        action="store_false",
        dest="swap_handedness",
        help="MediaPipe handedness 라벨을 그대로 사용",
    )
    parser.add_argument(
        "--right-only",
        action="store_true",
        help="오른손 조준/핀치 발사만 집중해서 표시",
    )
    parser.add_argument(
        "--pinch-open",
        type=float,
        default=DEFAULT_PINCH_OPEN_THRESHOLD,
        help="재장전으로 볼 엄지-검지 거리 (기본: 0.70)",
    )
    parser.add_argument(
        "--pinch-closed",
        type=float,
        default=DEFAULT_PINCH_CLOSED_THRESHOLD,
        help="발사로 볼 엄지-검지 닫힘 거리 (기본: 0.55)",
    )
    parser.add_argument(
        "--pinch-velocity",
        type=float,
        default=DEFAULT_PINCH_CLOSE_VELOCITY,
        help="발사로 볼 닫힘 속도 임계값 (기본: 0.60)",
    )
    parser.add_argument(
        "--pinch-delta",
        type=float,
        default=DEFAULT_PINCH_CLOSE_DELTA,
        help="발사로 볼 프레임 간 닫힘 변화량 (기본: 0.12)",
    )

    args = parser.parse_args()
    run_test(
        camera_id=args.camera,
        ema_alpha=args.ema_alpha,
        aim_sensitivity=args.aim_sensitivity,
        aim_sensitivity_x=args.aim_sensitivity_x,
        aim_sensitivity_y=args.aim_sensitivity_y,
        aim_center_x=args.aim_center_x,
        aim_center_y=args.aim_center_y,
        aim_anchor=args.aim_anchor,
        swap_handedness=args.swap_handedness,
        right_only=args.right_only,
        pinch_open_threshold=args.pinch_open,
        pinch_closed_threshold=args.pinch_closed,
        pinch_close_velocity=args.pinch_velocity,
        pinch_close_delta=args.pinch_delta,
    )


if __name__ == "__main__":
    main()
