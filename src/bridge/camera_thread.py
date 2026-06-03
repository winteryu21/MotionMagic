"""웹캠 캡처 + 제스처 추론 백그라운드 스레드."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision import (
    RunningMode as VisionRunningMode,
)

from src.ai.aim_tracker import AimAnchor
from src.ai.collector import HAND_LANDMARKER_MODEL, _draw_landmarks
from src.ai.gesture_modes import HandLabel, HandObservation
from src.ai.preprocessor import extract_landmarks
from src.bridge.gesture_event import GestureEvent
from src.bridge.gesture_mode_pipeline import (
    DEFAULT_AIM_SENSITIVITY,
    DEFAULT_EMA_ALPHA,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    GestureModePipeline,
    GestureModePipelineConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_HAND_DETECTION_CONFIDENCE = 0.7
DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class CameraDebugFrame:
    """게임 디버그 오버레이에 표시할 카메라 프레임.

    Attributes:
        rgb_bytes: RGB24 픽셀 바이트.
        width: 프레임 너비.
        height: 프레임 높이.
    """

    rgb_bytes: bytes
    width: int
    height: int


class CameraThread:
    """웹캠 캡처와 제스처 모드 판정을 별도 스레드에서 실행.

    게임 루프의 FPS 저하를 방지하기 위해 인식 파이프라인을
    백그라운드에서 동작시키고, 결과를 콜백으로 전달한다.

    Args:
        on_gesture: 제스처 인식 시 호출될 콜백 함수.
        camera_id: 카메라 디바이스 ID.
        frame_width: 카메라 입력 너비.
        frame_height: 카메라 입력 높이.
        mirror: 플레이어 기준 조작을 위해 프레임을 좌우 반전할지 여부.
        swap_handedness: 좌우 반전된 프레임의 handedness 라벨 보정 여부.
        ema_alpha: 오른손 조준점 EMA 현재 입력 반영 비율.
        aim_sensitivity: 화면 중심 기준 조준 감도.
        aim_sensitivity_x: X축 조준 감도.
        aim_sensitivity_y: Y축 조준 감도.
        aim_center_x: 화면 중앙으로 매핑할 입력 X 좌표.
        aim_center_y: 화면 중앙으로 매핑할 입력 Y 좌표.
        aim_anchor: 조준점 anchor. ``"index"`` 또는 ``"pinch"``.
    """

    def __init__(
        self,
        on_gesture: Callable[[GestureEvent], None],
        camera_id: int = 0,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        mirror: bool = True,
        swap_handedness: bool = True,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        aim_sensitivity: float = DEFAULT_AIM_SENSITIVITY,
        aim_sensitivity_x: float | None = None,
        aim_sensitivity_y: float | None = None,
        aim_center_x: float = 0.5,
        aim_center_y: float = 0.5,
        aim_anchor: AimAnchor = "index",
    ) -> None:
        self._on_gesture = on_gesture
        self._camera_id = camera_id
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._mirror = mirror
        self._swap_handedness = swap_handedness
        self._pipeline = GestureModePipeline(
            GestureModePipelineConfig(
                frame_width=frame_width,
                frame_height=frame_height,
                ema_alpha=ema_alpha,
                aim_sensitivity=aim_sensitivity,
                aim_sensitivity_x=aim_sensitivity_x,
                aim_sensitivity_y=aim_sensitivity_y,
                aim_center_x=aim_center_x,
                aim_center_y=aim_center_y,
                aim_anchor=aim_anchor,
            )
        )
        self._thread: threading.Thread | None = None
        self._running = False
        self._debug_lock = threading.Lock()
        self._debug_frame: CameraDebugFrame | None = None

    def start(self) -> None:
        """백그라운드 스레드 시작."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("카메라 스레드 시작")

    def stop(self) -> None:
        """백그라운드 스레드 중지."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("카메라 스레드 중지")

    def get_debug_frame(self) -> CameraDebugFrame | None:
        """최근 카메라 디버그 프레임을 반환한다.

        Returns:
            가장 최근 RGB 디버그 프레임. 아직 프레임이 없으면 ``None``.
        """
        with self._debug_lock:
            return self._debug_frame

    def _run(self) -> None:
        """스레드 메인 루프. MediaPipe + 모드 상태 머신을 수행."""
        if not HAND_LANDMARKER_MODEL.exists():
            logger.error("HandLandmarker 모델 파일 없음: %s", HAND_LANDMARKER_MODEL)
            self._running = False
            return

        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            logger.error("카메라(ID=%d)를 열 수 없습니다.", self._camera_id)
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=DEFAULT_MIN_TRACKING_CONFIDENCE,
        )

        started_at = time.time()
        try:
            with HandLandmarker.create_from_options(options) as landmarker:
                while self._running:
                    frame_started_at = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("카메라 프레임 읽기 실패")
                        continue

                    if self._mirror:
                        frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((frame_started_at - started_at) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    self._update_debug_frame(frame, result)
                    observations = self._build_observations(result)
                    events = self._pipeline.update(
                        observations,
                        timestamp=frame_started_at,
                    )
                    for event in events:
                        self._emit(event)
        finally:
            cap.release()
            self._running = False

    def _build_observations(self, result: object) -> list[HandObservation]:
        """MediaPipe 결과를 손 채널 관측값으로 변환한다.

        Args:
            result: HandLandmarker 결과.

        Returns:
            손 채널 관측값 목록.
        """
        hand_landmarks = getattr(result, "hand_landmarks", None)
        if not hand_landmarks:
            return []

        observations: list[HandObservation] = []
        for hand_index, landmarks in enumerate(hand_landmarks):
            raw_label, score = self._extract_handedness(result, hand_index)
            label = self._effective_handedness(raw_label)
            if label is None:
                continue

            landmark_dicts = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in landmarks
            ]
            observations.append(
                HandObservation(
                    label=label,
                    landmarks=extract_landmarks(landmark_dicts),
                    score=score,
                )
            )

        return observations

    def _update_debug_frame(self, frame: object, result: object) -> None:
        """랜드마크를 그린 최신 카메라 프레임을 저장한다.

        Args:
            frame: OpenCV BGR 이미지.
            result: HandLandmarker 결과.
        """
        debug_frame = frame.copy()
        hand_landmarks = getattr(result, "hand_landmarks", None)
        if hand_landmarks:
            for hand_index, landmarks in enumerate(hand_landmarks):
                raw_label, _score = self._extract_handedness(result, hand_index)
                label = self._effective_handedness(raw_label)
                landmark_dicts = [
                    {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                    for lm in landmarks
                ]
                color = (0, 255, 0) if label == "Left" else (255, 180, 0)
                _draw_landmarks(debug_frame, landmark_dicts, color=color)
                self._draw_debug_label(debug_frame, landmark_dicts, label)

        rgb_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        height, width = rgb_frame.shape[:2]
        snapshot = CameraDebugFrame(
            rgb_bytes=rgb_frame.tobytes(),
            width=int(width),
            height=int(height),
        )
        with self._debug_lock:
            self._debug_frame = snapshot

    def _draw_debug_label(
        self,
        frame: object,
        landmarks: list[dict[str, float]],
        label: HandLabel | None,
    ) -> None:
        """디버그 프레임에 손 라벨을 그린다.

        Args:
            frame: OpenCV BGR 이미지.
            landmarks: 손 랜드마크 좌표 목록.
            label: 보정된 handedness 라벨.
        """
        if not landmarks:
            return

        height, width = frame.shape[:2]
        wrist = landmarks[0]
        x = max(0, min(width - 1, int(wrist["x"] * width)))
        y = max(20, min(height - 1, int(wrist["y"] * height)))
        cv2.putText(
            frame,
            label or "?",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _extract_handedness(
        self,
        result: object,
        hand_index: int,
    ) -> tuple[str | None, float]:
        """MediaPipe 결과에서 handedness 라벨과 점수를 추출한다.

        Args:
            result: HandLandmarker 결과.
            hand_index: 손 인덱스.

        Returns:
            ``(label, score)`` 튜플.
        """
        handedness = getattr(result, "handedness", None)
        if (
            not handedness
            or hand_index >= len(handedness)
            or not handedness[hand_index]
        ):
            return None, 0.0

        category = handedness[hand_index][0]
        return str(category.category_name), float(category.score)

    def _effective_handedness(self, label: str | None) -> HandLabel | None:
        """실사용 기준 handedness 라벨로 보정한다.

        Args:
            label: MediaPipe handedness 라벨.

        Returns:
            ``"Left"``, ``"Right"`` 또는 ``None``.
        """
        if label not in {"Left", "Right"}:
            return None

        if self._mirror and self._swap_handedness:
            label = "Right" if label == "Left" else "Left"
        return label

    def _emit(self, event: GestureEvent) -> None:
        """콜백 예외가 카메라 루프를 죽이지 않도록 이벤트를 전달한다.

        Args:
            event: bridge 제스처 이벤트.
        """
        try:
            self._on_gesture(event)
        except Exception:
            logger.exception("제스처 이벤트 콜백 처리 실패: %s", event)
