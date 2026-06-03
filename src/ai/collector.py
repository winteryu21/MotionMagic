"""웹캠 제스처 데이터 수집기.

MediaPipe HandLandmarker (Tasks API)를 사용하여 실시간 웹캠 프레임에서
21개 손 관절 좌표를 추출하고 JSON 파일로 저장한다.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.vision import (
    RunningMode as VisionRunningMode,
)

from src.ai.preprocessor import LABEL_TO_INDEX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
HAND_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
TWO_HAND_GESTURES = {"clasp", "sonaldo"}
RESUME_COUNTDOWN_SECONDS = 3.0
COUNTDOWN_FONT_SCALE = 2.6
COUNTDOWN_FONT_THICKNESS = 4
COUNTDOWN_COLOR = (0, 255, 255)

# 랜드마크 연결 정보 (시각화용)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 엄지
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 검지
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # 중지
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # 약지
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 새끼
    (5, 9),
    (9, 13),
    (13, 17),  # 손바닥
]


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: list[dict[str, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """프레임에 랜드마크 점과 연결선을 그린다.

    Args:
        frame: BGR 이미지.
        landmarks: ``[{"x": ..., "y": ...}, ...]`` 형태의 좌표 리스트.
        color: 선 색상 (BGR).
        thickness: 선 두께.
    """
    h, w = frame.shape[:2]
    points = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, thickness)

    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def _draw_resume_countdown(frame: np.ndarray, seconds_left: int) -> None:
    """수집 재개 전 카운트다운을 프레임 중앙에 표시한다.

    Args:
        frame: BGR 이미지.
        seconds_left: 화면에 표시할 남은 초.
    """
    text = str(seconds_left)
    height, width = frame.shape[:2]
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        COUNTDOWN_FONT_SCALE,
        COUNTDOWN_FONT_THICKNESS,
    )
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        COUNTDOWN_FONT_SCALE,
        COUNTDOWN_COLOR,
        COUNTDOWN_FONT_THICKNESS,
    )


def _get_handedness(result: object, hand_index: int) -> tuple[str | None, float | None]:
    """MediaPipe 결과에서 손 방향 라벨과 점수를 추출한다.

    Args:
        result: ``HandLandmarker.detect`` 결과.
        hand_index: 손 인덱스.

    Returns:
        ``(handedness, score)`` 튜플. 정보가 없으면 ``(None, None)``.
    """
    handedness = getattr(result, "handedness", None)
    if not handedness or hand_index >= len(handedness) or not handedness[hand_index]:
        return None, None

    category = handedness[hand_index][0]
    return category.category_name, float(category.score)


def _serialize_landmarks(landmarks: list[dict[str, float]]) -> list[list[float]]:
    """랜드마크 딕셔너리 리스트를 JSON 저장용 배열로 변환한다.

    Args:
        landmarks: ``[{"x": ..., "y": ..., "z": ...}, ...]`` 좌표 리스트.

    Returns:
        ``[[x, y, z], ...]`` 배열.
    """
    return [[lm["x"], lm["y"], lm["z"]] for lm in landmarks]


def _build_sample(
    gesture: str,
    hands_data: list[dict],
    is_two_hand_gesture: bool,
) -> dict:
    """수집된 손 데이터를 원시 JSON 샘플로 변환한다.

    Args:
        gesture: 제스처 라벨.
        hands_data: 감지된 손 데이터 리스트.
        is_two_hand_gesture: 양손 제스처 여부.

    Returns:
        JSON 저장용 샘플 딕셔너리.
    """
    if is_two_hand_gesture:
        return {
            "label": gesture,
            "hands": [
                {
                    "handedness": hand["handedness"],
                    "handedness_score": hand["handedness_score"],
                    "landmarks": _serialize_landmarks(hand["landmarks"]),
                }
                for hand in hands_data[:2]
            ],
        }

    hand = hands_data[0]
    return {
        "label": gesture,
        "handedness": hand["handedness"],
        "handedness_score": hand["handedness_score"],
        "landmarks": _serialize_landmarks(hand["landmarks"]),
    }


# ---------------------------------------------------------------------------
# 수집기 클래스
# ---------------------------------------------------------------------------


class GestureCollector:
    """실시간 웹캠으로 제스처 학습 데이터를 수집.

    웹캠 화면에 MediaPipe HandLandmarker 결과를 오버레이하여 보여주고,
    자동 캡처 모드로 JSON 파일에 저장한다.

    Args:
        camera_id: 웹캠 디바이스 ID (기본 0).
        max_num_hands: 최대 감지 손 개수 (기본 1).
        min_detection_confidence: 최소 감지 신뢰도 (기본 0.7).
    """

    def __init__(
        self,
        camera_id: int = 0,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
    ) -> None:
        self._camera_id = camera_id
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._samples: list[dict] = []

    def collect(
        self,
        gesture: str,
        num_samples: int = 1000,
        output_dir: Path | None = None,
        auto_capture: bool = True,
        capture_interval_ms: int = 100,
    ) -> Path:
        """특정 제스처에 대한 학습 데이터를 수집.

        Args:
            gesture: 수집할 제스처 라벨 (``"rock"``, ``"paper"`` 등).
            num_samples: 수집할 프레임 수.
            output_dir: 저장 디렉터리. ``None``이면 ``data/raw/``.
            auto_capture: ``True``이면 자동 캡처 모드.
            capture_interval_ms: 자동 캡처 시 프레임 간격(밀리초).

        Returns:
            저장된 JSON 파일 경로.

        Raises:
            ValueError: 유효하지 않은 제스처 라벨일 경우.
            FileNotFoundError: HandLandmarker 모델 파일이 없는 경우.
        """
        valid_labels = set(LABEL_TO_INDEX) | TWO_HAND_GESTURES
        if gesture not in valid_labels:
            valid = ", ".join(sorted(valid_labels))
            raise ValueError(
                f"유효하지 않은 제스처: '{gesture}'. " f"유효한 라벨: {valid}"
            )

        is_two_hand_gesture = gesture in TWO_HAND_GESTURES
        if is_two_hand_gesture:
            num_hands = max(self._max_num_hands, 2)
        else:
            num_hands = self._max_num_hands

        if not HAND_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"HandLandmarker 모델 파일 없음: {HAND_LANDMARKER_MODEL}\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/"
                "hand_landmarker.task 에서 다운로드하세요."
            )

        if output_dir is None:
            output_dir = RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self._samples = []
        collected = 0
        last_capture_time = 0.0
        paused = False
        resume_at: float | None = None

        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"카메라(ID={self._camera_id})를 열 수 없습니다.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # HandLandmarker 옵션 설정
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            logger.info(
                "데이터 수집 시작: gesture='%s', target=%d",
                gesture,
                num_samples,
            )
            print(f"\n[수집 모드] 제스처: {gesture}")
            print(f"  목표 프레임: {num_samples}")
            print(f"  자동 캡처: {'켜짐' if auto_capture else '꺼짐'}")
            print("  일시정지/재개: Space (재개 전 3초 카운트다운)")
            print("  종료: 'q' 키 또는 ESC")
            if not auto_capture:
                print("  수동 캡처: 'c' 키")

            while collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("프레임 읽기 실패")
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe Tasks API로 감지
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                hands_data: list[dict] = []
                if result.hand_landmarks:
                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        hand_label, hand_score = _get_handedness(result, idx)

                        # 좌표 추출
                        landmarks_data = [
                            {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lms
                        ]

                        hands_data.append(
                            {
                                "landmarks": landmarks_data,
                                "handedness": hand_label,
                                "handedness_score": hand_score,
                            }
                        )

                        # 화면에 랜드마크 그리기
                        _draw_landmarks(frame, landmarks_data)

                now_seconds = time.time()
                if resume_at is not None and now_seconds >= resume_at:
                    resume_at = None
                    last_capture_time = now_seconds * 1000
                    print("  수집 재개")

                capture_blocked = paused or resume_at is not None

                # 자동/수동 캡처 로직
                now = now_seconds * 1000
                should_capture = False
                if auto_capture and not capture_blocked:
                    if now - last_capture_time >= capture_interval_ms:
                        should_capture = True
                        last_capture_time = now

                # 화면 표시
                status = f"[{gesture}] {collected}/{num_samples}"
                if paused:
                    status += " PAUSED"
                elif resume_at is not None:
                    seconds_left = max(1, math.ceil(resume_at - now_seconds))
                    status += f" RESUME IN {seconds_left}"
                can_capture = (
                    len(hands_data) >= 2 if is_two_hand_gesture else bool(hands_data)
                )
                if capture_blocked:
                    color = COUNTDOWN_COLOR
                else:
                    color = (0, 255, 0) if can_capture else (0, 0, 255)
                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                if resume_at is not None:
                    _draw_resume_countdown(frame, seconds_left)

                if can_capture and should_capture:
                    self._samples.append(
                        _build_sample(gesture, hands_data, is_two_hand_gesture)
                    )
                    collected += 1

                    # 캡처 표시
                    cv2.circle(frame, (620, 25), 10, (0, 0, 255), -1)

                cv2.imshow("MotionMagic Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # q 또는 ESC
                    logger.info(
                        "사용자가 수집을 중단함 (%d/%d)",
                        collected,
                        num_samples,
                    )
                    break

                if key == ord(" "):
                    if paused:
                        paused = False
                        resume_at = time.time() + RESUME_COUNTDOWN_SECONDS
                        print(f"  {RESUME_COUNTDOWN_SECONDS:.0f}초 후 수집 재개")
                    else:
                        paused = True
                        resume_at = None
                        print("  수집 일시정지")

                if (
                    not capture_blocked
                    and not auto_capture
                    and key == ord("c")
                    and can_capture
                ):
                    self._samples.append(
                        _build_sample(gesture, hands_data, is_two_hand_gesture)
                    )
                    collected += 1

        cap.release()
        cv2.destroyAllWindows()

        # JSON 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._samples, f, ensure_ascii=False, indent=2)

        logger.info(
            "데이터 저장 완료: %s (%d samples)",
            filepath,
            len(self._samples),
        )
        print(f"\n[완료] {len(self._samples)}개 샘플 → {filepath}")

        return filepath
