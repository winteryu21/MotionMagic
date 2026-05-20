"""웹캠 제스처 데이터 수집기.

MediaPipe HandLandmarker (Tasks API)를 사용하여 실시간 웹캠 프레임에서
21개 손 관절 좌표를 추출하고 JSON 파일로 저장한다.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.ai.preprocessor import (
    GESTURE_LABELS,
    LABEL_TO_INDEX,
    NUM_LANDMARKS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
HAND_LANDMARKER_MODEL = Path("models/hand_landmarker.task")

# MediaPipe Tasks API
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode as VisionRunningMode,
)

# 랜드마크 연결 정보 (시각화용)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),       # 검지
    (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
    (0, 13), (13, 14), (14, 15), (15, 16), # 약지
    (0, 17), (17, 18), (18, 19), (19, 20), # 새끼
    (5, 9), (9, 13), (13, 17),             # 손바닥
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
        if gesture not in LABEL_TO_INDEX:
            valid = ", ".join(LABEL_TO_INDEX.keys())
            raise ValueError(
                f"유효하지 않은 제스처: '{gesture}'. "
                f"유효한 라벨: {valid}"
            )

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

        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"카메라(ID={self._camera_id})를 열 수 없습니다."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # HandLandmarker 옵션 설정
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_MODEL)
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self._max_num_hands,
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
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect(mp_image)

                landmarks_data = None
                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0]

                    # 좌표 추출
                    landmarks_data = [
                        {"x": lm.x, "y": lm.y}
                        for lm in hand_lms
                    ]

                    # 화면에 랜드마크 그리기
                    _draw_landmarks(frame, landmarks_data)

                # 자동/수동 캡처 로직
                now = time.time() * 1000
                should_capture = False
                if auto_capture:
                    if now - last_capture_time >= capture_interval_ms:
                        should_capture = True
                        last_capture_time = now

                # 화면 표시
                status = f"[{gesture}] {collected}/{num_samples}"
                color = (0, 255, 0) if landmarks_data else (0, 0, 255)
                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                if landmarks_data and should_capture:
                    self._samples.append(
                        {
                            "label": gesture,
                            "landmarks": [
                                [lm["x"], lm["y"]]
                                for lm in landmarks_data
                            ],
                        }
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

                if (
                    not auto_capture
                    and key == ord("c")
                    and landmarks_data
                ):
                    self._samples.append(
                        {
                            "label": gesture,
                            "landmarks": [
                                [lm["x"], lm["y"]]
                                for lm in landmarks_data
                            ],
                        }
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
