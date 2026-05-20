"""웹캠 제스처 데이터 수집기.

MediaPipe Hand Landmarker를 사용하여 실시간 웹캠 프레임에서
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

# MediaPipe Hand Landmarker 설정
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles

# 키보드 바인딩: 숫자 키로 라벨 지정
KEY_BINDINGS: dict[int, str] = {
    ord("0"): "rock",
    ord("1"): "paper",
    ord("2"): "scissors",
    ord("3"): "trigger",
    ord("4"): "idle",
}


# ---------------------------------------------------------------------------
# 수집기 클래스
# ---------------------------------------------------------------------------


class GestureCollector:
    """실시간 웹캠으로 제스처 학습 데이터를 수집.

    웹캠 화면에 MediaPipe 손 추적 결과를 오버레이하여 보여주고,
    키보드 입력으로 라벨을 지정하여 JSON 파일에 저장한다.

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
        self._current_label: str | None = None

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
            auto_capture: ``True``이면 자동 캡처 모드 (일정 간격마다 저장).
            capture_interval_ms: 자동 캡처 시 프레임 간격(밀리초).

        Returns:
            저장된 JSON 파일 경로.

        Raises:
            ValueError: 유효하지 않은 제스처 라벨일 경우.
        """
        if gesture not in LABEL_TO_INDEX:
            valid = ", ".join(LABEL_TO_INDEX.keys())
            raise ValueError(
                f"유효하지 않은 제스처: '{gesture}'. "
                f"유효한 라벨: {valid}"
            )

        if output_dir is None:
            output_dir = RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self._current_label = gesture
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

        with MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=0.5,
        ) as hands:
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
                results = hands.process(rgb)

                landmarks_data = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # 화면에 랜드마크 그리기
                    MP_DRAWING.draw_landmarks(
                        frame,
                        hand_landmarks,
                        MP_HANDS.HAND_CONNECTIONS,
                        MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                        MP_DRAWING_STYLES.get_default_hand_connections_style(),
                    )

                    # 좌표 추출
                    landmarks_data = [
                        {"x": lm.x, "y": lm.y}
                        for lm in hand_landmarks.landmark
                    ]

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
                    logger.info("사용자가 수집을 중단함 (%d/%d)", collected, num_samples)
                    break

                if not auto_capture and key == ord("c") and landmarks_data:
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
            "데이터 저장 완료: %s (%d samples)", filepath, len(self._samples)
        )
        print(f"\n[완료] {len(self._samples)}개 샘플 → {filepath}")

        return filepath
