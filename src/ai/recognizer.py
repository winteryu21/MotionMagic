"""ONNX Runtime 기반 실시간 제스처 인식기.

학습 완료 후 내보낸 ``.onnx`` 모델을 로드하여
MediaPipe 프레임 스트림으로부터 초저지연 추론을 수행한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.ai.preprocessor import (
    GESTURE_LABELS,
    NUM_CLASSES,
    extract_2d_landmarks,
    extract_finger_states,
    normalize_landmarks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path("models/gesture_cnn.onnx")
DEFAULT_CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# 인식기 클래스
# ---------------------------------------------------------------------------


class GestureRecognizer:
    """ONNX Runtime 기반 실시간 제스처 인식기.

    ``.onnx`` 모델 파일을 로드하고, 매 프레임의 랜드마크를
    전처리한 뒤 초고속 추론을 수행한다.

    Args:
        model_path: ONNX 모델 파일 경로.
        confidence_threshold: 최소 예측 신뢰도 임계값.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold
        self._session = None

        self._load_model()

    def _load_model(self) -> None:
        """ONNX 모델을 ONNX Runtime 세션으로 로드."""
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX 모델 로드 완료: %s", self._model_path)
        except ImportError:
            logger.error(
                "onnxruntime이 설치되지 않았습니다. "
                "'pip install onnxruntime'을 실행해주세요."
            )
            raise
        except Exception as e:
            logger.error("ONNX 모델 로드 실패: %s", e)
            raise

    def predict(
        self,
        landmarks_raw: list[dict[str, float]] | np.ndarray,
    ) -> tuple[str, float] | None:
        """단일 프레임 제스처 예측.

        Args:
            landmarks_raw: 21개 관절 좌표 (MediaPipe 원시 출력).

        Returns:
            ``(제스처_라벨, 신뢰도)`` 튜플. 인식 실패 시 ``None``.
        """
        if self._session is None:
            logger.warning("ONNX 세션이 초기화되지 않았습니다.")
            return None

        # 전처리
        coords_2d = extract_2d_landmarks(landmarks_raw)  # (21, 2)
        normalized = normalize_landmarks(coords_2d)
        if normalized is None:
            return None

        finger_states = extract_finger_states(normalized)  # (5,)

        # 배치 차원 추가: (1, 21, 2), (1, 5)
        landmarks_input = normalized[np.newaxis, :, :].astype(np.float32)
        fingers_input = finger_states[np.newaxis, :].astype(np.float32)

        # ONNX 추론
        input_names = [inp.name for inp in self._session.get_inputs()]
        outputs = self._session.run(
            None,
            {
                input_names[0]: landmarks_input,
                input_names[1]: fingers_input,
            },
        )

        logits = outputs[0][0]  # (num_classes,)

        # Softmax 적용
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # 최고 확률 클래스
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if confidence < self._confidence_threshold:
            logger.debug(
                "신뢰도 미달: %.2f < %.2f",
                confidence,
                self._confidence_threshold,
            )
            return None

        gesture_label = GESTURE_LABELS.get(pred_idx, "idle")
        return gesture_label, confidence
