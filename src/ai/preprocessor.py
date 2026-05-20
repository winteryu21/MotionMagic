"""데이터 전처리 및 정규화.

MediaPipe에서 추출한 손 관절 좌표를 위치·크기 불변 형태로
정규화하고, 5차원 손가락 상태 힌트를 기하학적으로 도출한다.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 21
NUM_COORDS = 2  # X, Y만 사용 (Z축 제외)
NUM_FINGERS = 5

# 손가락별 (팁 인덱스, MCP/뿌리 인덱스) 매핑
# 엄지: tip=4, base=2 | 검지: tip=8, base=5 | 중지: tip=12, base=9
# 약지: tip=16, base=13 | 새끼: tip=20, base=17
FINGER_TIP_IDS = [4, 8, 12, 16, 20]
FINGER_MCP_IDS = [2, 5, 9, 13, 17]

# 손가락이 펴졌다고 판정하기 위한 거리 비율 임계값
FINGER_EXTENDED_THRESHOLD = 1.3

# 크기 정규화 기준: 손목(0) → 중지 시작(9)
SCALE_REF_START = 0
SCALE_REF_END = 9

# 제스처 라벨 매핑
GESTURE_LABELS: dict[int, str] = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "trigger",
    4: "idle",
}

LABEL_TO_INDEX: dict[str, int] = {v: k for k, v in GESTURE_LABELS.items()}
NUM_CLASSES = len(GESTURE_LABELS)


# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """두 2D 점 사이의 유클리드 거리.

    Args:
        a: 첫 번째 점 ``(2,)``.
        b: 두 번째 점 ``(2,)``.

    Returns:
        유클리드 거리 스칼라 값.
    """
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# 핵심 전처리 함수
# ---------------------------------------------------------------------------


def extract_2d_landmarks(
    landmarks_raw: list[dict[str, float]] | np.ndarray,
) -> np.ndarray:
    """MediaPipe 랜드마크에서 X, Y 2D 좌표만 추출.

    Args:
        landmarks_raw: 21개 관절의 좌표 목록.
            ``[{"x": ..., "y": ..., "z": ...}, ...]`` 딕셔너리 리스트
            또는 ``(21, 2)`` 혹은 ``(21, 3)`` NumPy 배열.

    Returns:
        ``(21, 2)`` NumPy 배열.
    """
    if isinstance(landmarks_raw, np.ndarray):
        return landmarks_raw[:, :NUM_COORDS].copy()

    coords = [[lm["x"], lm["y"]] for lm in landmarks_raw]
    return np.array(coords, dtype=np.float32)  # (21, 2)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    """손목 기준 평행 이동 + 크기 정규화.

    Args:
        landmarks: ``(21, 2)`` 정규화 전 좌표.

    Returns:
        ``(21, 2)`` 정규화된 좌표. 기준 거리가 0이면 ``None``.
    """
    wrist = landmarks[SCALE_REF_START].copy()  # (2,)

    # 1) 평행 이동: 손목을 원점으로
    centered = landmarks - wrist  # (21, 2)

    # 2) 크기 정규화: 손목→중지MCP 거리를 기준척도로
    ref_dist = _euclidean(
        centered[SCALE_REF_START], centered[SCALE_REF_END]
    )
    if ref_dist < 1e-6:
        logger.warning("기준 거리가 0에 가까워 정규화 불가")
        return None

    normalized = centered / ref_dist  # (21, 2)
    return normalized.astype(np.float32)


def extract_finger_states(landmarks: np.ndarray) -> np.ndarray:
    """5차원 손가락 상태(펴짐/접힘) 힌트를 기하학적으로 도출.

    손목(Landmark 0)에서 손가락 끝(Tip)까지의 거리와
    손목에서 손가락 뿌리(MCP)까지의 거리의 비율을 계산한다.
    비율이 임계값 이상이면 펴진 상태(1), 미만이면 접힌 상태(0).

    Args:
        landmarks: ``(21, 2)`` 정규화된 좌표 (손목 원점 기준).

    Returns:
        ``(5,)`` 배열. 각 원소는 0(접힘) 또는 1(펴짐).
    """
    wrist = landmarks[0]
    states = np.zeros(NUM_FINGERS, dtype=np.float32)

    for i, (tip_id, mcp_id) in enumerate(
        zip(FINGER_TIP_IDS, FINGER_MCP_IDS)
    ):
        tip_dist = _euclidean(wrist, landmarks[tip_id])
        mcp_dist = _euclidean(wrist, landmarks[mcp_id])

        if mcp_dist < 1e-6:
            states[i] = 0.0
            continue

        ratio = tip_dist / mcp_dist
        states[i] = 1.0 if ratio >= FINGER_EXTENDED_THRESHOLD else 0.0

    return states  # (5,)


def augment_rotation(
    landmarks: np.ndarray,
    max_angle_deg: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """랜덤 2D 회전 데이터 증강.

    원점(손목) 기준으로 ``[-max_angle_deg, +max_angle_deg]``
    범위에서 무작위 각도로 회전한다.

    Args:
        landmarks: ``(21, 2)`` 정규화된 좌표.
        max_angle_deg: 최대 회전 각도(도 단위).
        rng: NumPy 랜덤 생성기. ``None``이면 새로 생성.

    Returns:
        ``(21, 2)`` 회전 적용된 좌표.
    """
    if rng is None:
        rng = np.random.default_rng()

    angle_rad = rng.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180.0
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
    )  # (2, 2)

    return (landmarks @ rotation_matrix.T).astype(np.float32)  # (21, 2)


def augment_noise(
    landmarks: np.ndarray,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """가우시안 노이즈 데이터 증강.

    Args:
        landmarks: ``(21, 2)`` 정규화된 좌표.
        noise_std: 노이즈 표준편차.
        rng: NumPy 랜덤 생성기.

    Returns:
        ``(21, 2)`` 노이즈가 추가된 좌표.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, noise_std, size=landmarks.shape).astype(
        np.float32
    )
    return landmarks + noise


def preprocess_single(
    landmarks_raw: list[dict[str, float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """단일 프레임의 원시 랜드마크를 전처리 피처로 변환.

    Returns:
        ``(normalized_landmarks (21, 2), finger_states (5,))`` 튜플.
        정규화에 실패하면 ``None``.
    """
    coords_2d = extract_2d_landmarks(landmarks_raw)
    normalized = normalize_landmarks(coords_2d)
    if normalized is None:
        return None

    finger_states = extract_finger_states(normalized)
    return normalized, finger_states
