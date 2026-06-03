"""규칙 기반 오탐 검증기.

CNN 모델의 분류 결과를 기하학적 뼈대 규칙으로 2차 검증하여
오탐(false positive)을 방지한다.
"""

from __future__ import annotations

import logging

import numpy as np

from src.ai.preprocessor import extract_finger_states

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 각 제스처의 예상 손가락 상태 패턴
# [엄지, 검지, 중지, 약지, 새끼] = 1(펴짐) / 0(접힘)
# ---------------------------------------------------------------------------

# rock: 모든 손가락 접힘
EXPECTED_ROCK = [0, 0, 0, 0, 0]

# paper: 모든 손가락 펴짐
EXPECTED_PAPER = [1, 1, 1, 1, 1]

# scissors: 검지+중지만 펴짐
EXPECTED_SCISSORS = [0, 1, 1, 0, 0]

# trigger: 엄지+검지만 펴짐 (권총 모양)
EXPECTED_TRIGGER = [1, 1, 0, 0, 0]

EXPECTED_PATTERNS: dict[str, list[int]] = {
    "rock": EXPECTED_ROCK,
    "paper": EXPECTED_PAPER,
    "scissors": EXPECTED_SCISSORS,
    "trigger": EXPECTED_TRIGGER,
}

# 허용 오차: 각 손가락에 대해 예상 패턴과 다를 수 있는 최대 개수
MAX_FINGER_MISMATCH = 1


def validate_gesture(
    gesture_label: str,
    landmarks: np.ndarray,
    confidence: float,
    min_confidence: float = 0.85,
) -> str | None:
    """CNN 분류 결과를 규칙 기반으로 2차 검증.

    Args:
        gesture_label: CNN이 예측한 제스처 라벨 (예: ``"rock"``).
        landmarks: ``(21, 2)`` 또는 ``(21, 3)`` 정규화된 좌표.
        confidence: CNN 모델의 예측 확신도 (0.0 ~ 1.0).
        min_confidence: 최소 신뢰도 임계값.

    Returns:
        검증을 통과한 제스처 라벨. 실패 시 ``None``.
    """
    # 1) 신뢰도 필터
    if confidence < min_confidence:
        logger.debug("신뢰도 미달: %.2f < %.2f → 무시", confidence, min_confidence)
        return None

    # 2) 알 수 없는 라벨 처리
    if gesture_label not in EXPECTED_PATTERNS:
        logger.warning("알 수 없는 제스처 라벨: '%s'", gesture_label)
        return None

    # 3) 손가락 상태 기반 기하학적 검증
    actual_states = extract_finger_states(landmarks)  # (5,)
    expected = EXPECTED_PATTERNS[gesture_label]

    mismatches = 0
    for i in range(len(expected)):
        if int(actual_states[i]) != expected[i]:
            mismatches += 1

    if mismatches > MAX_FINGER_MISMATCH:
        logger.debug(
            "규칙 검증 실패: '%s' 예상 %s ≠ 실제 %s (불일치 %d개)",
            gesture_label,
            expected,
            actual_states.tolist(),
            mismatches,
        )
        return None

    return gesture_label


def select_primary_hand(
    hands_data: list[dict],
) -> dict | None:
    """다중 손 감지 시 주 조작 손을 선택.

    양손 제스처 조건을 먼저 확인하고, 해당하지 않으면
    Y축이 가장 높은(화면 상단) 손을 선택한다.

    Args:
        hands_data: 감지된 손 정보 리스트. 각 원소는
            ``{"landmarks": np.ndarray, "score": float}`` 형태.

    Returns:
        선택된 손 정보 딕셔너리. 손이 없으면 ``None``.
    """
    if not hands_data:
        return None

    if len(hands_data) == 1:
        return hands_data[0]

    # 신뢰도 상위 2개만 유지
    sorted_by_score = sorted(hands_data, key=lambda h: h["score"], reverse=True)
    top_two = sorted_by_score[:2]

    # Y축이 더 높은(값이 작은 = 화면 위쪽) 손 선택
    primary = min(
        top_two,
        key=lambda h: h["landmarks"][0][1],  # 손목 Y좌표 기준
    )

    return primary
