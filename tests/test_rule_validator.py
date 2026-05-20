"""rule_validator 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.rule_validator import (
    select_primary_hand,
    validate_gesture,
)


# ---------------------------------------------------------------------------
# 테스트 유틸
# ---------------------------------------------------------------------------


def _make_open_hand() -> np.ndarray:
    """모든 손가락이 펴진 손 (paper용)."""
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = [0.0, 0.0]
    lm[2] = [0.5, -0.2]
    lm[4] = [0.9, -0.4]
    lm[5] = [0.2, -0.4]
    lm[8] = [0.2, -1.0]
    lm[9] = [0.0, -0.5]
    lm[12] = [0.0, -1.1]
    lm[13] = [-0.2, -0.4]
    lm[16] = [-0.2, -1.0]
    lm[17] = [-0.4, -0.3]
    lm[20] = [-0.4, -0.9]
    return lm


def _make_fist() -> np.ndarray:
    """주먹 (rock용)."""
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = [0.0, 0.0]
    lm[2] = [0.4, -0.2]
    lm[4] = [0.3, -0.1]
    lm[5] = [0.2, -0.4]
    lm[8] = [0.15, -0.25]
    lm[9] = [0.0, -0.5]
    lm[12] = [0.0, -0.3]
    lm[13] = [-0.2, -0.4]
    lm[16] = [-0.15, -0.25]
    lm[17] = [-0.4, -0.3]
    lm[20] = [-0.3, -0.2]
    return lm


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


class TestValidateGesture:
    """validate_gesture 함수 테스트."""

    def test_low_confidence_rejected(self) -> None:
        """신뢰도가 임계값 미만이면 None 반환."""
        lm = _make_open_hand()
        result = validate_gesture("paper", lm, confidence=0.5, min_confidence=0.85)
        assert result is None

    def test_paper_with_open_hand_accepted(self) -> None:
        """paper 제스처 + 펴진 손 → 통과."""
        lm = _make_open_hand()
        result = validate_gesture("paper", lm, confidence=0.95)
        assert result == "paper"

    def test_rock_with_fist_accepted(self) -> None:
        """rock 제스처 + 주먹 → 통과."""
        lm = _make_fist()
        result = validate_gesture("rock", lm, confidence=0.95)
        assert result == "rock"

    def test_rock_with_open_hand_rejected(self) -> None:
        """rock 제스처 + 펴진 손 → 불일치로 거부."""
        lm = _make_open_hand()
        result = validate_gesture("rock", lm, confidence=0.95)
        assert result is None

    def test_idle_always_passes(self) -> None:
        """idle은 규칙 검증 없이 통과."""
        lm = _make_open_hand()
        result = validate_gesture("idle", lm, confidence=0.90)
        assert result == "idle"

    def test_unknown_gesture_rejected(self) -> None:
        """알 수 없는 제스처 라벨 → None."""
        lm = _make_open_hand()
        result = validate_gesture("fireball", lm, confidence=0.99)
        assert result is None


class TestSelectPrimaryHand:
    """select_primary_hand 함수 테스트."""

    def test_empty_list(self) -> None:
        """빈 리스트 → None."""
        assert select_primary_hand([]) is None

    def test_single_hand(self) -> None:
        """손 1개 → 그대로 반환."""
        hand = {"landmarks": np.zeros((21, 2)), "score": 0.9}
        result = select_primary_hand([hand])
        assert result is hand

    def test_two_hands_selects_higher(self) -> None:
        """두 손 중 Y축이 높은(값이 작은) 손 선택."""
        hand_high = {
            "landmarks": np.array([[0.5, 0.2]] + [[0.0, 0.0]] * 20),
            "score": 0.9,
        }
        hand_low = {
            "landmarks": np.array([[0.5, 0.8]] + [[0.0, 0.0]] * 20),
            "score": 0.95,
        }
        result = select_primary_hand([hand_low, hand_high])
        # Y=0.2가 Y=0.8보다 위쪽
        assert result is hand_high

    def test_three_hands_filters_to_top_two_by_score(self) -> None:
        """3개 손: 신뢰도 상위 2개 중 Y축 높은 손."""
        hand_a = {
            "landmarks": np.array([[0.5, 0.1]] + [[0.0, 0.0]] * 20),
            "score": 0.5,  # 낮은 신뢰도 → 탈락
        }
        hand_b = {
            "landmarks": np.array([[0.5, 0.3]] + [[0.0, 0.0]] * 20),
            "score": 0.9,
        }
        hand_c = {
            "landmarks": np.array([[0.5, 0.7]] + [[0.0, 0.0]] * 20),
            "score": 0.95,
        }
        result = select_primary_hand([hand_a, hand_b, hand_c])
        # 신뢰도 상위 2: hand_c(0.95), hand_b(0.9)
        # Y축 높은(작은): hand_b(0.3) < hand_c(0.7)
        assert result is hand_b
