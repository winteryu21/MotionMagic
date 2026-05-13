"""규칙 기반 검증기 테스트."""

from __future__ import annotations

from src.ai.rule_validator import RuleValidator


class TestRuleValidator:
    """RuleValidator 단위 테스트."""

    def setup_method(self) -> None:
        self.validator = RuleValidator()

    def test_fist_all_closed(self) -> None:
        """주먹: 모든 손가락 접힘 → 통과."""
        assert self.validator.validate("fist", [False, False, False, False]) is True

    def test_fist_rejected_when_finger_extended(self) -> None:
        """주먹: 검지 펼침 → 거부."""
        assert self.validator.validate("fist", [True, False, False, False]) is False

    def test_palm_all_open(self) -> None:
        """손바닥: 모든 손가락 펼침 → 통과."""
        assert self.validator.validate("palm", [True, True, True, True]) is True

    def test_point_only_index(self) -> None:
        """포인트: 검지만 펼침 → 통과."""
        assert self.validator.validate("point", [True, False, False, False]) is True

    def test_scissors_index_and_middle(self) -> None:
        """가위: 검지+중지 펼침 → 통과."""
        assert self.validator.validate("scissors", [True, True, False, False]) is True

    def test_unknown_gesture_rejected(self) -> None:
        """미등록 제스처 → 거부."""
        assert self.validator.validate("unknown", [True, True, True, True]) is False
