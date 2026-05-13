"""규칙 기반 검증 레이어.

CNN 예측 결과가 손가락 상태 패턴과 일치하는지 규칙으로 검증.
불일치 시 예측을 거부하여 오인식을 차단한다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 제스처별 필수 손가락 패턴
# 인덱스: [검지, 중지, 약지, 소지] (엄지는 z축 한계로 제외)
# True = 펼침, False = 접힘
GESTURE_RULES: dict[str, list[bool]] = {
    "fist": [False, False, False, False],       # 모두 접힘
    "palm": [True, True, True, True],           # 모두 펼침
    "point": [True, False, False, False],       # 검지만 펼침
    "scissors": [True, True, False, False],     # 검지+중지 펼침
}


class RuleValidator:
    """CNN 예측에 대한 규칙 기반 후처리 검증기."""

    def validate(self, gesture: str, finger_states: list[bool]) -> bool:
        """제스처 예측이 손가락 상태 규칙과 일치하는지 검증.

        Args:
            gesture: CNN이 예측한 제스처 라벨.
            finger_states: 손가락 펼침/접힘 상태 [검지, 중지, 약지, 소지].

        Returns:
            규칙과 일치하면 True, 불일치하면 False.
        """
        expected = GESTURE_RULES.get(gesture)
        if expected is None:
            logger.warning("알 수 없는 제스처 라벨: %s", gesture)
            return False

        is_valid = finger_states == expected
        if not is_valid:
            logger.debug(
                "규칙 검증 실패: %s 예측, 손가락=%s, 기대=%s",
                gesture,
                finger_states,
                expected,
            )
        return is_valid
