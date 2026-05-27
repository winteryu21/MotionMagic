"""게임 전역 상태 및 자원 관리 매니저 (싱글톤)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 플레이어 기본 능력치 설정
DEFAULT_MAX_HP = 100.0
DEFAULT_MAX_MP = 100.0
DEFAULT_MANA_RECOVERY = 5.0   # 초당 마나 회복량
DEFAULT_COOLDOWN_REDUCTION = 0.0  # 쿨타임 감소율 (0.0 ~ 1.0)


class GameManager:
    """게임의 전반적인 상태를 중앙 관리하는 싱글톤 클래스.

    플레이어의 공유 체력(HP), 마나(MP), 현재 웨이브, 점수, 그리고 스킬 쿨타임과
    활성화된 전장 상태 등을 총괄한다.
    """

    _instance: GameManager | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> GameManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.reset()
        logger.info("GameManager 싱글톤 인스턴스 생성 및 초기화 완료")

    def reset(self) -> None:
        """게임 전역 자원 및 상태를 초기치로 리셋한다."""
        self.hp: float = DEFAULT_MAX_HP
        self.max_hp: float = DEFAULT_MAX_HP
        self.mp: float = DEFAULT_MAX_MP
        self.max_mp: float = DEFAULT_MAX_MP
        self.mana_regen_rate: float = DEFAULT_MANA_RECOVERY
        self.cooldown_reduction: float = DEFAULT_COOLDOWN_REDUCTION
        self.score: int = 0
        self.current_wave: int = 1
        self.active_field: int = 0  # 0 또는 1 (두 전장 교대)

        # 각 마법별 남은 쿨타임 (초 단위)
        self.spell_cooldowns: dict[str, float] = {}
        # 웨이브 클리어 후 보상 선택 대기 여부
        self.reward_pending: bool = False
        logger.debug("GameManager 상태 리셋")

    def update(self, dt: float, is_recharging: bool = False) -> None:
        """시간 경과에 따른 매니저 상태 업데이트 (쿨다운 감소 및 선택적 마나 회복).

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
            is_recharging: 마나를 충전(회복) 중인지 여부.
        """
        # 1. 마나 회복 (충전 중일 때만 회복)
        if is_recharging and self.mp < self.max_mp:
            self.mp = min(self.max_mp, self.mp + self.mana_regen_rate * dt)

        # 2. 스킬 쿨타임 업데이트
        for spell in list(self.spell_cooldowns.keys()):
            if self.spell_cooldowns[spell] > 0.0:
                self.spell_cooldowns[spell] = max(0.0, self.spell_cooldowns[spell] - dt)

    def take_damage(self, amount: float) -> None:
        """플레이어 공유 체력 감소 및 사망 조건 검사.

        Args:
            amount: 플레이어가 받는 데미지 양.
        """
        self.hp = max(0.0, self.hp - amount)
        logger.info("플레이어 피해 발생: -%.1f (남은 HP: %.1f)", amount, self.hp)

    def consume_mana(self, amount: float) -> bool:
        """마나 소비 가능 여부를 확인하고 마나를 소모한다.

        Args:
            amount: 소모할 마나 양.

        Returns:
            마나가 충분하여 소모에 성공하면 True, 부족하면 False.
        """
        if self.mp >= amount:
            self.mp -= amount
            return True
        return False

    def is_spell_ready(self, spell_name: str) -> bool:
        """지정한 마법이 시전 가능한 상태(쿨다운 종료)인지 검사한다.

        Args:
            spell_name: 마법 이름.

        Returns:
            쿨다운이 끝나 사용 가능하면 True, 아직 쿨다운 중이면 False.
        """
        cooldown = self.spell_cooldowns.get(spell_name, 0.0)
        return cooldown <= 0.0

    def start_spell_cooldown(self, spell_name: str, cooldown: float) -> None:
        """지정한 마법의 쿨다운을 적용한다.

        Args:
            spell_name: 마법 이름.
            cooldown: 적용할 쿨타임 시간(초).
        """
        effective = cooldown * (1.0 - self.cooldown_reduction)
        self.spell_cooldowns[spell_name] = effective
        logger.debug("마법 '%s' 쿨다운 시작: %.2f초 (감소율 %.0f%%)", spell_name, effective, self.cooldown_reduction * 100)
