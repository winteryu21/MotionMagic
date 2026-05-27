"""마법 데이터 모델 — SpellData 데이터클래스 및 전역 레지스트리."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 레벨당 데미지 배율 증가율
_DAMAGE_SCALE_PER_LEVEL: float = 0.20  # 레벨당 +20%
# 레벨당 쿨타임 감소율
_COOLDOWN_SCALE_PER_LEVEL: float = 0.08  # 레벨당 -8%
# 레벨당 투사체 속도 증가율
_SPEED_SCALE_PER_LEVEL: float = 0.10  # 레벨당 +10%
# 레벨당 피어싱 관통 수 증가 (정수)
_PIERCE_PER_LEVEL: int = 1  # 레벨당 관통 +1
# 레벨당 화염구 폭발 반경 증가율
_AOE_SCALE_PER_LEVEL: float = 0.15  # 레벨당 +15%
# 체인 라이트닝 레벨당 체인 수 증가
_CHAIN_PER_LEVEL: int = 1  # 레벨당 체인 +1


@dataclass
class SpellData:
    """마법 하나의 설정 및 상태를 담는 데이터 컨테이너.

    Attributes:
        name: 마법 식별자 (내부 키).
        display_name: HUD에 표시할 이름.
        base_mp_cost: 레벨 1 기준 마나 소비량.
        base_cooldown: 레벨 1 기준 쿨타임(초).
        base_damage: 레벨 1 기준 피해량.
        base_speed: 레벨 1 기준 투사체 속도(px/s). 히트스캔 마법은 0.
        unlocked: 해금 여부. False 이면 발동 불가.
        level: 현재 강화 레벨 (1 이상).
        status_effect: 적중 시 부여할 상태이상 ("stun" | "dot" | None).
        hotkey: HUD에 표시할 단축키 레이블.
        color: HUD 색상 (R, G, B).
    """

    name: str
    display_name: str
    base_mp_cost: float
    base_cooldown: float
    base_damage: float
    base_speed: float
    unlocked: bool = False
    level: int = 1
    status_effect: Optional[str] = None
    hotkey: str = ""
    color: tuple[int, int, int] = field(default_factory=lambda: (200, 200, 200))

    # ── 레벨 스케일링 프로퍼티 ────────────────────────────────────────────

    @property
    def mp_cost(self) -> float:
        """현재 레벨의 마나 소비량."""
        return self.base_mp_cost

    @property
    def cooldown(self) -> float:
        """현재 레벨의 쿨타임(초). 레벨당 -8%."""
        return self.base_cooldown * ((1.0 - _COOLDOWN_SCALE_PER_LEVEL) ** (self.level - 1))

    @property
    def damage(self) -> float:
        """현재 레벨의 기본 피해량. 레벨당 +20%."""
        return self.base_damage * (1.0 + _DAMAGE_SCALE_PER_LEVEL * (self.level - 1))

    @property
    def speed(self) -> float:
        """현재 레벨의 투사체 속도. 레벨당 +10%."""
        return self.base_speed * (1.0 + _SPEED_SCALE_PER_LEVEL * (self.level - 1))

    @property
    def pierce_count(self) -> int:
        """관통 마탄의 관통 수. 레벨당 +1 (기본 3)."""
        return 3 + _PIERCE_PER_LEVEL * (self.level - 1)

    @property
    def aoe_radius(self) -> float:
        """화염구 폭발 반경. 레벨당 +15% (기본 1200 px)."""
        return 1200.0 * (1.0 + _AOE_SCALE_PER_LEVEL * (self.level - 1))

    @property
    def chain_count(self) -> int:
        """체인 라이트닝 최대 체인 수. 레벨당 +1 (기본 4)."""
        return 4 + _CHAIN_PER_LEVEL * (self.level - 1)

    # ── 레벨업 ──────────────────────────────────────────────────────────

    def level_up(self) -> None:
        """마법을 한 단계 강화한다."""
        self.level += 1
        logger.info("마법 '%s' 레벨업: Lv.%d", self.name, self.level)

    def unlock(self) -> None:
        """마법을 해금한다."""
        self.unlocked = True
        logger.info("마법 '%s' 해금됨", self.name)


# ── 전역 마법 레지스트리 ──────────────────────────────────────────────────────
# 게임 내에서 이 dict를 직접 수정하여 해금/레벨업 관리

SPELL_REGISTRY: dict[str, SpellData] = {
    "piercing_bullet": SpellData(
        name="piercing_bullet",
        display_name="Piercing Bullet",
        base_mp_cost=15.0,
        base_cooldown=1.0,
        base_damage=20.0,
        base_speed=600.0,
        unlocked=True,          # 기본 해금
        hotkey="1",
        color=(155, 89, 182),
    ),
    "fireball": SpellData(
        name="fireball",
        display_name="Fireball",
        base_mp_cost=30.0,
        base_cooldown=3.0,
        base_damage=40.0,
        base_speed=350.0,
        unlocked=True,          # 기본 해금
        hotkey="2",
        color=(230, 126, 34),
    ),
    "chain_lightning": SpellData(
        name="chain_lightning",
        display_name="Chain Lightning",
        base_mp_cost=45.0,
        base_cooldown=5.0,
        base_damage=25.0,
        base_speed=0.0,         # 히트스캔
        unlocked=False,         # 잠금 (스테이지 보상으로 해금)
        hotkey="3",
        color=(52, 152, 219),
    ),
}
