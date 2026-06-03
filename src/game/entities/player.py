"""플레이어 상태."""

from __future__ import annotations

from dataclasses import dataclass

from src.game.settings import MANA_CHARGE_PER_SECOND


@dataclass(slots=True)
class Player:
    max_hp: float = 100.0
    hp: float = 100.0
    max_mana: float = 100.0
    mana: float = 70.0
    mana_recovery_multiplier: float = 1.0
    global_cooldown_multiplier: float = 1.0

    def charge_mana(self, dt: float) -> None:
        amount = MANA_CHARGE_PER_SECOND * self.mana_recovery_multiplier * dt
        self.mana = min(self.max_mana, self.mana + amount)

    def spend_mana(self, amount: float) -> bool:
        if self.mana < amount:
            return False
        self.mana -= amount
        return True

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def heal_to_full(self) -> None:
        self.hp = self.max_hp
