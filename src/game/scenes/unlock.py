"""스킬 해금 씬 상태와 시연 관리."""

from __future__ import annotations

from dataclasses import dataclass, field
import pygame

from src.game.entities.enemy import Enemy
from src.game.entities.player import Player
from src.game.entities.projectile import Explosion, LightningStrike, MagicMissile
from src.game.settings import SCREEN_WIDTH
from src.game.systems.magic import MagicSystem, Spell


@dataclass(slots=True)
class UnlockDemoField:
    enemies: list[Enemy] = field(default_factory=list)
    projectiles: list[MagicMissile] = field(default_factory=list)
    effects: list[Explosion | LightningStrike] = field(default_factory=list)


class UnlockScene:
    """보상 선택 후 표시되는 신규 스킬 해금 시연 씬."""

    def __init__(self) -> None:
        # 이 배열에 spell key를 추가하면 순서대로 해금 씬이 작동함.
        # 예: ["lightning", "ice_spear", "meteor"]
        self.unlock_order: list[str] = [
            "lightning",
            "explosion",
            "piercing_bullet",
            "meteor",
        ]
        self.pending = False
        self.spell: Spell | None = None
        self.next_stage = 1
        self.demo_time = 0.0
        self.demo_field: UnlockDemoField | None = None
        self.demo_enemy: Enemy | None = None
        self.demo_player = Player(max_mana=999.0, mana=999.0)
        self.demo_cycle = -1

    def positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        box = pygame.Rect(SCREEN_WIDTH // 2 - 330, 120, 660, 405)
        player_pos = (box.centerx - 200, box.y + 210)
        enemy_pos = (box.centerx + 200, box.y + 210)
        return player_pos, enemy_pos

    def next_unlock_spell(self, magic: MagicSystem) -> Spell | None:
        for spell_key in self.unlock_order:
            spell = magic.spells.get(spell_key)
            if spell is not None and not spell.is_unlocked():
                return spell
        return None

    def has_pending_unlock(self, magic: MagicSystem) -> bool:
        return self.next_unlock_spell(magic) is not None

    def should_open(self, next_stage: int, magic: MagicSystem) -> bool:
        return next_stage % 2 == 0 and self.has_pending_unlock(magic)

    def open(self, spell: Spell, next_stage: int) -> None:
        self.pending = True
        self.spell = spell
        self.next_stage = next_stage
        self.demo_time = 0.0
        self.demo_cycle = -1
        self.reset_demo_field()

    def close(self, magic: MagicSystem) -> str:
        if self.spell is not None:
            magic.unlock_spell(self.spell)
            unlocked_name = self.spell.name
        else:
            unlocked_name = "새 스킬"
        self.pending = False
        self.spell = None
        self.demo_field = None
        self.demo_enemy = None
        return unlocked_name

    def reset_demo_field(self) -> None:
        _player_pos, enemy_pos = self.positions()
        self.demo_enemy = Enemy(
            x=enemy_pos[0],
            y=enemy_pos[1],
            hp=9999.0,
            max_hp=9999.0,
            speed=0.0,
            size=48,
        )
        self.demo_field = UnlockDemoField(enemies=[self.demo_enemy])
        self.demo_player.mana = self.demo_player.max_mana

    def cast_demo_spell(self, magic: MagicSystem) -> None:
        if self.spell is None:
            return
        if self.demo_field is None or self.demo_enemy is None:
            self.reset_demo_field()
        if self.demo_field is None or self.demo_enemy is None:
            return

        player_pos, enemy_pos = self.positions()
        self.demo_field.projectiles.clear()
        self.demo_field.effects.clear()
        self.demo_enemy.x = enemy_pos[0]
        self.demo_enemy.y = enemy_pos[1]
        self.demo_enemy.hp = self.demo_enemy.max_hp
        self.demo_enemy.reached_base = False
        self.demo_enemy.status_effects.clear()
        magic.cast(
            self.spell,
            self.demo_player,
            self.demo_field,
            enemy_pos,
            ignore_requirements=True,
            consume_resources=False,
            origin_pos=player_pos,
        )

    def update(self, dt: float, magic: MagicSystem) -> None:
        if self.spell is None:
            return
        self.demo_time += dt
        cycle = int(self.demo_time / 1.45)
        if cycle != self.demo_cycle:
            self.demo_cycle = cycle
            self.cast_demo_spell(magic)

        if self.demo_field is None:
            return
        for projectile in list(self.demo_field.projectiles):
            projectile.update(dt)
            if not projectile.alive:
                self.demo_field.projectiles.remove(projectile)
        for effect in list(self.demo_field.effects):
            effect.update(dt, self.demo_field.enemies)
            if not effect.alive:
                self.demo_field.effects.remove(effect)
