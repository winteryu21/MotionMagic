"""전투 씬 — 2개 전장 전환 방식."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pygame

from src.game.entities.enemy import Enemy
from src.game.entities.player import Player
from src.game.entities.projectile import Explosion, LightningStrike, MagicMissile, Meteor
from src.game.settings import (
    BASE_LINE_X,
    BATTLE_BACKGROUND_FILES,
    COLOR_BASE,
    COLOR_FIELD_BG,
    COLOR_INACTIVE_FIELD_BG,
    COLOR_MUTED,
    COLOR_WHITE,
    GESTURE_COMBO_SIZE,
    GESTURE_PAPER,
    GESTURE_ROCK,
    GESTURE_SCISSORS,
    NUM_FIELDS,
    PLAYER_X,
    PLAYER_Y,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.game.systems.magic import MagicSystem
from src.game.systems.combat import CombatSystem
from src.game.systems.spawner import WaveSpawner
from src.game.ui.crosshair import Crosshair
from src.game.systems.reward import RewardOption, RewardSystem
from src.game.ui.fonts import get_font
from src.game.ui.hud import Hud
from src.game.scenes.unlock import UnlockScene


@dataclass(slots=True)
class BattleField:
    index: int
    background: pygame.Surface | None = None
    enemies: list[Enemy] = field(default_factory=list)
    projectiles: list[MagicMissile] = field(default_factory=list)
    effects: list[Explosion | LightningStrike | Meteor] = field(default_factory=list)
    @property
    def remaining_enemies(self) -> int:
        return sum(1 for enemy in self.enemies if enemy.alive)

    @property
    def base_line_x(self) -> int:
        return BASE_LINE_X if self.index == 0 else SCREEN_WIDTH - BASE_LINE_X

    @property
    def player_pos(self) -> tuple[int, int]:
        return (PLAYER_X, PLAYER_Y) if self.index == 0 else (SCREEN_WIDTH - PLAYER_X, PLAYER_Y)

    def update(self, dt: float, player: Player) -> int:
        defeated_count = 0
        for enemy in list(self.enemies):
            enemy.update(dt, self.base_line_x)
            if enemy.reached_base:
                player.take_damage(enemy.siege_damage)
                self.enemies.remove(enemy)
            elif not enemy.alive:
                defeated_count += 1
                self.enemies.remove(enemy)

        for projectile in list(self.projectiles):
            projectile.update(dt)

        CombatSystem.update_projectile_collisions(self)

        for projectile in list(self.projectiles):
            if not projectile.alive:
                self.projectiles.remove(projectile)

        for effect in list(self.effects):
            effect.update(dt, self.enemies)
            if not effect.alive:
                self.effects.remove(effect)

        return defeated_count

    def draw(self, surface: pygame.Surface, active: bool) -> None:
        field_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        if self.background is not None:
            surface.blit(self.background, (0, 0))
            if not active:
                inactive_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                inactive_overlay.fill((0, 0, 0, 120))
                surface.blit(inactive_overlay, (0, 0))
        else:
            pygame.draw.rect(surface, COLOR_FIELD_BG if active else COLOR_INACTIVE_FIELD_BG, field_rect)

        # 전장 방향에 맞춘 기지 라인과 플레이어
        base_x = self.base_line_x
        player_x, player_y = self.player_pos
        pygame.draw.line(surface, COLOR_BASE, (base_x, 70), (base_x, SCREEN_HEIGHT - 70), 5)
        pygame.draw.circle(surface, (90, 170, 255), (player_x, player_y), 22)
        pygame.draw.circle(surface, COLOR_WHITE, (player_x, player_y), 26, 2)

        for enemy in self.enemies:
            enemy.draw(surface, active)
        for projectile in self.projectiles:
            projectile.draw(surface)
        for effect in self.effects:
            effect.draw(surface)


class BattleScene:
    mouse_visible = False

    def __init__(self) -> None:
        self.player = Player()
        self.field_backgrounds = self._load_field_backgrounds()
        self.fields = [BattleField(index=i, background=self.field_backgrounds[i % len(self.field_backgrounds)] if self.field_backgrounds else None) for i in range(NUM_FIELDS)]
        self.active_field_index = 0
        self.spawner = WaveSpawner()
        self.magic = MagicSystem()
        self.hud = Hud()
        self.reward_system = RewardSystem()
        self.reward_options: list[RewardOption] = []
        self.crosshair = Crosshair()
        self.current_combo: list[str] = []
        self.aim_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.message = ""
        self.message_timer = 0.0
        self.reward_pending = False
        self.unlock_scene = UnlockScene()
        self.next_scene: str | None = None
        self.result_cleared_stage = 0
        self.game_over = False

    def _load_field_backgrounds(self) -> list[pygame.Surface]:
        project_root = Path(__file__).resolve().parents[3]
        backgrounds: list[pygame.Surface] = []
        for relative_path in BATTLE_BACKGROUND_FILES:
            image_path = project_root / relative_path
            if not image_path.exists():
                continue
            image = pygame.image.load(str(image_path)).convert()
            image = pygame.transform.smoothscale(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
            backgrounds.append(image)
        return backgrounds

    @property
    def active_field(self) -> BattleField:
        return self.fields[self.active_field_index]

    @property
    def total_remaining_enemies(self) -> int:
        alive = sum(field.remaining_enemies for field in self.fields)
        return alive + self.spawner.remaining_to_spawn

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.unlock_scene.pending:
            self._handle_unlock_event(event)
            return

        if self.reward_pending:
            self._handle_reward_event(event)
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self.active_field_index = (self.active_field_index + 1) % len(self.fields)
                self.message = f"전장 {self.active_field_index + 1}로 전환"
                self.message_timer = 1.2
            elif event.key == pygame.K_q:
                self._push_gesture(GESTURE_SCISSORS)
            elif event.key == pygame.K_w:
                self._push_gesture(GESTURE_ROCK)
            elif event.key == pygame.K_e:
                self._push_gesture(GESTURE_PAPER)
        elif event.type == pygame.MOUSEMOTION:
            self.aim_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.message = self.magic.cast_by_combo(
                self.current_combo,
                self.player,
                self.active_field,
                self.aim_pos,
                fields=self.fields,
                origin_pos=self.active_field.player_pos,
            )
            self.message_timer = 1.4
            self.current_combo.clear()

    def _push_gesture(self, gesture: str) -> None:
        self.current_combo.append(gesture)
        if len(self.current_combo) > GESTURE_COMBO_SIZE:
            self.current_combo.pop(0)
        self.message = "입력: " + " + ".join({"scissors": "가위", "rock": "바위", "paper": "보"}.get(g, g) for g in self.current_combo)
        self.message_timer = 1.2

    def _handle_unlock_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._close_unlock_scene()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._close_unlock_scene()

    def _handle_reward_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            key_to_index = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}
            if event.key in key_to_index:
                self._select_reward(key_to_index[event.key])
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, _option in enumerate(self.reward_options):
                if self.hud.reward_card_rect(index, len(self.reward_options)).collidepoint(event.pos):
                    self._select_reward(index)
                    break

    def _open_reward_selection(self) -> None:
        self.reward_pending = True
        pygame.mouse.set_visible(True)
        self.reward_options = self.reward_system.make_options(self.player, self.magic, 3)
        self.message = "스테이지 클리어! 보상 1개를 선택하면 다음 스테이지로 진행"
        self.message_timer = 3.0

    def _select_reward(self, index: int) -> None:
        if index < 0 or index >= len(self.reward_options):
            return
        self.message = self.reward_system.apply(self.reward_options[index], self.player, self.magic)
        self.message_timer = 2.0
        self.reward_pending = False
        self.reward_options.clear()
        self.current_combo.clear()

        next_stage = self.spawner.stage + 1
        if self.unlock_scene.should_open(next_stage, self.magic):
            unlock_spell = self.unlock_scene.next_unlock_spell(self.magic)
            if unlock_spell is not None:
                self._open_unlock_scene(unlock_spell, next_stage)
                return

        pygame.mouse.set_visible(False)
        self.spawner.start_stage(next_stage)

    def _open_unlock_scene(self, spell, next_stage: int) -> None:
        self.unlock_scene.open(spell, next_stage)
        pygame.mouse.set_visible(True)
        self.message = f"새 스킬 해금: {spell.name}"
        self.message_timer = 2.0

    def _close_unlock_scene(self) -> None:
        unlocked_name = self.unlock_scene.close(self.magic)
        pygame.mouse.set_visible(False)
        self.spawner.start_stage(self.unlock_scene.next_stage)
        self.message = f"{unlocked_name} 해금! Stage {self.spawner.stage} 시작"
        self.message_timer = 1.5

    def _open_result_scene(self) -> None:
        if self.game_over:
            return
        self.game_over = True
        self.result_cleared_stage = max(0, self.spawner.stage - 1)
        self.next_scene = "result"

    def update(self, dt: float) -> None:
        if self.unlock_scene.pending:
            self.unlock_scene.update(dt, self.magic)
            self.message_timer = max(0.0, self.message_timer - dt)
            return

        if self.reward_pending:
            self.message_timer = max(0.0, self.message_timer - dt)
            return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.player.charge_mana(dt)

        for field in self.fields:
            defeated = field.update(dt, self.player)
            if defeated:
                self.spawner.record_defeated(defeated)

        if not self.player.alive:
            self._open_result_scene()
            return

        stage_cleared = self.spawner.update(dt, self.fields, self.active_field_index)
        if stage_cleared:
            self._open_reward_selection()

        self.message_timer = max(0.0, self.message_timer - dt)

    def draw(self, surface: pygame.Surface) -> None:
        self.active_field.draw(surface, active=True)
        self._draw_inactive_field_minimap(surface)
        self.crosshair.draw(surface, self.aim_pos)
        self.hud.draw(
            surface=surface,
            player=self.player,
            stage=self.spawner.stage,
            active_field_index=self.active_field_index,
            fields=self.fields,
            remaining_enemies=self.total_remaining_enemies,
            current_combo=self.current_combo,
            spells=self.magic.possible_spells(self.current_combo),
            message=self.message if self.message_timer > 0 else "",
            reward_options=self.reward_options if self.reward_pending else None,
        )
        if self.unlock_scene.pending and self.unlock_scene.spell is not None:
            self.hud.draw_unlock_overlay(
                surface,
                self.unlock_scene.spell,
                self.unlock_scene.demo_time,
                self.unlock_scene.demo_field,
            )


    def _draw_inactive_field_minimap(self, surface: pygame.Surface) -> None:
        mini_w, mini_h = 320, 180  # 16:9 비율 유지 (1920x1080 / 6)
        x, y = SCREEN_WIDTH - mini_w - 20, SCREEN_HEIGHT - mini_h - 20
        inactive_index = 1 - self.active_field_index
        inactive = self.fields[inactive_index]

        # 반대편 전장을 임시 서페이스에 그대로 렌더링
        temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        inactive.draw(temp_surface, active=False)

        # 렌더링된 전장을 미니맵 크기로 축소
        minimap_surface = pygame.transform.smoothscale(temp_surface, (mini_w, mini_h))
        
        # 메인 화면에 축소된 미니맵 그리기
        rect = pygame.Rect(x, y, mini_w, mini_h)
        surface.blit(minimap_surface, (x, y))
        
        # 테두리
        pygame.draw.rect(surface, (100, 100, 120), rect, 2, border_radius=4)

        # 반대편 전장 텍스트 표시
        title_font = get_font(18, bold=True)
        font = get_font(16)
        
        title = title_font.render(f"Field {inactive_index + 1}", True, (255, 255, 255))
        surface.blit(title, (x + 10, y + 8))
        
        enemies_count = font.render(f"남은 적: {inactive.remaining_enemies}", True, (255, 100, 100))
        surface.blit(enemies_count, (x + 10, y + 30))
