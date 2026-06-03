"""HUD — 체력/마나/스테이지/스킬 조합/보상 선택 표시."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from src.game.entities.enemy import Enemy
from src.game.entities.player import Player
from src.game.entities.projectile import Explosion, LightningStrike, MagicMissile
from src.game.settings import (
    COLOR_HUD_HP,
    COLOR_HUD_MP,
    COLOR_MUTED,
    COLOR_WHITE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.game.systems.magic import Spell
from src.game.systems.reward import RewardOption
from src.game.ui.fonts import get_font

if TYPE_CHECKING:
    from src.game.scenes.battle import BattleField

GESTURE_KR = {
    "scissors": "가위",
    "rock": "바위",
    "paper": "보",
}

GESTURE_COLORS = {
    "scissors": (78, 205, 196),
    "rock": (255, 177, 66),
    "paper": (116, 185, 255),
}

SKILL_PANEL_X = 320
SKILL_PANEL_Y = 26
SKILL_PANEL_RIGHT_MARGIN = 190
SKILL_PANEL_TARGET_ROWS_PER_COL = 4
SKILL_PANEL_ROW_W = 210
SKILL_PANEL_ROW_H = 30
SKILL_PANEL_ROW_GAP = 5
SKILL_PANEL_COMBO_GAP = 8
SKILL_PANEL_COMBO_GESTURE_W = 30
SKILL_PANEL_COMBO_GESTURE_H = 24
SKILL_PANEL_COMBO_STEP = 34
SKILL_PANEL_COL_GAP = 18
SKILL_PANEL_STATUS_AREA_W = 60


def _skill_panel_combo_area_width(max_combo_len: int) -> int:
    combo_len = max(1, max_combo_len)
    trailing_gap = SKILL_PANEL_COMBO_STEP - SKILL_PANEL_COMBO_GESTURE_W
    return combo_len * SKILL_PANEL_COMBO_STEP - trailing_gap


def _skill_panel_layout(
    spell_count: int,
    max_combo_len: int,
) -> tuple[int, int, int, int]:
    spell_count = max(1, spell_count)
    combo_area_w = _skill_panel_combo_area_width(max_combo_len)
    col_w = (
        SKILL_PANEL_ROW_W + SKILL_PANEL_COMBO_GAP + combo_area_w + SKILL_PANEL_COL_GAP
    )
    available_w = max(1, SCREEN_WIDTH - SKILL_PANEL_RIGHT_MARGIN - SKILL_PANEL_X)
    max_columns = max(1, available_w // col_w)
    target_columns = math.ceil(spell_count / SKILL_PANEL_TARGET_ROWS_PER_COL)
    columns = max(1, min(spell_count, max_columns, target_columns))
    rows_per_col = max(1, math.ceil(spell_count / columns))
    return columns, rows_per_col, col_w, combo_area_w


class Hud:
    def __init__(self) -> None:
        self.font = get_font(25)
        self.small_font = get_font(20)
        self.tiny_font = get_font(17)
        self.big_font = get_font(44, bold=True)
        self.title_font = get_font(30, bold=True)

        # 보상창은 기존보다 2pt 낮춘 전용 폰트 사용
        self.reward_big_font = get_font(42, bold=True)
        self.reward_title_font = get_font(28, bold=True)
        self.reward_small_font = get_font(18)
        self.reward_desc_font = get_font(17)
        self.reward_tiny_font = get_font(15)
        self.gesture_icons = self._load_gesture_icons()

    def _load_gesture_icons(self) -> dict[str, pygame.Surface]:
        project_root = Path(__file__).resolve().parents[3]
        icon_dir = project_root / "assets" / "icons" / "gestures"
        icon_files = {
            "scissors": "scissors.png",
            "rock": "rock.png",
            "paper": "paper.png",
        }
        icons: dict[str, pygame.Surface] = {}

        for gesture, filename in icon_files.items():
            path = icon_dir / filename
            if path.exists():
                icons[gesture] = pygame.image.load(str(path)).convert_alpha()

        return icons

    def draw(
        self,
        surface: pygame.Surface,
        player: Player,
        stage: int,
        active_field_index: int,
        fields: list[BattleField],
        remaining_enemies: int,
        current_combo: list[str],
        spells: list[Spell],
        message: str,
        reward_options: list[RewardOption] | None = None,
    ) -> None:
        self._bar(
            surface,
            30,
            26,
            260,
            22,
            player.hp / player.max_hp,
            COLOR_HUD_HP,
            f"Life {int(player.hp)}/{int(player.max_hp)}",
        )
        self._bar(
            surface,
            30,
            58,
            260,
            22,
            player.mana / player.max_mana,
            COLOR_HUD_MP,
            f"Mana {int(player.mana)}/{int(player.max_mana)}",
        )

        stage_text = self.title_font.render(f"Stage {stage}", True, COLOR_WHITE)
        surface.blit(stage_text, (SCREEN_WIDTH - stage_text.get_width() - 28, 24))

        self._draw_skill_panel(surface, player, current_combo, spells)
        self._draw_current_combo_center(surface, current_combo)

        if reward_options:
            self.draw_reward_overlay(surface, reward_options)

    def draw_reward_overlay(
        self, surface: pygame.Surface, reward_options: list[RewardOption]
    ) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        surface.blit(overlay, (0, 0))

        title = self.reward_big_font.render(
            "보상을 선택해 주세요", True, (255, 220, 120)
        )
        surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 70))

        for index, option in enumerate(reward_options):
            rect = self.reward_card_rect(index, len(reward_options))
            self._draw_reward_card(surface, rect, index, option)

    def reward_card_rect(self, index: int, total: int) -> pygame.Rect:
        card_w, card_h = 245, 360
        gap = 34
        total_w = total * card_w + (total - 1) * gap
        x = SCREEN_WIDTH // 2 - total_w // 2 + index * (card_w + gap)
        y = SCREEN_HEIGHT // 2 - card_h // 2 + 35
        return pygame.Rect(x, y, card_w, card_h)

    def _draw_reward_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        index: int,
        option: RewardOption,
    ) -> None:
        pygame.draw.rect(surface, (245, 227, 188), rect, border_radius=8)
        pygame.draw.rect(surface, (168, 128, 72), rect, 3, border_radius=8)

        header = pygame.Rect(rect.x, rect.y, rect.w, 64)
        pygame.draw.rect(
            surface,
            (255, 190, 90),
            header,
            border_top_left_radius=8,
            border_top_right_radius=8,
        )
        badge_text = "플레이어" if option.category == "player" else "마법"
        badge_color = (210, 80, 65) if option.category == "player" else (70, 150, 100)
        badge = pygame.Rect(rect.x + 14, rect.y + 14, 72, 26)
        pygame.draw.rect(surface, badge_color, badge, border_radius=6)
        self._center_text(
            surface, badge_text, self.reward_tiny_font, badge, COLOR_WHITE
        )

        num = self.reward_small_font.render(f"{index + 1}", True, (80, 52, 30))
        surface.blit(num, (rect.right - 30, rect.y + 16))

        title_rect = pygame.Rect(rect.x + 16, rect.y + 78, rect.w - 32, 52)
        self._wrapped_text(
            surface,
            option.title,
            self.reward_title_font,
            title_rect,
            (55, 42, 35),
            line_gap=2,
        )

        pygame.draw.line(
            surface,
            (190, 165, 120),
            (rect.x + 18, rect.y + 144),
            (rect.right - 18, rect.y + 144),
            2,
        )

        desc_rect = pygame.Rect(rect.x + 20, rect.y + 160, rect.w - 40, rect.h - 192)
        self._wrapped_text(
            surface,
            option.description,
            self.reward_desc_font,
            desc_rect,
            (65, 58, 50),
            line_gap=7,
        )

        footer = pygame.Rect(rect.x + 28, rect.bottom - 48, rect.w - 56, 30)
        pygame.draw.rect(surface, (238, 238, 232), footer, border_radius=6)
        self._center_text(surface, "선택", self.reward_small_font, footer, (65, 58, 50))

    def _bar(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        ratio: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        ratio = max(0.0, min(1.0, ratio))
        bg = pygame.Rect(x, y, w, h)
        fg = pygame.Rect(x, y, int(w * ratio), h)
        pygame.draw.rect(surface, (55, 57, 72), bg, border_radius=6)
        pygame.draw.rect(surface, color, fg, border_radius=6)
        text = self.small_font.render(label, True, COLOR_WHITE)
        surface.blit(text, (x + 8, y + 1))

    def _draw_skill_panel(
        self,
        surface: pygame.Surface,
        player: Player,
        current_combo: list[str],
        spells: list[Spell],
    ) -> None:
        now = time.monotonic()
        title = self.tiny_font.render("Skills", True, COLOR_WHITE)
        combo_title = self.tiny_font.render("Combo", True, COLOR_WHITE)
        surface.blit(title, (SKILL_PANEL_X + 2, SKILL_PANEL_Y - 20))
        surface.blit(
            combo_title,
            (
                SKILL_PANEL_X + SKILL_PANEL_ROW_W + SKILL_PANEL_COMBO_GAP + 2,
                SKILL_PANEL_Y - 20,
            ),
        )

        visible_spells = [spell for spell in spells if spell.is_unlocked()]
        if not visible_spells:
            empty = self.tiny_font.render("해금된 스킬 없음", True, COLOR_MUTED)
            surface.blit(empty, (SKILL_PANEL_X, SKILL_PANEL_Y + 8))
            return

        max_combo_len = max(len(spell.combo) for spell in visible_spells)
        _columns, rows_per_col, col_w, _combo_area_w = _skill_panel_layout(
            len(visible_spells),
            max_combo_len,
        )

        for index, spell in enumerate(visible_spells):
            col = index // rows_per_col
            row = index % rows_per_col
            base_x = SKILL_PANEL_X + col * col_w
            row_y = SKILL_PANEL_Y + row * (SKILL_PANEL_ROW_H + SKILL_PANEL_ROW_GAP)
            combo_x = base_x + SKILL_PANEL_ROW_W + SKILL_PANEL_COMBO_GAP

            possible = (not current_combo) or spell.combo[
                : len(current_combo)
            ] == tuple(current_combo)
            cooldown_left = spell.cooldown_remaining(player, now)
            enough_mana = player.mana >= spell.stat.mana_cost
            ready = cooldown_left <= 0.0 and enough_mana and possible

            if ready:
                fill = (30, 34, 50)
                border = (145, 95, 190)
                text_color = COLOR_WHITE
            else:
                fill = (7, 8, 13)
                border = (45, 34, 55)
                text_color = (105, 108, 125)

            rect = pygame.Rect(
                base_x,
                row_y,
                SKILL_PANEL_ROW_W,
                SKILL_PANEL_ROW_H,
            )
            pygame.draw.rect(surface, fill, rect, border_radius=7)
            pygame.draw.rect(surface, border, rect, 2, border_radius=7)

            label_text = (
                f"{spell.name} Lv.{spell.level}  ({int(spell.stat.mana_cost)} MP)"
            )
            label = self.tiny_font.render(label_text, True, text_color)
            max_label_w = SKILL_PANEL_ROW_W - SKILL_PANEL_STATUS_AREA_W - 10
            if label.get_width() > max_label_w:
                short_text = f"{spell.name} Lv.{spell.level}"
                label = self.tiny_font.render(short_text, True, text_color)
                if label.get_width() > max_label_w:
                    while len(short_text) > 4 and label.get_width() > max_label_w:
                        short_text = short_text[:-2]
                        label = self.tiny_font.render(
                            short_text + "…", True, text_color
                        )
            surface.blit(label, (rect.x + 10, rect.y + 7))

            if cooldown_left > 0.0:
                status_text = f"{cooldown_left:.1f}s"
                status_color = (255, 100, 74)
            elif not enough_mana:
                status_text = "MP"
                status_color = (95, 150, 255)
            elif possible:
                status_text = "READY"
                status_color = (70, 225, 150)
            else:
                status_text = "-"
                status_color = (100, 96, 116)

            status = self.tiny_font.render(status_text, True, status_color)
            surface.blit(status, (rect.right - status.get_width() - 8, rect.y + 6))

            remaining = (
                spell.combo[len(current_combo) :]
                if possible and current_combo
                else spell.combo
            )
            gx = combo_x
            for gesture in remaining:
                self._gesture_box(
                    surface,
                    gx,
                    row_y + 3,
                    gesture,
                    SKILL_PANEL_COMBO_GESTURE_W,
                    SKILL_PANEL_COMBO_GESTURE_H,
                    dim=not ready and cooldown_left > 0.0,
                )
                gx += SKILL_PANEL_COMBO_STEP

    def _gesture_box(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        gesture: str,
        w: int = 50,
        h: int = 34,
        dim: bool = False,
    ) -> None:
        rect = pygame.Rect(x, y, w, h)
        glow = GESTURE_COLORS.get(gesture, (100, 220, 255))
        fill = (8, 10, 14) if dim else (14, 18, 25)
        border = (40, 45, 52) if dim else tuple(min(255, int(v * 0.85)) for v in glow)

        pygame.draw.rect(surface, fill, rect, border_radius=4)
        pygame.draw.rect(surface, border, rect, 1, border_radius=4)

        icon = self.gesture_icons.get(gesture)
        if icon is not None:
            icon_size = min(rect.w - 2, rect.h - 2)
            icon_image = pygame.transform.smoothscale(icon, (icon_size, icon_size))
            if dim:
                icon_image = icon_image.copy()
                icon_image.set_alpha(125)
            surface.blit(icon_image, icon_image.get_rect(center=rect.center))
            return

        text = self.tiny_font.render(
            GESTURE_KR.get(gesture, gesture),
            True,
            COLOR_WHITE if not dim else (120, 124, 138),
        )
        surface.blit(
            text,
            (
                x + rect.w // 2 - text.get_width() // 2,
                y + rect.h // 2 - text.get_height() // 2,
            ),
        )

    def _draw_current_combo_center(
        self, surface: pygame.Surface, current_combo: list[str]
    ) -> None:
        if not current_combo:
            return
        overlay = pygame.Surface((360, 90), pygame.SRCALPHA)
        pygame.draw.rect(
            overlay, (255, 255, 255, 34), overlay.get_rect(), border_radius=18
        )
        x = 32
        for gesture in current_combo:
            self._gesture_box(overlay, x, 29, gesture)
            x += 70
        surface.blit(overlay, (SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2 - 45))

    def draw_unlock_overlay(
        self,
        surface: pygame.Surface,
        spell: Spell,
        demo_time: float,
        demo_field: BattleField | None = None,
    ) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 185))
        surface.blit(overlay, (0, 0))

        title = self.reward_big_font.render("새로운 스킬 해금", True, (255, 220, 120))
        surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 54))

        box = pygame.Rect(SCREEN_WIDTH // 2 - 330, 120, 660, 405)
        pygame.draw.rect(surface, (26, 29, 43), box, border_radius=18)
        pygame.draw.rect(surface, (145, 95, 190), box, 3, border_radius=18)

        name = self.title_font.render(
            f"{spell.name} Lv.{spell.level}", True, COLOR_WHITE
        )
        surface.blit(name, (box.centerx - name.get_width() // 2, box.y + 24))

        player_pos = (box.centerx - 200, box.y + 210)
        enemy_pos = (box.centerx + 200, box.y + 210)
        phase = demo_time % 1.45

        pygame.draw.circle(surface, (90, 170, 255), player_pos, 24)
        pygame.draw.circle(surface, COLOR_WHITE, player_pos, 28, 2)
        player_label = self.tiny_font.render("Player", True, COLOR_WHITE)
        surface.blit(
            player_label,
            (player_pos[0] - player_label.get_width() // 2, player_pos[1] + 38),
        )

        if demo_field is not None and demo_field.enemies:
            demo_field.enemies[0].draw(surface, True)
        else:
            pygame.draw.rect(
                surface,
                (235, 87, 87),
                pygame.Rect(enemy_pos[0] - 24, enemy_pos[1] - 24, 48, 48),
                border_radius=8,
            )
            pygame.draw.rect(
                surface,
                (40, 220, 130),
                pygame.Rect(enemy_pos[0] - 26, enemy_pos[1] - 38, 52, 6),
                border_radius=3,
            )
        enemy_label = self.tiny_font.render("Target", True, COLOR_WHITE)
        surface.blit(
            enemy_label,
            (enemy_pos[0] - enemy_label.get_width() // 2, enemy_pos[1] + 38),
        )

        active_demo_objects = False
        if demo_field is not None:
            for effect in demo_field.effects:
                effect.draw(surface)
                active_demo_objects = True
            for projectile in demo_field.projectiles:
                projectile.draw(surface)
                active_demo_objects = True

        if not active_demo_objects and phase >= 0.9:
            self._draw_demo_ready_text(surface, box.center)

        combo_label = self.small_font.render(f"조합법: {spell.name}", True, COLOR_WHITE)
        surface.blit(
            combo_label, (box.centerx - combo_label.get_width() // 2, box.bottom - 93)
        )
        total_w = len(spell.combo) * 56 + (len(spell.combo) - 1) * 12
        gx = box.centerx - total_w // 2
        gy = box.bottom - 58
        for gesture in spell.combo:
            self._gesture_box(surface, gx, gy, gesture, 56, 38)
            gx += 68

    def _draw_lightning_demo(
        self,
        surface: pygame.Surface,
        enemy_pos: tuple[int, int],
        phase: float,
        radius: float,
        box_center: tuple[int, int],
    ) -> None:
        if phase < 0.9:
            effect = LightningStrike(
                x=enemy_pos[0],
                y=enemy_pos[1],
                damage=0,
                radius=radius,
            )
            effect.draw(surface)
        else:
            self._draw_demo_ready_text(surface, box_center)

    def _draw_explosion_demo(
        self,
        surface: pygame.Surface,
        enemy_pos: tuple[int, int],
        phase: float,
        radius: float,
        box_center: tuple[int, int],
    ) -> None:
        if phase < 0.9:
            effect = Explosion(
                x=enemy_pos[0],
                y=enemy_pos[1],
                damage=0,
                radius=radius,
            )
            effect.age = min(effect.duration, effect.duration * (phase / 0.9))
            effect.draw(surface)
        else:
            self._draw_demo_ready_text(surface, box_center)

    def _draw_projectile_demo(
        self,
        surface: pygame.Surface,
        player_pos: tuple[int, int],
        enemy_pos: tuple[int, int],
        phase: float,
        box_center: tuple[int, int],
    ) -> None:
        if phase < 0.9:
            t = min(1.0, phase / 0.9)
            px = player_pos[0] + (enemy_pos[0] - player_pos[0]) * t
            py = player_pos[1] + (enemy_pos[1] - player_pos[1]) * t
            dummy_target = Enemy(x=enemy_pos[0], y=enemy_pos[1])
            projectile = MagicMissile(
                x=px,
                y=py,
                target=dummy_target,
                damage=0,
                speed=0,
            )
            projectile.draw(surface)
        else:
            self._draw_demo_ready_text(surface, box_center)

    def _draw_demo_ready_text(
        self, surface: pygame.Surface, box_center: tuple[int, int]
    ) -> None:
        ready = self.small_font.render("재시전 준비...", True, COLOR_MUTED)
        surface.blit(
            ready,
            (
                box_center[0] - ready.get_width() // 2,
                box_center[1] - ready.get_height() // 2,
            ),
        )

    def _center_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        rect: pygame.Rect,
        color: tuple[int, int, int],
    ) -> None:
        image = font.render(text, True, color)
        surface.blit(
            image,
            (
                rect.centerx - image.get_width() // 2,
                rect.centery - image.get_height() // 2,
            ),
        )

    def _wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        line_gap: int = 4,
    ) -> None:
        y = rect.y
        for paragraph in text.split("\n"):
            line = ""
            for ch in paragraph:
                test = line + ch
                if font.size(test)[0] <= rect.w:
                    line = test
                else:
                    if line:
                        image = font.render(line, True, color)
                        surface.blit(image, (rect.x, y))
                        y += image.get_height() + line_gap
                    line = ch
            if line:
                image = font.render(line, True, color)
                surface.blit(image, (rect.x, y))
                y += image.get_height() + line_gap
            y += line_gap
            if y > rect.bottom:
                break
