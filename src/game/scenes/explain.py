"""조작법 설명 씬."""

from __future__ import annotations

from pathlib import Path

import pygame

from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.fonts import get_font


GESTURE_COLORS = {
    "scissors": (78, 205, 196),
    "rock": (255, 177, 66),
    "paper": (116, 185, 255),
}


class ExplainScene:
    """전투 시작 전 조작법을 손모양 아이콘으로 안내하는 씬."""

    mouse_visible = True

    def __init__(self) -> None:
        self.next_scene: str | None = None
        self.title_font = get_font(44, bold=True)
        self.font = get_font(25)
        self.small_font = get_font(20)
        self.tiny_font = get_font(17)
        self.start_button = pygame.Rect(SCREEN_WIDTH // 2 - 135, SCREEN_HEIGHT - 120, 270, 62)
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

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.next_scene = "battle"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.start_button.collidepoint(event.pos):
                self.next_scene = "battle"

    def update(self, dt: float) -> None:
        return

    def draw(self, surface: pygame.Surface) -> None:
        panel = pygame.Rect(SCREEN_WIDTH // 2 - 430, 68, 860, 520)
        pygame.draw.rect(surface, (22, 27, 43), panel, border_radius=22)
        pygame.draw.rect(surface, COLOR_MUTED, panel, 2, border_radius=22)

        title = self.title_font.render("조작법", True, COLOR_WHITE)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 52)))

        subtitle = self.small_font.render("손모양 조합을 입력하고 마우스로 마법을 발사함", True, COLOR_MUTED)
        surface.blit(subtitle, subtitle.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 92)))

        y = panel.y + 128
        self._draw_gesture_control_row(
            surface,
            panel.x + 64,
            y,
            "Q",
            "scissors",
            "가위 손모양 입력",
        )
        self._draw_gesture_control_row(
            surface,
            panel.x + 64,
            y + 72,
            "W",
            "rock",
            "바위 손모양 입력",
        )
        self._draw_gesture_control_row(
            surface,
            panel.x + 64,
            y + 144,
            "E",
            "paper",
            "보 손모양 입력",
        )

        right_x = panel.centerx + 56
        self._draw_text_control_row(surface, right_x, y, "마우스 이동", "조준점 이동")
        self._draw_text_control_row(surface, right_x, y + 58, "마우스 클릭", "입력한 조합으로 마법 발사")
        self._draw_text_control_row(surface, right_x, y + 116, "Space 유지", "마나 충전")
        self._draw_text_control_row(surface, right_x, y + 174, "Tab", "전장 전환")

        guide_box = pygame.Rect(panel.x + 70, panel.bottom - 105, panel.w - 140, 56)
        pygame.draw.rect(surface, (14, 18, 28), guide_box, border_radius=12)
        pygame.draw.rect(surface, (78, 205, 196), guide_box, 1, border_radius=12)
        guide = self.small_font.render("스킬창의 손모양 순서대로 Q / W / E를 입력하면 조합 마법이 준비됨", True, COLOR_WHITE)
        surface.blit(guide, guide.get_rect(center=guide_box.center))

        pygame.draw.rect(surface, (34, 42, 66), self.start_button, border_radius=16)
        pygame.draw.rect(surface, COLOR_MUTED, self.start_button, 2, border_radius=16)
        label = self.font.render("전투 시작", True, COLOR_WHITE)
        surface.blit(label, label.get_rect(center=self.start_button.center))

        bottom_guide = self.small_font.render("Enter / Space 또는 버튼 클릭", True, COLOR_MUTED)
        surface.blit(bottom_guide, bottom_guide.get_rect(center=(SCREEN_WIDTH // 2, self.start_button.bottom + 28)))

    def _draw_gesture_control_row(self, surface: pygame.Surface, x: int, y: int, key: str, gesture: str, desc: str) -> None:
        key_rect = pygame.Rect(x, y + 10, 52, 44)
        pygame.draw.rect(surface, (8, 10, 14), key_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_MUTED, key_rect, 2, border_radius=8)
        key_text = self.font.render(key, True, COLOR_WHITE)
        surface.blit(key_text, key_text.get_rect(center=key_rect.center))

        self._gesture_icon_box(surface, x + 74, y, gesture, 64, 64)

        desc_text = self.font.render(desc, True, COLOR_WHITE)
        surface.blit(desc_text, (x + 158, y + 17))

    def _draw_text_control_row(self, surface: pygame.Surface, x: int, y: int, key: str, desc: str) -> None:
        key_rect = pygame.Rect(x, y + 4, 138, 38)
        pygame.draw.rect(surface, (8, 10, 14), key_rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_MUTED, key_rect, 2, border_radius=8)
        key_text = self.small_font.render(key, True, COLOR_WHITE)
        surface.blit(key_text, key_text.get_rect(center=key_rect.center))

        desc_text = self.small_font.render(desc, True, COLOR_WHITE)
        surface.blit(desc_text, (x + 156, y + 10))

    def _gesture_icon_box(self, surface: pygame.Surface, x: int, y: int, gesture: str, w: int, h: int) -> None:
        rect = pygame.Rect(x, y, w, h)
        glow = GESTURE_COLORS.get(gesture, (100, 220, 255))
        pygame.draw.rect(surface, (14, 18, 25), rect, border_radius=8)
        pygame.draw.rect(surface, tuple(min(255, int(v * 0.85)) for v in glow), rect, 2, border_radius=8)

        icon = self.gesture_icons.get(gesture)
        if icon is not None:
            icon_size = min(rect.w - 10, rect.h - 10)
            icon_image = pygame.transform.smoothscale(icon, (icon_size, icon_size))
            icon_rect = icon_image.get_rect(center=rect.center)
            surface.blit(icon_image, icon_rect)
            return

        fallback = self.tiny_font.render(gesture, True, COLOR_WHITE)
        surface.blit(fallback, fallback.get_rect(center=rect.center))
