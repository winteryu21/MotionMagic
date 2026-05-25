"""타이틀 화면 씬."""

from __future__ import annotations

import pygame

from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.fonts import get_font


class TitleScene:
    """게임 시작 전 타이틀 화면."""

    mouse_visible = True

    def __init__(self) -> None:
        self.next_scene: str | None = None
        self.title_font = get_font(72, bold=True)
        self.button_font = get_font(30, bold=True)
        self.small_font = get_font(20)
        self.start_button = pygame.Rect(SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 + 55, 260, 70)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.next_scene = "explain"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.start_button.collidepoint(event.pos):
                self.next_scene = "explain"

    def update(self, dt: float) -> None:
        return

    def draw(self, surface: pygame.Surface) -> None:
        title = self.title_font.render("motion-maigc", True, COLOR_WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        surface.blit(title, title_rect)

        pygame.draw.rect(surface, (34, 42, 66), self.start_button, border_radius=18)
        pygame.draw.rect(surface, COLOR_MUTED, self.start_button, 2, border_radius=18)
        label = self.button_font.render("시작", True, COLOR_WHITE)
        surface.blit(label, label.get_rect(center=self.start_button.center))

        guide = self.small_font.render("Enter / Space 또는 버튼 클릭", True, COLOR_MUTED)
        surface.blit(guide, guide.get_rect(center=(SCREEN_WIDTH // 2, self.start_button.bottom + 38)))
