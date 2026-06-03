"""조작법 설명 씬."""

from __future__ import annotations

from pathlib import Path

import pygame

from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.fonts import get_font


class ExplainScene:
    """전투 시작 전 조작법 이미지를 보여주는 씬."""

    mouse_visible = True

    def __init__(self) -> None:
        self.next_scene: str | None = None
        self.small_font = get_font(20)
        self.start_button = pygame.Rect(SCREEN_WIDTH // 2 - 135, SCREEN_HEIGHT - 74, 270, 52)
        self.guide_image = self._load_guide_image()

    def _load_guide_image(self) -> pygame.Surface | None:
        project_root = Path(__file__).resolve().parents[3]
        image_path = project_root / "assets" / "tutorial" / "gesture_guide.png"
        if not image_path.exists():
            return None
        return pygame.image.load(str(image_path)).convert_alpha()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.next_scene = "battle"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.start_button.collidepoint(event.pos):
                self.next_scene = "battle"

    def update(self, dt: float) -> None:
        return

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((7, 13, 24))

        if self.guide_image is not None:
            max_w = SCREEN_WIDTH - 40
            max_h = SCREEN_HEIGHT - 92
            scale = min(max_w / self.guide_image.get_width(), max_h / self.guide_image.get_height())
            draw_w = max(1, int(self.guide_image.get_width() * scale))
            draw_h = max(1, int(self.guide_image.get_height() * scale))
            image = pygame.transform.smoothscale(self.guide_image, (draw_w, draw_h))
            rect = image.get_rect(center=(SCREEN_WIDTH // 2, 318))
            surface.blit(image, rect)
        else:
            text = self.small_font.render("조작 설명 이미지를 찾을 수 없음", True, COLOR_WHITE)
            surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        pygame.draw.rect(surface, (34, 42, 66), self.start_button, border_radius=14)
        pygame.draw.rect(surface, COLOR_MUTED, self.start_button, 2, border_radius=14)
        label = self.small_font.render("전투 시작", True, COLOR_WHITE)
        surface.blit(label, label.get_rect(center=self.start_button.center))

        guide = self.small_font.render("Enter / Space 또는 버튼 클릭", True, COLOR_MUTED)
        surface.blit(guide, guide.get_rect(center=(SCREEN_WIDTH // 2, self.start_button.top - 18)))
