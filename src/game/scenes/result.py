"""게임 오버 결과 화면 씬."""

from __future__ import annotations

import pygame

from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.fonts import get_font


class ResultScene:
    """플레이어 라이프가 0이 되었을 때 표시되는 결과 화면."""

    mouse_visible = True

    def __init__(self, cleared_stage: int) -> None:
        self.next_scene: str | None = None
        self.cleared_stage = max(0, cleared_stage)
        self.title_font = get_font(72, bold=True)
        self.stage_font = get_font(32, bold=True)
        self.guide_font = get_font(22)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self.next_scene = "title"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.next_scene = "title"

    def update(self, dt: float) -> None:
        return

    def draw(self, surface: pygame.Surface) -> None:
        panel = pygame.Rect(SCREEN_WIDTH // 2 - 330, SCREEN_HEIGHT // 2 - 180, 660, 360)
        pygame.draw.rect(surface, (24, 27, 42), panel, border_radius=24)
        pygame.draw.rect(surface, (120, 70, 80), panel, 3, border_radius=24)

        title = self.title_font.render("GAME OVER", True, (255, 110, 105))
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 90)))

        stage_text = self.stage_font.render(f"최종 클리어 스테이지 : {self.cleared_stage}", True, COLOR_WHITE)
        surface.blit(stage_text, stage_text.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 185)))

        guide = self.guide_font.render("아무 키나 누르면 메인 화면으로 돌아감", True, COLOR_MUTED)
        surface.blit(guide, guide.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 275)))
