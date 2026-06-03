"""타이틀 화면 씬."""

from __future__ import annotations

import pygame

from src.bridge.gesture_event import GestureEvent
from src.game.gesture_input import screen_pos_from_gesture_event
from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.crosshair import Crosshair
from src.game.ui.fonts import get_font


class TitleScene:
    """게임 시작 전 타이틀 화면."""

    mouse_visible = False

    def __init__(self) -> None:
        self.next_scene: str | None = None
        self.title_font = get_font(72, bold=True)
        self.button_font = get_font(30, bold=True)
        self.small_font = get_font(20)
        self.start_button = pygame.Rect(
            SCREEN_WIDTH // 2 - 130,
            SCREEN_HEIGHT // 2 + 55,
            260,
            70,
        )
        self.aim_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.crosshair = Crosshair()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_RETURN,
            pygame.K_SPACE,
        ):
            self.next_scene = "explain"
        elif event.type == pygame.MOUSEMOTION:
            self.aim_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.aim_pos = event.pos
            self._select_at(self.aim_pos)

    def handle_gesture_event(self, event: GestureEvent) -> None:
        """오른손 조준점으로 시작 버튼을 선택한다.

        Args:
            event: bridge 계층에서 전달된 제스처 이벤트.
        """
        if event.kind not in {"aim", "fire"}:
            return

        self.aim_pos = screen_pos_from_gesture_event(event)
        if event.kind == "fire":
            self._select_at(self.aim_pos)

    def _select_at(self, pos: tuple[int, int]) -> None:
        if self.start_button.collidepoint(pos):
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

        guide = self.small_font.render(
            "Enter / Space 또는 버튼 클릭", True, COLOR_MUTED
        )
        surface.blit(
            guide,
            guide.get_rect(center=(SCREEN_WIDTH // 2, self.start_button.bottom + 38)),
        )
        self.crosshair.draw(surface, self.aim_pos)
