"""조작법 설명 씬."""

from __future__ import annotations

import pygame

from src.bridge.gesture_event import GestureEvent
from src.game.gesture_input import screen_pos_from_gesture_event
from src.game.settings import COLOR_MUTED, COLOR_WHITE, SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.ui.crosshair import Crosshair
from src.game.ui.fonts import get_font


class ExplainScene:
    """전투 시작 전 조작법을 간단히 안내하는 씬."""

    mouse_visible = False

    def __init__(self) -> None:
        self.next_scene: str | None = None
        self.title_font = get_font(44, bold=True)
        self.font = get_font(25)
        self.small_font = get_font(20)
        self.start_button = pygame.Rect(
            SCREEN_WIDTH // 2 - 135,
            SCREEN_HEIGHT - 135,
            270,
            62,
        )
        self.aim_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.crosshair = Crosshair()
        self.lines = [
            "Q / W / E : 가위 / 바위 / 보 제스처 입력",
            "마우스 이동 : 조준점 이동",
            "마우스 왼쪽 클릭 : 입력한 조합으로 마법 발사",
            "Space 유지 : 마나 충전",
            "Tab : 전장 전환",
            "적이 왼쪽 기지 라인에 닿으면 플레이어 라이프가 감소함",
            "스테이지 클리어 후 보상 3개 중 1개를 선택함",
        ]

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_RETURN,
            pygame.K_SPACE,
        ):
            self.next_scene = "battle"
        elif event.type == pygame.MOUSEMOTION:
            self.aim_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.aim_pos = event.pos
            self._select_at(self.aim_pos)

    def handle_gesture_event(self, event: GestureEvent) -> None:
        """오른손 조준점으로 전투 시작 버튼을 선택한다.

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
            self.next_scene = "battle"

    def update(self, dt: float) -> None:
        return

    def draw(self, surface: pygame.Surface) -> None:
        panel = pygame.Rect(SCREEN_WIDTH // 2 - 390, 88, 780, 500)
        pygame.draw.rect(surface, (22, 27, 43), panel, border_radius=22)
        pygame.draw.rect(surface, COLOR_MUTED, panel, 2, border_radius=22)

        title = self.title_font.render("조작법", True, COLOR_WHITE)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, panel.y + 58)))

        y = panel.y + 125
        for line in self.lines:
            text = self.font.render(line, True, COLOR_WHITE)
            surface.blit(text, (panel.x + 70, y))
            y += 48

        pygame.draw.rect(surface, (34, 42, 66), self.start_button, border_radius=16)
        pygame.draw.rect(surface, COLOR_MUTED, self.start_button, 2, border_radius=16)
        label = self.font.render("전투 시작", True, COLOR_WHITE)
        surface.blit(label, label.get_rect(center=self.start_button.center))

        guide = self.small_font.render(
            "Enter / Space 또는 버튼 클릭", True, COLOR_MUTED
        )
        surface.blit(
            guide,
            guide.get_rect(center=(SCREEN_WIDTH // 2, self.start_button.bottom + 30)),
        )
        self.crosshair.draw(surface, self.aim_pos)
