"""에임 크로스헤어 렌더링."""

from __future__ import annotations

import pygame

from src.game.settings import COLOR_WHITE


class Crosshair:
    def draw(self, surface: pygame.Surface, pos: tuple[int, int]) -> None:
        x, y = pos
        pygame.draw.circle(surface, COLOR_WHITE, (x, y), 12, 1)
        pygame.draw.line(surface, COLOR_WHITE, (x - 18, y), (x - 6, y), 1)
        pygame.draw.line(surface, COLOR_WHITE, (x + 6, y), (x + 18, y), 1)
        pygame.draw.line(surface, COLOR_WHITE, (x, y - 18), (x, y - 6), 1)
        pygame.draw.line(surface, COLOR_WHITE, (x, y + 6), (x, y + 18), 1)
