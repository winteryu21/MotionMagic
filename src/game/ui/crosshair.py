"""에임 크로스헤어 렌더링."""

from __future__ import annotations

import pygame

from src.game.settings import COLOR_CROSSHAIR


class Crosshair:
    """전투 조준점을 렌더링한다."""

    def draw(self, surface: pygame.Surface, pos: tuple[int, int]) -> None:
        """지정 좌표에 조준점을 그린다.

        Args:
            surface: 조준점을 그릴 pygame Surface.
            pos: 화면 좌표 ``(x, y)``.
        """
        x, y = pos
        pygame.draw.circle(surface, COLOR_CROSSHAIR, (x, y), 13, 2)
        pygame.draw.line(surface, COLOR_CROSSHAIR, (x - 22, y), (x - 7, y), 2)
        pygame.draw.line(surface, COLOR_CROSSHAIR, (x + 7, y), (x + 22, y), 2)
        pygame.draw.line(surface, COLOR_CROSSHAIR, (x, y - 22), (x, y - 7), 2)
        pygame.draw.line(surface, COLOR_CROSSHAIR, (x, y + 7), (x, y + 22), 2)
