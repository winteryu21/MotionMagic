"""에임 크로스헤어 렌더링."""

from __future__ import annotations

import pygame


def draw_crosshair(
    surface: pygame.Surface,
    rx: float,
    ry: float,
    charged: bool = False,
) -> None:
    """크로스헤어(조준점)를 지정된 Surface 위에 그린다.

    Args:
        surface: 렌더링 대상 pygame.Surface.
        rx: X 위치 비율 (0.0 ~ 1.0, Surface 너비 기준).
        ry: Y 위치 비율 (0.0 ~ 1.0, Surface 높이 기준).
        charged: 마법 발동 대기 상태이면 True. 크로스헤어 색상과 크기가 변경된다.
    """
    x = int(rx * surface.get_width())
    y = int(ry * surface.get_height())

    if charged:
        # 마법 충전됨 — 금색, 외곽 원 크게, 선 두껍게
        COLOR_OUTER = (241, 196, 15, 230)  # 금색
        COLOR_INNER = (255, 230, 80, 255)
        R = 18
        ARM = 12
        W = 3
        DOT_R = 4
        # 충전 표시 추가 외곽 링
        pygame.draw.circle(surface, (241, 196, 15, 80), (x, y), R + 8, 1)
    else:
        # 기본 — 흰색 외곽 + 붉은 십자선
        COLOR_OUTER = (255, 255, 255, 200)
        COLOR_INNER = (255, 80, 80, 255)
        R = 14
        ARM = 10
        W = 2
        DOT_R = 2

    pygame.draw.circle(surface, COLOR_OUTER, (x, y), R, W)
    pygame.draw.line(surface, COLOR_INNER, (x - R - ARM, y), (x - R + 2, y), W)
    pygame.draw.line(surface, COLOR_INNER, (x + R - 2, y), (x + R + ARM, y), W)
    pygame.draw.line(surface, COLOR_INNER, (x, y - R - ARM), (x, y - R + 2), W)
    pygame.draw.line(surface, COLOR_INNER, (x, y + R - 2), (x, y + R + ARM), W)
    pygame.draw.circle(surface, COLOR_INNER, (x, y), DOT_R)
