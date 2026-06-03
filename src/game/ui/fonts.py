"""pygame 한글 폰트 유틸리티."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import pygame


KOREAN_FONT_CANDIDATES = [
    "malgungothic",      # Windows: 맑은 고딕
    "malgun gothic",
    "applegothic",      # macOS
    "nanumgothic",      # Linux/optional
    "notosanscjkkr",
    "notosanskr",
    "unifont",
]


@lru_cache(maxsize=32)
def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """한글이 깨지지 않는 시스템 폰트를 우선 사용함."""
    for name in KOREAN_FONT_CANDIDATES:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)

    if sys.platform.startswith("win"):
        windows_font = Path("C:/Windows/Fonts/malgun.ttf")
        if windows_font.exists():
            return pygame.font.Font(str(windows_font), size)

    return pygame.font.Font(None, size)
