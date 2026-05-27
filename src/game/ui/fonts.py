"""한국어 지원 폰트 로더.

Windows 기본 내장 폰트인 맑은 고딕(Malgun Gothic)을 우선 로드하고,
파일이 없는 환경에서는 pygame SysFont로 폴백한다.
"""

from __future__ import annotations

import os

import pygame

# Windows 맑은 고딕 경로
_MALGUN_REGULAR = "C:/Windows/Fonts/malgun.ttf"
_MALGUN_BOLD = "C:/Windows/Fonts/malgunbd.ttf"

# 캐시: (size, bold) → Font
_cache: dict[tuple[int, bool], pygame.font.Font] = {}


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """한국어를 지원하는 pygame.font.Font 인스턴스를 반환한다.

    같은 (size, bold) 조합은 캐시하여 재사용한다.

    Args:
        size: 포인트 크기.
        bold: True이면 볼드 폰트를 시도한다.
    """
    key = (size, bold)
    if key in _cache:
        return _cache[key]

    font = _load(size, bold)
    _cache[key] = font
    return font


def _load(size: int, bold: bool) -> pygame.font.Font:
    # 볼드 요청 시 malgunbd.ttf 우선, 없으면 일반 파일로 폴백
    if bold and os.path.exists(_MALGUN_BOLD):
        return pygame.font.Font(_MALGUN_BOLD, size)
    if os.path.exists(_MALGUN_REGULAR):
        return pygame.font.Font(_MALGUN_REGULAR, size)
    # 시스템 폰트 폴백 (타 OS 호환)
    return pygame.font.SysFont("malgun gothic", size, bold=bold)
