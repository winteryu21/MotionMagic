"""전투 씬 — Tab으로 전환하는 전체화면 전장 2개."""

from __future__ import annotations

from pathlib import Path

import pygame

from src.game.entities.player import Player
from src.game.game_manager import GameManager
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, TILE_SIZE
from src.game.systems.magic import MagicSystem
from src.game.systems.spawner import Spawner
from src.game.ui.crosshair import draw_crosshair
from src.game.ui.fonts import get_font

# 배경 이미지 경로 (Field 0 → ice, Field 1 → fire)
_BG_PATHS: tuple[str, str] = (
    "assets/maps/ice_battle.png",
    "assets/maps/fire_battle.png",
)

# Field별 테마 색상 (배경 로드 실패 시 폴백 + 번개 이펙트)
_FIELD_SKY_COLOR: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (24, 28, 40),   # Field 0 — 차가운 남색
    (32, 18, 28),   # Field 1 — 뜨거운 붉은 계열
)
_FIELD_GROUND_COLOR: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (41, 128, 185),   # Field 0 — 얼음 블루
    (142, 44, 44),    # Field 1 — 용암 적색
)
_FIELD_LIGHTNING_COLOR: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (52, 152, 219),   # Field 0
    (155, 89, 182),   # Field 1
)

# 미니맵 레이아웃
_MINI_W: int = 260
_MINI_H: int = 146
_MINI_MARGIN: int = 18


class BattleScene:
    """전투 씬 클래스.

    두 개의 독립된 전장(Field 0: 얼음, Field 1: 화염)을 각각 전체 화면으로 렌더링하며,
    Tab 키로 활성 전장을 전환한다. 비활성 전장은 우하단 미니맵으로 표시된다.
    """

    def __init__(self) -> None:
        """전투 씬을 초기화한다."""
        self.game_manager = GameManager()

        # 스프라이트 그룹
        self.all_sprites = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        self.projectiles = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()

        # 전투 시스템
        from src.game.systems.combat import CombatSystem
        self.combat = CombatSystem(self)

        # 전체화면 뷰 크기
        self.view_width: int = SCREEN_WIDTH
        self.view_height: int = SCREEN_HEIGHT

        # 각 전장 렌더링용 서브 Surface (캐싱)
        self.field_surfaces: list[pygame.Surface] = [
            pygame.Surface((self.view_width, self.view_height)),
            pygame.Surface((self.view_width, self.view_height)),
        ]

        # 미니맵 Surface
        self.mini_surface = pygame.Surface((_MINI_W, _MINI_H))

        # 배경 이미지 로드
        self.backgrounds: list[pygame.Surface | None] = self._load_backgrounds()

        # 플레이어 Y 좌표 — 지면(TILE_SIZE) 위
        ground_y = self.view_height - TILE_SIZE
        self.player_y = float(ground_y - TILE_SIZE // 2)

        # Field 0 플레이어 (좌측), Field 1 플레이어 (우측)
        self.player_left = Player(x=200.0, y=self.player_y, field_id=0)
        self.player_right = Player(x=float(SCREEN_WIDTH - 200), y=self.player_y, field_id=1)

        self.players.add(self.player_left, self.player_right)
        self.all_sprites.add(self.player_left, self.player_right)

        # 마법·스폰 시스템
        self.magic_system = MagicSystem(self)
        self.spawner = Spawner(self)

        # 번개 이펙트 목록 [{"path": [...], "timer": float, "field_id": int}]
        self.lightning_effects: list[dict] = []

        # HUD 보조 폰트
        self.font = get_font(22, bold=True)
        self.font_small = get_font(17, bold=True)

    # ── 공개 API ──────────────────────────────────────────────────────────

    def handle_spell_input(
        self,
        spell_name: str,
        target_pos: tuple[float, float] | None = None,
    ) -> bool:
        """마법 발동 요청을 받아 마법 시스템을 격발시킨다.

        Args:
            spell_name: 발동할 마법명.
            target_pos: 정규화 조준 좌표 (x: 0.0~1.0, y: 0.0~1.0). None이면 고정 방향.

        Returns:
            발동 성공 여부.
        """
        return self.magic_system.cast_spell(spell_name, target_pos)

    def add_lightning_effect(self, path: list[tuple[int, int]], field_id: int) -> None:
        """체인 라이트닝 경로를 수신하여 이펙트를 예약한다.

        Args:
            path: 번개가 경유하는 좌표 리스트.
            field_id: 전장 ID.
        """
        self.lightning_effects.append({
            "path": path,
            "timer": 0.18,
            "field_id": field_id,
        })

    # ── 업데이트 ──────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """모든 전장 엔티티를 실시간으로 업데이트한다.

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
        """
        self.spawner.update(dt)

        for effect in self.lightning_effects[:]:
            effect["timer"] -= dt
            if effect["timer"] <= 0.0:
                self.lightning_effects.remove(effect)

        self.all_sprites.update(dt)
        self.combat.update()

    # ── 렌더링 ────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface, selected_spell: str | None = None) -> None:
        """활성 전장을 전체 화면으로, 비활성 전장을 미니맵으로 렌더링한다.

        Args:
            screen: 렌더링할 Pygame 화면 Surface.
            selected_spell: 현재 발동 대기 중인 마법 이름. None이면 대기 없음.
        """
        charged = selected_spell is not None
        active = self.game_manager.active_field
        inactive = 1 - active

        # 두 전장 모두 백버퍼에 그리기 (미니맵·전환 효과 위해)
        self._draw_field(self.field_surfaces[0], field_id=0, active=(active == 0), charged=charged)
        self._draw_field(self.field_surfaces[1], field_id=1, active=(active == 1), charged=charged)

        # ── 1. 활성 전장 → 전체 화면 합성 ──
        screen.blit(self.field_surfaces[active], (0, 0))

        # ── 2. 비활성 전장 → 우하단 미니맵 ──
        self._draw_minimap(screen, inactive)

        # ── 3. 전환 UI: 탭 힌트 (상단 중앙) ──
        self._draw_field_tab_bar(screen, active)

    # ── 내부 렌더링 헬퍼 ──────────────────────────────────────────────────

    def _draw_field(
        self,
        surf: pygame.Surface,
        field_id: int,
        active: bool,
        charged: bool,
    ) -> None:
        """지정된 Surface에 전장 한 개를 완전히 렌더링한다."""
        ground_y = self.view_height - TILE_SIZE
        bg = self.backgrounds[field_id]

        # 배경
        if bg is not None:
            surf.blit(bg, (0, 0))
        else:
            surf.fill(_FIELD_SKY_COLOR[field_id])
            pygame.draw.rect(surf, _FIELD_GROUND_COLOR[field_id],
                             (0, ground_y, self.view_width, TILE_SIZE))
            pygame.draw.rect(surf, (50, 50, 50),
                             (0, ground_y, self.view_width, TILE_SIZE), width=2)

        # 비활성 전장에 어두운 오버레이
        if not active:
            dim = pygame.Surface((self.view_width, self.view_height), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 110))
            surf.blit(dim, (0, 0))

        # 스프라이트 렌더링
        for sprite in self.all_sprites:
            if hasattr(sprite, "field_id") and sprite.field_id == field_id:
                surf.blit(sprite.image, sprite.rect)

        # 번개 이펙트
        lc = _FIELD_LIGHTNING_COLOR[field_id]
        for effect in self.lightning_effects:
            if effect["field_id"] == field_id:
                path = effect["path"]
                for i in range(len(path) - 1):
                    pygame.draw.line(surf, lc, path[i], path[i + 1], 4)
                    pygame.draw.line(surf, (255, 255, 255), path[i], path[i + 1], 1)

        # 크로스헤어 (활성 전장만)
        if active:
            mx, my = pygame.mouse.get_pos()
            rx = max(0.0, min(1.0, mx / self.view_width))
            ry = max(0.0, min(1.0, my / self.view_height))
            draw_crosshair(surf, rx, ry, charged=charged)

    def _draw_minimap(self, screen: pygame.Surface, field_id: int) -> None:
        """비활성 전장을 우하단 미니맵으로 렌더링한다."""
        # 전장 서브 Surface를 미니맵 크기로 스케일링
        pygame.transform.scale(self.field_surfaces[field_id], (_MINI_W, _MINI_H), self.mini_surface)

        mx = self.view_width - _MINI_W - _MINI_MARGIN
        my = self.view_height - _MINI_H - _MINI_MARGIN

        # 테두리 + 배경 패널
        panel_rect = pygame.Rect(mx - 4, my - 4, _MINI_W + 8, _MINI_H + 8)
        pygame.draw.rect(screen, (12, 14, 22), panel_rect, border_radius=10)
        pygame.draw.rect(screen, (80, 90, 110), panel_rect, 2, border_radius=10)

        screen.blit(self.mini_surface, (mx, my))

        # 미니맵 테두리 강조
        pygame.draw.rect(screen, (80, 90, 110),
                         pygame.Rect(mx, my, _MINI_W, _MINI_H), 1, border_radius=4)

        # 라벨
        label = self.font_small.render(
            f"FIELD {field_id}  [TAB]",
            True, (200, 210, 230),
        )
        screen.blit(label, (mx + (_MINI_W - label.get_width()) // 2, my - 22))

    def _draw_field_tab_bar(self, screen: pygame.Surface, active: int) -> None:
        """상단 중앙에 Field 0 / Field 1 탭 인디케이터를 그린다."""
        field_names = ["❄ FIELD 0 — ICE", "🔥 FIELD 1 — FIRE"]
        tab_colors = [(100, 180, 255), (255, 120, 60)]
        tab_w, tab_h = 220, 38
        gap = 12
        total_w = 2 * tab_w + gap
        base_x = (self.view_width - total_w) // 2
        base_y = 14

        for i, (name, color) in enumerate(zip(field_names, tab_colors)):
            tx = base_x + i * (tab_w + gap)
            is_active = (i == active)

            # 탭 배경
            bg_color = (20, 22, 36) if not is_active else (color[0] // 5, color[1] // 5, color[2] // 5)
            pygame.draw.rect(screen, bg_color, (tx, base_y, tab_w, tab_h), border_radius=8)

            # 탭 테두리 (활성: 해당 테마색, 비활성: 회색)
            border_col = color if is_active else (60, 65, 80)
            border_w = 3 if is_active else 1
            pygame.draw.rect(screen, border_col, (tx, base_y, tab_w, tab_h),
                             width=border_w, border_radius=8)

            # 활성 탭 하단 강조선
            if is_active:
                pygame.draw.rect(screen, color, (tx + 8, base_y + tab_h - 4, tab_w - 16, 3),
                                 border_radius=2)

            # 텍스트
            text_color = color if is_active else (100, 108, 128)
            txt = self.font_small.render(name, True, text_color)
            screen.blit(txt, (tx + (tab_w - txt.get_width()) // 2,
                               base_y + (tab_h - txt.get_height()) // 2))

    # ── 초기화 헬퍼 ───────────────────────────────────────────────────────

    def _load_backgrounds(self) -> list[pygame.Surface | None]:
        """배경 이미지를 로드하여 화면 크기에 맞게 스케일링한다."""
        project_root = Path(__file__).resolve().parents[3]
        result: list[pygame.Surface | None] = []
        for path_str in _BG_PATHS:
            full = project_root / path_str
            if full.exists():
                img = pygame.image.load(str(full)).convert()
                img = pygame.transform.smoothscale(img, (self.view_width, self.view_height))
                result.append(img)
            else:
                result.append(None)
        return result
