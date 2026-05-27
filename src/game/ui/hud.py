"""HUD — 체력바, 마나바, 마법 쿨다운 핫바, 웨이브 표시."""

from __future__ import annotations

import pygame

from src.game.game_manager import GameManager
from src.game.settings import COLOR_HUD_HP, COLOR_HUD_MP, COLOR_WHITE, SCREEN_WIDTH
from src.game.systems.spell_data import SPELL_REGISTRY
from src.game.ui.fonts import get_font

# 잠긴 마법 테두리 색상
_COLOR_LOCKED = (80, 80, 90)


class HUD:
    """게임 HUD 렌더러.

    HP 바, MP 바, 마나 충전 텍스트, 스킬 핫바(쿨타임·잠금·레벨 시각화),
    웨이브 번호를 메인 화면에 렌더링한다.
    """

    def __init__(self) -> None:
        self.font: pygame.font.Font = get_font(28, bold=True)
        self.font_small: pygame.font.Font = get_font(20, bold=True)

    def draw(
        self,
        screen: pygame.Surface,
        game_manager: GameManager,
        is_recharging: bool,
        selected_spell: str | None = None,
    ) -> None:
        self._draw_hp_bar(screen, game_manager)
        self._draw_mp_bar(screen, game_manager)
        self._draw_recharge_text(screen, is_recharging)
        self._draw_wave(screen, game_manager)
        self._draw_skill_hotbar(screen, game_manager, selected_spell)
        if selected_spell is not None:
            self._draw_selected_indicator(screen, selected_spell)

    # ── 내부 렌더링 헬퍼 ──────────────────────────────────────────────────

    def _draw_hp_bar(self, screen: pygame.Surface, gm: GameManager) -> None:
        bar_w, bar_h = 300, 25
        ratio = gm.hp / gm.max_hp
        pygame.draw.rect(screen, (44, 62, 80), (50, 50, bar_w, bar_h))
        pygame.draw.rect(screen, COLOR_HUD_HP, (50, 50, int(bar_w * ratio), bar_h))
        text = self.font.render(f"HP: {int(gm.hp)}/{int(gm.max_hp)}", True, COLOR_WHITE)
        screen.blit(text, (50, 15))

    def _draw_mp_bar(self, screen: pygame.Surface, gm: GameManager) -> None:
        bar_w, bar_h = 300, 25
        ratio = gm.mp / gm.max_mp
        pygame.draw.rect(screen, (44, 62, 80), (50, 120, bar_w, bar_h))
        pygame.draw.rect(screen, COLOR_HUD_MP, (50, 120, int(bar_w * ratio), bar_h))
        text = self.font.render(f"MP: {int(gm.mp)}/{int(gm.max_mp)}", True, COLOR_WHITE)
        screen.blit(text, (50, 85))

    def _draw_recharge_text(self, screen: pygame.Surface, is_recharging: bool) -> None:
        if is_recharging:
            text = self.font.render("RECHARGING MANA...", True, (241, 196, 15))
            screen.blit(text, (50, 160))

    def _draw_wave(self, screen: pygame.Surface, gm: GameManager) -> None:
        text = self.font.render(f"WAVE  {gm.current_wave}", True, (241, 196, 15))
        screen.blit(text, (50, 195))

    def _draw_skill_hotbar(
        self,
        screen: pygame.Surface,
        gm: GameManager,
        selected_spell: str | None = None,
    ) -> None:
        """우상단에 스킬 핫바를 그린다. SPELL_REGISTRY 순서로 렌더링."""
        start_x = SCREEN_WIDTH - 380
        start_y = 30
        box_w, box_h = 330, 50

        for spell in SPELL_REGISTRY.values():
            cd = gm.spell_cooldowns.get(spell.name, 0.0)
            is_selected = spell.name == selected_spell
            border_color = spell.color if spell.unlocked else _COLOR_LOCKED

            # 선택된 마법: 골드 글로우 배경
            if is_selected:
                glow = pygame.Surface((box_w + 8, box_h + 8), pygame.SRCALPHA)
                glow.fill((241, 196, 15, 60))
                screen.blit(glow, (start_x - 4, start_y - 4))

            # 박스 배경 및 테두리
            pygame.draw.rect(screen, (20, 20, 30), (start_x, start_y, box_w, box_h), border_radius=6)

            # 선택 시: 골드 두꺼운 테두리, 평상시: 마법 색 테두리
            if is_selected:
                pygame.draw.rect(screen, (241, 196, 15), (start_x, start_y, box_w, box_h), width=3, border_radius=6)
            else:
                pygame.draw.rect(screen, border_color, (start_x, start_y, box_w, box_h), width=2, border_radius=6)

            if spell.unlocked:
                # 마법명 + 단축키 + MP 비용
                label = f"{spell.hotkey}: {spell.display_name}  ({int(spell.mp_cost)} MP)"
                text_color = (241, 196, 15) if is_selected else COLOR_WHITE
                text_surf = self.font_small.render(label, True, text_color)
                screen.blit(text_surf, (start_x + 10, start_y + 8))

                # 레벨 배지 (Lv.2 이상일 때만)
                if spell.level > 1:
                    lv_text = self.font_small.render(f"Lv.{spell.level}", True, (241, 196, 15))
                    screen.blit(lv_text, (start_x + box_w - 55, start_y + 8))

                # 선택 마크 (▶)
                if is_selected:
                    mark = self.font_small.render("▶", True, (241, 196, 15))
                    screen.blit(mark, (start_x - 22, start_y + 14))

                # 쿨타임 오버레이
                if cd > 0.0:
                    mask = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
                    mask.fill((0, 0, 0, 180))
                    screen.blit(mask, (start_x, start_y))
                    cd_text = self.font.render(f"{cd:.1f}s", True, (231, 76, 60))
                    screen.blit(cd_text, (start_x + box_w - 70, start_y + 10))
                elif gm.mp < spell.mp_cost:
                    mask = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
                    mask.fill((0, 0, 0, 128))
                    screen.blit(mask, (start_x, start_y))
            else:
                # 잠금 상태
                mask = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
                mask.fill((0, 0, 0, 200))
                screen.blit(mask, (start_x, start_y))
                label = f"{spell.hotkey}: {spell.display_name}"
                text_surf = self.font_small.render(label, True, (120, 120, 130))
                screen.blit(text_surf, (start_x + 10, start_y + 8))
                lock_surf = self.font_small.render("LOCKED", True, (180, 60, 60))
                screen.blit(lock_surf, (start_x + box_w - 80, start_y + 8))

            start_y += box_h + 15

    def _draw_selected_indicator(self, screen: pygame.Surface, selected_spell: str) -> None:
        """화면 하단 중앙에 선택된 마법 이름과 클릭 안내를 표시한다."""
        spell = SPELL_REGISTRY.get(selected_spell)
        color = spell.color if spell else (241, 196, 15)
        name = spell.display_name if spell else selected_spell

        text = self.font.render(f"[ {name} ]  —  클릭하여 발동", True, color)
        x = (SCREEN_WIDTH - text.get_width()) // 2
        screen.blit(text, (x, 1040))
