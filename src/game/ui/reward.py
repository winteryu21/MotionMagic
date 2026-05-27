"""보상 UI — 웨이브 클리어 후 3개 카드 선택 오버레이."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import pygame

from src.game.game_manager import GameManager
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH
from src.game.systems.spell_data import SPELL_REGISTRY, SpellData
from src.game.ui.fonts import get_font

# ── 레이아웃 ──────────────────────────────────────────────────────────────
_CARD_W: int = 390
_CARD_H: int = 290
_CARD_GAP: int = 55

_HEADER_CY: int = 260   # 웨이브 클리어 텍스트 중앙 y
_SUB_CY: int = 315      # 부제 텍스트 중앙 y
_CARD_Y: int = 370      # 카드 상단 y

# ── 색상 ──────────────────────────────────────────────────────────────────
_DIM_COLOR = (0, 0, 0, 195)
_CARD_BG = (16, 20, 34)
_DIVIDER = (45, 50, 68)
_TEXT_BODY = (168, 174, 196)
_TEXT_TITLE = (238, 241, 255)
_GOLD = (241, 196, 15)


# ── 보상 옵션 데이터 ──────────────────────────────────────────────────────

@dataclass
class RewardOption:
    """카드 하나의 데이터 컨테이너."""
    title: str
    subtitle: str
    description: str                    # \n 으로 줄바꿈
    color: tuple[int, int, int]
    apply_fn: Callable[[GameManager], None]


# ── 플레이어 강화 적용 함수 (람다 대신 네임드 함수로 명확하게) ───────────

def _apply_max_hp(gm: GameManager) -> None:
    gm.max_hp += 20
    gm.hp = min(gm.max_hp, gm.hp + 20)


def _apply_max_mp(gm: GameManager) -> None:
    gm.max_mp += 20


def _apply_mana_regen(gm: GameManager) -> None:
    gm.mana_regen_rate += 2.0


def _apply_cooldown_reduction(gm: GameManager) -> None:
    gm.cooldown_reduction = min(0.5, gm.cooldown_reduction + 0.10)


_PLAYER_UPGRADES: list[RewardOption] = [
    RewardOption(
        title="최대 체력 +20",
        subtitle="플레이어 강화",
        description="최대 HP가 20 증가합니다.\n현재 HP도 20 즉시 회복합니다.",
        color=(76, 209, 55),
        apply_fn=_apply_max_hp,
    ),
    RewardOption(
        title="최대 마나 +20",
        subtitle="플레이어 강화",
        description="최대 MP가 20 증가합니다.",
        color=(72, 126, 255),
        apply_fn=_apply_max_mp,
    ),
    RewardOption(
        title="마나 회복 +2/s",
        subtitle="플레이어 강화",
        description="초당 마나 회복량이 2 증가합니다.",
        color=(155, 89, 182),
        apply_fn=_apply_mana_regen,
    ),
    RewardOption(
        title="쿨타임 감소 +10%",
        subtitle="플레이어 강화",
        description="모든 마법 쿨타임이 10% 감소합니다.\n(최대 50%까지 중첩 가능)",
        color=_GOLD,
        apply_fn=_apply_cooldown_reduction,
    ),
]


# ── 마법 보상 생성 함수 ───────────────────────────────────────────────────

def _make_unlock(spell: SpellData) -> RewardOption:
    """잠긴 마법을 해금하는 보상 카드를 생성한다."""
    desc = (
        f"MP {int(spell.mp_cost)}  |  쿨타임 {spell.base_cooldown:.1f}s\n"
        f"기본 데미지 {int(spell.base_damage)}"
    )
    return RewardOption(
        title=f"{spell.display_name} 해금",
        subtitle="새 마법 획득",
        description=desc,
        color=spell.color,
        apply_fn=lambda gm: spell.unlock(),
    )


def _make_levelup(spell: SpellData) -> RewardOption:
    """해금된 마법을 한 단계 강화하는 보상 카드를 생성한다."""
    bonus_lines = ["데미지 +20%  |  쿨타임 -8%"]
    if spell.name == "piercing_bullet":
        bonus_lines.append(f"관통 수  {spell.pierce_count}회 → {spell.pierce_count + 1}회")
    elif spell.name == "fireball":
        bonus_lines.append("폭발 범위 +15%")
    elif spell.name == "chain_lightning":
        bonus_lines.append(f"체인 수  {spell.chain_count}회 → {spell.chain_count + 1}회")

    return RewardOption(
        title=f"{spell.display_name} 강화",
        subtitle=f"Lv.{spell.level}  →  Lv.{spell.level + 1}",
        description="\n".join(bonus_lines),
        color=spell.color,
        apply_fn=lambda gm: spell.level_up(),
    )


# ── 보상 풀 생성 ──────────────────────────────────────────────────────────

def generate_options(gm: GameManager, count: int = 3) -> list[RewardOption]:
    """현재 게임 상태를 기반으로 보상 풀을 구성하고 count개를 무작위 반환한다."""
    pool: list[RewardOption] = []

    for spell in SPELL_REGISTRY.values():
        if not spell.unlocked:
            pool.append(_make_unlock(spell))
        else:
            pool.append(_make_levelup(spell))

    for upg in _PLAYER_UPGRADES:
        if "쿨타임" in upg.title and gm.cooldown_reduction >= 0.5:
            continue
        pool.append(upg)

    random.shuffle(pool)
    return pool[:count]


# ── 오버레이 클래스 ───────────────────────────────────────────────────────

class RewardOverlay:
    """웨이브 클리어 시 화면에 표시되는 보상 선택 오버레이.

    게임은 오버레이가 활성화된 동안 일시정지된다.
    마우스 클릭 또는 숫자키 1/2/3으로 카드를 선택한다.
    """

    def __init__(self, game_manager: GameManager) -> None:
        self.game_manager = game_manager
        self.options: list[RewardOption] = generate_options(game_manager)
        self.active: bool = True
        self._hovered: int = -1

        # 폰트 (맑은 고딕 — 한국어 지원)
        self._f_header = get_font(46, bold=True)
        self._f_sub = get_font(26)
        self._f_card_title = get_font(28, bold=True)
        self._f_card_badge = get_font(20, bold=True)
        self._f_body = get_font(19)
        self._f_hint = get_font(20, bold=True)
        self._f_key = get_font(17, bold=True)

        # 카드 Rect (카드 수에 따라 동적 중앙 정렬)
        n = len(self.options)
        total_w = n * _CARD_W + (n - 1) * _CARD_GAP
        sx = (SCREEN_WIDTH - total_w) // 2
        self._card_rects: list[pygame.Rect] = [
            pygame.Rect(sx + i * (_CARD_W + _CARD_GAP), _CARD_Y, _CARD_W, _CARD_H)
            for i in range(n)
        ]

        # 전체 화면 dim 레이어 (매 프레임 재생성 방지)
        self._dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self._dim.fill(_DIM_COLOR)

    # ── 이벤트 처리 ───────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> None:
        """오버레이 관련 이벤트를 처리한다."""
        if not self.active:
            return

        if event.type == pygame.MOUSEMOTION:
            self._hovered = self._hit_card(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            idx = self._hit_card(event.pos)
            if idx >= 0:
                self._select(idx)

        elif event.type == pygame.KEYDOWN:
            key_map = {
                pygame.K_1: 0, pygame.K_KP1: 0,
                pygame.K_2: 1, pygame.K_KP2: 1,
                pygame.K_3: 2, pygame.K_KP3: 2,
            }
            idx = key_map.get(event.key, -1)
            if 0 <= idx < len(self.options):
                self._select(idx)

    # ── 렌더링 ────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        """오버레이 전체를 렌더링한다."""
        if not self.active:
            return

        # 반투명 배경
        screen.blit(self._dim, (0, 0))

        # 헤더 텍스트
        cx = SCREEN_WIDTH // 2
        wave = self.game_manager.current_wave
        h1 = self._f_header.render(f"WAVE  {wave - 1}  클리어!", True, _GOLD)
        h2 = self._f_sub.render("보상을 하나 선택하세요  (1 / 2 / 3)", True, (195, 200, 215))
        screen.blit(h1, h1.get_rect(centerx=cx, centery=_HEADER_CY))
        screen.blit(h2, h2.get_rect(centerx=cx, centery=_SUB_CY))

        # 카드
        for i, (opt, rect) in enumerate(zip(self.options, self._card_rects)):
            self._draw_card(screen, opt, rect, idx=i, hovered=(i == self._hovered))

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _hit_card(self, pos: tuple[int, int]) -> int:
        for i, rect in enumerate(self._card_rects):
            if rect.collidepoint(pos):
                return i
        return -1

    def _select(self, idx: int) -> None:
        self.options[idx].apply_fn(self.game_manager)
        self.active = False
        self.game_manager.reward_pending = False

    def _draw_card(
        self,
        screen: pygame.Surface,
        opt: RewardOption,
        rect: pygame.Rect,
        idx: int,
        hovered: bool,
    ) -> None:
        # ① 배경
        pygame.draw.rect(screen, _CARD_BG, rect, border_radius=14)

        # ② 호버 글로우
        if hovered:
            glow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            glow.fill((255, 255, 255, 20))
            screen.blit(glow, rect.topleft)

        # ③ 테두리 (호버 시 더 밝고 두껍게)
        border_col = tuple(min(255, c + 80) for c in opt.color) if hovered else opt.color
        pygame.draw.rect(screen, border_col, rect,
                         width=(4 if hovered else 2), border_radius=14)

        # ④ 상단 컬러 바
        top = pygame.Rect(rect.x + 2, rect.y + 2, rect.w - 4, 8)
        pygame.draw.rect(screen, opt.color, top, border_radius=6)

        # ⑤ 단축키 배지 (우상단)
        key_bg = pygame.Rect(rect.right - 38, rect.y + 14, 26, 26)
        pygame.draw.rect(screen, opt.color, key_bg, border_radius=5)
        key_surf = self._f_key.render(str(idx + 1), True, (10, 10, 10))
        screen.blit(key_surf, key_surf.get_rect(center=key_bg.center))

        # ⑥ 텍스트 (padding 18px)
        pad = 18
        tx, ty = rect.x + pad, rect.y + 26

        # 제목
        title_s = self._f_card_title.render(opt.title, True, _TEXT_TITLE)
        screen.blit(title_s, (tx, ty))
        ty += title_s.get_height() + 6

        # 부제목 배지
        badge_s = self._f_card_badge.render(opt.subtitle, True, opt.color)
        screen.blit(badge_s, (tx, ty))
        ty += badge_s.get_height() + 14

        # 구분선
        pygame.draw.line(screen, _DIVIDER,
                         (rect.x + pad, ty), (rect.right - pad, ty), 1)
        ty += 10

        # 설명 (줄바꿈)
        for line in opt.description.split("\n"):
            ls = self._f_body.render(line, True, _TEXT_BODY)
            screen.blit(ls, (tx, ty))
            ty += ls.get_height() + 5

        # ⑦ 호버 힌트
        if hovered:
            hint = self._f_hint.render("▶  클릭하여 선택", True, _GOLD)
            screen.blit(hint, hint.get_rect(centerx=rect.centerx, bottom=rect.bottom - 14))
