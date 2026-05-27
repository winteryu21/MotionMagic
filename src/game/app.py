"""메인 게임 루프 및 씬 관리."""

from __future__ import annotations

import logging
import sys

import pygame

from src.bridge.gesture_event import GESTURE_EVENT, GestureEvent
from src.game.game_manager import GameManager
from src.game.scenes.battle import BattleScene
from src.game.settings import FPS, SCREEN_HEIGHT, SCREEN_WIDTH, TITLE
from src.game.ui.hud import HUD
from src.game.ui.reward import RewardOverlay

logger = logging.getLogger(__name__)

# ── 제스처 조합 → 마법 이름 매핑 ────────────────────────────────────────────
# 접두사 매칭으로 2·3티어 조합까지 확장 가능한 구조
_GESTURE_COMBO: dict[tuple[str, ...], str] = {
    # 1티어 — 제스처 1개
    ("scissors",):        "piercing_bullet",  # 가위 → 관통 마탄
    ("fist",):            "fireball",          # 주먹 → 화염구
    ("palm",):            "chain_lightning",   # 보 → 체인 라이트닝
    # 2티어 — 제스처 2개 (미구현 마법은 가장 유사한 기존 마법으로 임시 매핑)
    ("palm", "scissors"): "piercing_bullet",  # 보+가위 → 마탄
    ("scissors", "fist"): "piercing_bullet",  # 가위+주먹 → 낙뢰 (미구현)
    ("fist", "palm"):     "fireball",          # 주먹+보 → 폭발 (미구현)
}
_TRIGGER_GESTURE: str = "point"       # 이 제스처가 오면 선택된 마법을 즉시 발동
_MAX_GESTURE_BUFFER: int = 3          # 조합 버퍼 최대 길이
_GESTURE_BUFFER_TIMEOUT: float = 2.0  # 마지막 입력 후 이 시간이 지나면 버퍼 초기화(초)


class App:
    """게임 애플리케이션. 초기화, 메인 루프, 씬 전환을 관리한다."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.running = True

        self.game_manager = GameManager()
        self.is_recharging = False
        self.battle_scene = BattleScene()
        self.hud = HUD()
        self.reward_overlay: RewardOverlay | None = None

        # 마법 선택 상태
        self.selected_spell: str | None = None    # 발동 대기 중인 마법 이름
        self._gesture_buffer: list[str] = []      # 조합 빌딩용 제스처 버퍼
        self._buffer_timer: float = 0.0           # 마지막 제스처 이후 경과 시간(초)

        logger.info("MotionMagic 초기화 완료 (%dx%d)", SCREEN_WIDTH, SCREEN_HEIGHT)

    def run(self) -> None:
        """메인 게임 루프."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()

        pygame.quit()
        sys.exit()

    # ── 이벤트 처리 ───────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        """이벤트 처리."""
        self.is_recharging = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            # 보상 오버레이 활성 중: 오버레이에만 전달, 게임 이벤트 차단
            if self.reward_overlay is not None:
                self.reward_overlay.handle_event(event)
                continue

            # ── 키보드 ──────────────────────────────────────────────────
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_TAB:
                    self.game_manager.active_field = 1 - self.game_manager.active_field
                    logger.info("전장 전환: Field %d 활성화", self.game_manager.active_field)

                # 디버그 단축키: 숫자키로 마법 선택 (발동 X, 선택만)
                elif event.key == pygame.K_1:
                    self._select_spell("piercing_bullet")
                elif event.key == pygame.K_2:
                    self._select_spell("fireball")
                elif event.key == pygame.K_3:
                    self._select_spell("chain_lightning")

                elif event.key == pygame.K_SPACE:
                    self.is_recharging = True

            # ── 마우스 ──────────────────────────────────────────────────
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # 좌클릭 = 트리거: 선택된 마법을 마우스 조준점으로 발동
                    self._fire_selected(self._mouse_to_local_pos(event.pos))
                elif event.button == 3:
                    # 우클릭 = 선택 취소
                    self._cancel_selection()

            # ── 제스처 이벤트 ────────────────────────────────────────────
            elif event.type == GESTURE_EVENT:
                ge: GestureEvent = event.gesture_event

                if ge.gesture == _TRIGGER_GESTURE:
                    # point(트리거) 제스처 → 제스처 에임으로 선택 마법 발동
                    self._fire_selected((ge.aim_x, ge.aim_y))
                    logger.info("트리거 제스처 발동 (신뢰도: %.2f)", ge.confidence)
                else:
                    # 그 외 제스처 → 조합 버퍼에 추가하여 마법 선택
                    self._push_gesture(ge.gesture)
                    logger.debug("제스처 입력: %s (신뢰도: %.2f)", ge.gesture, ge.confidence)

        # 오버레이 활성 중에는 이후 게임 입력 무시
        if self.reward_overlay is not None:
            return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.is_recharging = True

    # ── 선택 · 발동 헬퍼 ─────────────────────────────────────────────────

    def _select_spell(self, spell_name: str) -> None:
        """마법을 발동 대기 상태로 설정한다. 버퍼는 초기화된다."""
        self.selected_spell = spell_name
        self._gesture_buffer.clear()
        self._buffer_timer = 0.0
        logger.info("마법 선택: %s", spell_name)

    def _cancel_selection(self) -> None:
        """현재 마법 선택 및 버퍼를 초기화한다."""
        if self.selected_spell is not None or self._gesture_buffer:
            logger.debug("마법 선택 취소")
        self.selected_spell = None
        self._gesture_buffer.clear()
        self._buffer_timer = 0.0

    def _push_gesture(self, gesture: str) -> None:
        """비트리거 제스처를 버퍼에 추가하고 마법 조합을 탐색한다.

        정확한 조합이 매칭되면 selected_spell 을 설정하고 버퍼를 비운다.
        더 긴 조합의 접두사이면 계속 대기한다.
        어떤 조합과도 맞지 않으면 버퍼를 초기화한다.
        """
        self._gesture_buffer.append(gesture)
        self._buffer_timer = 0.0
        key = tuple(self._gesture_buffer)

        # 정확한 조합 매칭
        if key in _GESTURE_COMBO:
            spell = _GESTURE_COMBO[key]
            self.selected_spell = spell
            self._gesture_buffer.clear()
            logger.info("조합 완성 → 마법 선택: %s  (%s)", spell, " → ".join(key))
            return

        # 접두사 매칭: 더 긴 조합의 앞부분이면 대기 유지
        if any(c[:len(key)] == key for c in _GESTURE_COMBO if len(c) > len(key)):
            logger.debug("조합 대기 중: %s", " → ".join(key))
            return

        # 버퍼가 최대 길이 초과 또는 어떤 조합과도 불일치 → 초기화
        self._gesture_buffer.clear()
        logger.debug("조합 불일치 → 버퍼 초기화")

    def _fire_selected(self, target_pos: tuple[float, float] | None = None) -> None:
        """발동 대기 중인 마법을 target_pos 방향으로 발사한다.

        성공 시 selected_spell 을 초기화한다.
        실패(쿨타임·마나 부족)해도 선택은 유지되어 재시도 가능하다.
        """
        if self.selected_spell is None:
            return
        pos = target_pos if target_pos is not None else self._mouse_to_local_pos()
        success = self.battle_scene.handle_spell_input(self.selected_spell, pos)
        if success:
            logger.info("마법 발동: %s → %.2f, %.2f", self.selected_spell, *pos)
            self.selected_spell = None
            self._gesture_buffer.clear()

    # ── 좌표 변환 ─────────────────────────────────────────────────────────

    def _mouse_to_local_pos(
        self, screen_pos: tuple[int, int] | None = None
    ) -> tuple[float, float]:
        """스크린 절대 좌표를 활성 전장 서브-Surface 기준 정규화 비율로 변환한다."""
        if screen_pos is None:
            screen_pos = pygame.mouse.get_pos()
        mx, my = screen_pos
        view_width = self.battle_scene.view_width
        view_height = self.battle_scene.view_height
        rx = max(0.0, min(1.0, mx / view_width))
        ry = max(0.0, min(1.0, my / view_height))
        return (rx, ry)

    # ── 업데이트 ──────────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        """게임 상태 업데이트."""
        # 보상 오버레이 처리
        if self.reward_overlay is not None:
            if not self.reward_overlay.active:
                self.reward_overlay = None
            else:
                return  # 오버레이 활성 중 → 게임 일시정지

        # 제스처 버퍼 타임아웃
        if self._gesture_buffer:
            self._buffer_timer += dt
            if self._buffer_timer >= _GESTURE_BUFFER_TIMEOUT:
                logger.debug("제스처 버퍼 타임아웃 → 초기화")
                self._gesture_buffer.clear()
                self._buffer_timer = 0.0

        self.game_manager.update(dt, is_recharging=self.is_recharging)
        self.battle_scene.update(dt)

        # 웨이브 클리어 후 보상 오버레이 생성
        if self.game_manager.reward_pending and self.reward_overlay is None:
            self.reward_overlay = RewardOverlay(self.game_manager)

    # ── 렌더링 ────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        """화면 렌더링."""
        self.battle_scene.draw(self.screen, self.selected_spell)
        self.hud.draw(self.screen, self.game_manager, self.is_recharging, self.selected_spell)

        if self.reward_overlay is not None:
            self.reward_overlay.draw(self.screen)

        pygame.display.flip()


def main() -> None:
    """엔트리포인트."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = App()
    app.run()


if __name__ == "__main__":
    main()
