"""메인 게임 루프 및 씬 관리."""

from __future__ import annotations

import logging
import sys
from queue import Empty, Queue

import pygame

from src.bridge.camera_thread import CameraThread
from src.bridge.gesture_event import GestureEvent
from src.game.scenes.battle import BattleScene
from src.game.scenes.explain import ExplainScene
from src.game.scenes.result import ResultScene
from src.game.scenes.title import TitleScene
from src.game.settings import (
    AIM_EMA_ALPHA,
    AIM_SENSITIVITY,
    COLOR_BG,
    FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TITLE,
)

logger = logging.getLogger(__name__)


class App:
    """게임 애플리케이션. 초기화, 메인 루프, 씬 전환을 관리한다."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene = TitleScene()
        self.gesture_events: Queue[GestureEvent] = Queue()
        self.camera_thread = CameraThread(
            self.gesture_events.put,
            ema_alpha=AIM_EMA_ALPHA,
            aim_sensitivity=AIM_SENSITIVITY,
        )
        self.camera_thread.start()
        self._sync_mouse_visibility()

        logger.info("MotionMagic 초기화 완료 (%dx%d)", SCREEN_WIDTH, SCREEN_HEIGHT)

    def run(self) -> None:
        """메인 게임 루프."""
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                self._handle_events()
                self._handle_gesture_events()
                self._update(dt)
                self._draw()
        finally:
            self.camera_thread.stop()
            pygame.quit()
            sys.exit()

    def _handle_events(self) -> None:
        """이벤트 처리."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            else:
                self.scene.handle_event(event)

    def _update(self, dt: float) -> None:
        self.scene.update(dt)
        self._change_scene_if_needed()

    def _handle_gesture_events(self) -> None:
        """백그라운드 AI 스레드에서 넘어온 제스처 이벤트를 현재 씬에 전달한다."""
        handler = getattr(self.scene, "handle_gesture_event", None)
        while True:
            try:
                event = self.gesture_events.get_nowait()
            except Empty:
                return

            if handler is not None:
                handler(event)

    def _draw(self) -> None:
        self.screen.fill(COLOR_BG)
        self.scene.draw(self.screen)
        pygame.display.flip()

    def _change_scene_if_needed(self) -> None:
        next_scene = getattr(self.scene, "next_scene", None)
        if next_scene is None:
            return

        if next_scene == "title":
            self.scene = TitleScene()
        elif next_scene == "explain":
            self.scene = ExplainScene()
        elif next_scene == "battle":
            self.scene = BattleScene()
        elif next_scene == "result":
            cleared_stage = int(getattr(self.scene, "result_cleared_stage", 0))
            self.scene = ResultScene(cleared_stage)
        else:
            raise ValueError(f"알 수 없는 씬 이름입니다: {next_scene}")
        self._sync_mouse_visibility()

    def _sync_mouse_visibility(self) -> None:
        pygame.mouse.set_visible(bool(getattr(self.scene, "mouse_visible", False)))


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
