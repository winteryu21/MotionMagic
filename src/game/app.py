"""메인 게임 루프 및 씬 관리."""

from __future__ import annotations

import logging
import os
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
    COLOR_MUTED,
    DEBUG_CAMERA_OVERLAY_DEFAULT,
    DEBUG_CAMERA_OVERLAY_WIDTH,
    DEFAULT_CAMERA_ID,
    FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TITLE,
)

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """환경변수 정수 값을 읽는다.

    Args:
        name: 환경변수 이름.
        default: 환경변수가 없거나 정수가 아닐 때 사용할 값.

    Returns:
        환경변수에서 읽은 정수 또는 기본값.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "%s=%r 값을 정수로 해석할 수 없어 기본값을 사용합니다.", name, raw_value
        )
        return default


def _env_flag(name: str, default: bool) -> bool:
    """환경변수 boolean 값을 읽는다.

    Args:
        name: 환경변수 이름.
        default: 환경변수가 없을 때 사용할 값.

    Returns:
        환경변수가 true 계열이면 ``True``, false 계열이면 ``False``.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class App:
    """게임 애플리케이션. 초기화, 메인 루프, 씬 전환을 관리한다."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene = TitleScene()
        self.debug_camera_overlay = _env_flag(
            "MOTIONMAGIC_DEBUG_CAMERA",
            default=DEBUG_CAMERA_OVERLAY_DEFAULT,
        )
        self.gesture_events: Queue[GestureEvent] = Queue()
        self.camera_thread = CameraThread(
            self.gesture_events.put,
            camera_id=_env_int("MOTIONMAGIC_CAMERA_ID", DEFAULT_CAMERA_ID),
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.debug_camera_overlay = not self.debug_camera_overlay
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
        if self.debug_camera_overlay:
            self._draw_camera_debug_overlay()
        pygame.display.flip()

    def _draw_camera_debug_overlay(self) -> None:
        """좌측 하단에 최신 카메라 디버그 프레임을 표시한다."""
        frame = self.camera_thread.get_debug_frame()
        if frame is None:
            return

        camera_surface = pygame.image.frombuffer(
            frame.rgb_bytes,
            (frame.width, frame.height),
            "RGB",
        )
        overlay_width = min(DEBUG_CAMERA_OVERLAY_WIDTH, SCREEN_WIDTH)
        overlay_height = max(1, round(overlay_width * frame.height / frame.width))
        overlay = pygame.transform.smoothscale(
            camera_surface,
            (overlay_width, overlay_height),
        )
        rect = overlay.get_rect(left=12, bottom=SCREEN_HEIGHT - 12)
        border = rect.inflate(4, 4)
        pygame.draw.rect(self.screen, COLOR_BG, border)
        pygame.draw.rect(self.screen, COLOR_MUTED, border, 1)
        self.screen.blit(overlay, rect)

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
