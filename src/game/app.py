"""메인 게임 루프 및 씬 관리."""

from __future__ import annotations

import logging
import sys

import pygame

from src.game.settings import COLOR_BG, FPS, SCREEN_HEIGHT, SCREEN_WIDTH, TITLE

logger = logging.getLogger(__name__)


class App:
    """게임 애플리케이션. 초기화, 메인 루프, 씬 전환을 관리한다."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

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

    def _handle_events(self) -> None:
        """이벤트 처리."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def _update(self, dt: float) -> None:
        """게임 상태 업데이트.

        Args:
            dt: 이전 프레임으로부터 경과된 시간(초).
        """

    def _draw(self) -> None:
        """화면 렌더링."""
        self.screen.fill(COLOR_BG)
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
