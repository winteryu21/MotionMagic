"""웹캠 캡처 + 제스처 추론 백그라운드 스레드."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from src.bridge.gesture_event import GestureEvent

logger = logging.getLogger(__name__)


class CameraThread:
    """웹캠 캡처와 CNN 추론을 별도 스레드에서 실행.

    게임 루프의 FPS 저하를 방지하기 위해 인식 파이프라인을
    백그라운드에서 동작시키고, 결과를 콜백으로 전달한다.

    Args:
        on_gesture: 제스처 인식 시 호출될 콜백 함수.
    """

    def __init__(self, on_gesture: Callable[[GestureEvent], None]) -> None:
        self._on_gesture = on_gesture
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """백그라운드 스레드 시작."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("카메라 스레드 시작")

    def stop(self) -> None:
        """백그라운드 스레드 중지."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("카메라 스레드 중지")

    def _run(self) -> None:
        """스레드 메인 루프. MediaPipe + CNN 추론을 수행."""
        # TODO: 웹캠 캡처 → MediaPipe → CNN 추론 → self._on_gesture 호출
        pass
