"""App-level gesture bridge tests."""

from __future__ import annotations

from queue import Queue

import pygame
from pytest import MonkeyPatch

from src.bridge.gesture_event import GestureEvent
from src.game.app import App


class _SceneStub:
    """Current scene test double."""

    def __init__(self) -> None:
        self.events: list[GestureEvent] = []

    def handle_gesture_event(self, event: GestureEvent) -> None:
        """Record the dispatched gesture event."""
        self.events.append(event)


def test_app_dispatches_gesture_events_without_warping_os_mouse(
    monkeypatch: MonkeyPatch,
) -> None:
    """App should not move the OS cursor when dispatching gesture events."""
    app = object.__new__(App)
    app.gesture_events = Queue()
    app.scene = _SceneStub()
    event = GestureEvent(
        gesture="aim",
        confidence=0.9,
        aim_x=0.25,
        aim_y=0.75,
        kind="aim",
        channel="right",
    )
    app.gesture_events.put(event)

    def fail_set_pos(pos: tuple[int, int]) -> None:
        raise AssertionError(f"unexpected OS mouse warp: {pos}")

    monkeypatch.setattr(pygame.mouse, "set_pos", fail_set_pos)

    app._handle_gesture_events()

    assert app.scene.events == [event]
