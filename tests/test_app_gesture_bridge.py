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
        self.input_events: list[pygame.event.Event] = []

    def handle_event(self, event: pygame.event.Event) -> None:
        """Record forwarded pygame events."""
        self.input_events.append(event)

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


def test_app_toggles_camera_overlay_with_i(monkeypatch: MonkeyPatch) -> None:
    """The debug camera overlay should be toggled with I."""
    app = object.__new__(App)
    app.running = True
    app.debug_camera_overlay = False
    app.scene = _SceneStub()
    key_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_i})
    monkeypatch.setattr(pygame.event, "get", lambda: [key_event])

    app._handle_events()

    assert app.debug_camera_overlay is True
    assert app.scene.input_events == []


def test_app_forwards_f3_after_camera_overlay_key_changed(
    monkeypatch: MonkeyPatch,
) -> None:
    """F3 should no longer toggle the debug camera overlay."""
    app = object.__new__(App)
    app.running = True
    app.debug_camera_overlay = False
    app.scene = _SceneStub()
    key_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F3})
    monkeypatch.setattr(pygame.event, "get", lambda: [key_event])

    app._handle_events()

    assert app.debug_camera_overlay is False
    assert app.scene.input_events == [key_event]
