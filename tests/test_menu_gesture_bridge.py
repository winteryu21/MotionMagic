"""Menu scene gesture bridge tests."""

from __future__ import annotations

import pygame

from src.bridge.gesture_event import GestureEvent
from src.game.scenes.explain import ExplainScene
from src.game.scenes.result import ResultScene
from src.game.scenes.title import TitleScene
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH


def _fire_event_at(pos: tuple[int, int]) -> GestureEvent:
    """Create a normalized fire event for a screen position."""
    return GestureEvent(
        gesture="fire",
        confidence=0.9,
        aim_x=pos[0] / SCREEN_WIDTH,
        aim_y=pos[1] / SCREEN_HEIGHT,
        kind="fire",
        channel="right",
    )


def test_title_scene_accepts_gesture_fire_on_start_button() -> None:
    """Title screen should allow right-hand fire to press Start."""
    scene = object.__new__(TitleScene)
    scene.next_scene = None
    scene.start_button = pygame.Rect(100, 100, 200, 80)

    scene.handle_gesture_event(_fire_event_at(scene.start_button.center))

    assert scene.next_scene == "explain"


def test_explain_scene_accepts_gesture_fire_on_battle_button() -> None:
    """Explain screen should allow right-hand fire to start battle."""
    scene = object.__new__(ExplainScene)
    scene.next_scene = None
    scene.start_button = pygame.Rect(100, 100, 200, 80)

    scene.handle_gesture_event(_fire_event_at(scene.start_button.center))

    assert scene.next_scene == "battle"


def test_result_scene_accepts_gesture_fire_to_return_to_title() -> None:
    """Result screen should allow right-hand fire to return to title."""
    scene = object.__new__(ResultScene)
    scene.next_scene = None

    scene.handle_gesture_event(_fire_event_at((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    assert scene.next_scene == "title"
