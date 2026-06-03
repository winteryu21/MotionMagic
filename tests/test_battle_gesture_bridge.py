"""BattleScene gesture bridge tests."""

from __future__ import annotations

from types import SimpleNamespace

from src.bridge.gesture_event import GestureEvent
from src.game.scenes.battle import BattleScene
from src.game.settings import (
    GESTURE_COMBO_SIZE,
    GESTURE_ROCK,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


class _MagicStub:
    """Minimal MagicSystem test double."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def cast_by_combo(
        self,
        combo: list[str],
        player: object,
        field: object,
        aim_pos: tuple[int, int],
        *,
        fields: list[object],
        origin_pos: tuple[int, int],
    ) -> str:
        """Record cast arguments and return a user-facing message."""
        self.calls.append(
            {
                "combo": list(combo),
                "player": player,
                "field": field,
                "aim_pos": aim_pos,
                "fields": fields,
                "origin_pos": origin_pos,
            }
        )
        return "시전됨"


def _scene_stub() -> BattleScene:
    """Create a BattleScene instance without loading pygame assets."""
    scene = object.__new__(BattleScene)
    scene.unlock_scene = SimpleNamespace(pending=False)
    scene.reward_pending = False
    scene.game_over = False
    scene.current_combo = []
    scene.aim_pos = (0, 0)
    scene.message = ""
    scene.message_timer = 0.0
    scene.player = object()
    scene.fields = [SimpleNamespace(player_pos=(11, 22))]
    scene.active_field_index = 0
    scene.magic = _MagicStub()
    scene.player_cast_frames = []
    scene.player_cast_timer = 0.0
    scene.player_cast_total_time = 0.0
    return scene


def test_battle_scene_maps_aim_event_to_screen_coordinates() -> None:
    """Aim events should move the crosshair to normalized screen coordinates."""
    scene = _scene_stub()

    scene.handle_gesture_event(
        GestureEvent(
            gesture="aim",
            confidence=0.9,
            aim_x=0.25,
            aim_y=0.75,
            kind="aim",
            channel="right",
        )
    )

    assert scene.aim_pos == (round(SCREEN_WIDTH * 0.25), round(SCREEN_HEIGHT * 0.75))


def test_battle_scene_pushes_stack_gesture_from_bridge_event() -> None:
    """Stack events should append only battle combo gestures."""
    scene = _scene_stub()

    for _ in range(GESTURE_COMBO_SIZE + 1):
        scene.handle_gesture_event(
            GestureEvent(
                gesture=GESTURE_ROCK,
                confidence=0.9,
                aim_x=0.5,
                aim_y=0.5,
                kind="stack",
                channel="left",
            )
        )

    assert scene.current_combo == [GESTURE_ROCK] * GESTURE_COMBO_SIZE
    assert "바위" in scene.message


def test_battle_scene_casts_current_combo_from_fire_event() -> None:
    """Fire events should cast at the event aim position and clear the combo."""
    scene = _scene_stub()
    scene.current_combo = [GESTURE_ROCK]

    scene.handle_gesture_event(
        GestureEvent(
            gesture="fire",
            confidence=0.9,
            aim_x=0.2,
            aim_y=0.3,
            kind="fire",
            channel="right",
        )
    )

    magic = scene.magic
    assert isinstance(magic, _MagicStub)
    assert magic.calls[0]["combo"] == [GESTURE_ROCK]
    assert magic.calls[0]["aim_pos"] == (
        round(SCREEN_WIDTH * 0.2),
        round(SCREEN_HEIGHT * 0.3),
    )
    assert scene.aim_pos == magic.calls[0]["aim_pos"]
    assert scene.current_combo == []


def test_battle_scene_reports_special_gesture_without_casting() -> None:
    """Special events should not consume or cast the current combo yet."""
    scene = _scene_stub()
    scene.current_combo = [GESTURE_ROCK]

    scene.handle_gesture_event(
        GestureEvent(
            gesture="clasp",
            confidence=0.9,
            aim_x=0.5,
            aim_y=0.5,
            kind="special",
            channel="both",
        )
    )

    magic = scene.magic
    assert isinstance(magic, _MagicStub)
    assert magic.calls == []
    assert scene.current_combo == [GESTURE_ROCK]
    assert "다이아몬드" in scene.message
