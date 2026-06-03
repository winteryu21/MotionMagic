"""BattleScene gesture bridge tests."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from pytest import MonkeyPatch

from src.bridge.gesture_event import GestureEvent
from src.game.scenes.battle import BattleScene
from src.game.settings import (
    GESTURE_COMBO_SIZE,
    GESTURE_ROCK,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SPECIAL_GESTURE_HOLD_GRACE_SECONDS,
)
from src.game.systems.magic import MagicSystem


class _MagicStub:
    """Minimal MagicSystem test double."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response = "시전됨"

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
        return self.response


class _PlayerStub:
    """Minimal Player test double."""

    def __init__(self) -> None:
        self.charged_seconds = 0.0
        self.alive = True

    def charge_mana(self, dt: float) -> None:
        """Record mana charge duration."""
        self.charged_seconds += dt


class _FieldStub:
    """Minimal BattleField test double."""

    player_pos = (11, 22)

    def __init__(self) -> None:
        self.enemies: list[_EnemyStub] = []

    def update(self, dt: float, player: object) -> int:
        """Return zero defeated enemies."""
        return 0


class _EnemyStub:
    """Minimal enemy test double."""

    def __init__(self, x: float, y: float, alive: bool = True) -> None:
        self.x = x
        self.y = y
        self.alive = alive


class _PressedKeysStub:
    """pygame.key.get_pressed test double."""

    def __getitem__(self, key: int) -> bool:
        """Report every key as unpressed."""
        return False


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
    scene.debug_notice_text = ""
    scene.debug_notice_timer = 0.0
    scene.special_hold_gesture = None
    scene.special_hold_timer = 0.0
    scene.player = _PlayerStub()
    scene.fields = [_FieldStub(), _FieldStub()]
    scene.active_field_index = 0
    scene.spawner = SimpleNamespace(
        record_defeated=lambda defeated: None,
        update=lambda dt, fields, active_field_index: False,
    )
    scene.magic = _MagicStub()
    scene.player_cast_frames = []
    scene.player_cast_timers = [0.0, 0.0]
    scene.player_cast_frame_time = 0.06
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


def test_battle_scene_keeps_aim_event_at_raw_position_with_enemies() -> None:
    """Aim events should keep the visible crosshair at the raw aim position."""
    scene = _scene_stub()
    scene.active_field.enemies = [
        _EnemyStub(900.0, 600.0),
        _EnemyStub(310.0, 220.0),
        _EnemyStub(250.0, 180.0, alive=False),
    ]

    scene.handle_gesture_event(
        GestureEvent(
            gesture="aim",
            confidence=0.9,
            aim_x=0.25,
            aim_y=0.30,
            kind="aim",
            channel="right",
        )
    )

    assert scene.aim_pos == (
        round(SCREEN_WIDTH * 0.25),
        round(SCREEN_HEIGHT * 0.30),
    )


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


def test_battle_scene_casts_fire_event_at_nearest_enemy() -> None:
    """Fire events should assist only the spell target, not the crosshair."""
    scene = _scene_stub()
    scene.current_combo = [GESTURE_ROCK]
    scene.active_field.enemies = [
        _EnemyStub(900.0, 600.0),
        _EnemyStub(255.0, 210.0),
    ]

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
    assert magic.calls[0]["aim_pos"] == (255, 210)
    assert scene.aim_pos == (
        round(SCREEN_WIDTH * 0.2),
        round(SCREEN_HEIGHT * 0.3),
    )


def test_battle_scene_starts_cast_animation_only_on_active_field() -> None:
    """Casting should animate only the field where the spell was cast."""
    scene = _scene_stub()
    scene.current_combo = [GESTURE_ROCK]
    scene.magic.response = "마법 Lv.1"
    scene.player_cast_frames = [pygame.Surface((1, 1))]
    scene.player_cast_total_time = 0.3
    scene.active_field_index = 1

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

    assert scene.player_cast_timers == [0.0, 0.3]


def test_battle_scene_player_image_uses_each_field_cast_timer() -> None:
    """Player sprites should be selected from each field's own cast timer."""
    scene = _scene_stub()
    idle_image = pygame.Surface((1, 1))
    cast_image = pygame.Surface((1, 1))
    scene.player_idle_image = idle_image
    scene.player_cast_frames = [cast_image]
    scene.player_cast_timers = [0.0, 0.3]
    scene.player_cast_total_time = 0.3

    assert scene._current_player_image(0) is idle_image
    assert scene._current_player_image(1) is cast_image


def test_battle_scene_keeps_crosshair_raw_during_update(
    monkeypatch: MonkeyPatch,
) -> None:
    """Battle updates should not attach the crosshair to a moving enemy."""
    scene = _scene_stub()
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeysStub())
    enemy = _EnemyStub(410.0, 250.0)
    scene.active_field.enemies = [enemy]
    scene.aim_pos = (400, 240)

    scene.update(0.1)
    assert scene.aim_pos == (400, 240)

    enemy.x = 430.0
    enemy.y = 260.0
    scene.update(0.1)

    assert scene.aim_pos == (400, 240)


def test_battle_scene_debug_u_unlocks_all_spells() -> None:
    """U should unlock every spell and show the centered debug notice."""
    scene = _scene_stub()
    scene.magic = MagicSystem()
    locked_count = len(scene.magic.locked_spells())

    scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_u}))

    assert locked_count > 0
    assert scene.magic.locked_spells() == []
    assert scene.debug_notice_text == "디버그 모드: 모든 마법 언락"
    assert scene.debug_notice_timer > 0.0
    assert "디버그 모드: 모든 마법 언락" in scene.message


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


def test_battle_scene_charges_mana_while_clasp_is_held(
    monkeypatch: MonkeyPatch,
) -> None:
    """Clasp special hold should charge mana during battle updates."""
    scene = _scene_stub()
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _PressedKeysStub())

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
    scene.update(0.1)

    player = scene.player
    assert isinstance(player, _PlayerStub)
    assert scene.special_hold_timer == SPECIAL_GESTURE_HOLD_GRACE_SECONDS - 0.1
    assert player.charged_seconds == 0.1


def test_battle_scene_switches_field_once_per_sonaldo_hold() -> None:
    """Sonaldo should switch fields once until the current hold expires."""
    scene = _scene_stub()
    event = GestureEvent(
        gesture="sonaldo",
        confidence=0.9,
        aim_x=0.5,
        aim_y=0.5,
        kind="special",
        channel="both",
    )

    scene.handle_gesture_event(event)
    scene.handle_gesture_event(event)

    assert scene.active_field_index == 1

    scene._update_special_hold(SPECIAL_GESTURE_HOLD_GRACE_SECONDS)
    scene.handle_gesture_event(event)

    assert scene.active_field_index == 0
