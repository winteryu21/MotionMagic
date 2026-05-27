"""웨이브 기반 적 스폰 시스템.

이 파일은 스폰 테이블과 스폰 흐름만 관리한다.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.game.entities.enemy import Enemy
from src.game.game_manager import GameManager
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, TILE_SIZE

if TYPE_CHECKING:
    from src.game.scenes.battle import BattleScene

# 웨이브 진행 주기(초)
_WAVE_DURATION: float = 30.0
# 최소 스폰 간격(초)
_MIN_SPAWN_INTERVAL: float = 0.8
# 웨이브당 스폰 간격 감소량(초)
_INTERVAL_STEP: float = 0.15
# 웨이브당 적 스텟 스케일 증가율
_STAT_SCALE_PER_WAVE: float = 0.15
# 웨이브 3마다 동시 스폰 수 +1 (최대 5)
_SPAWN_COUNT_STEP: int = 3
_MAX_SPAWN_COUNT: int = 5


class Spawner:
    """웨이브 번호에 따라 적 스텟·스폰 간격·동시 출현 수를 조정하는 클래스."""

    def __init__(self, scene: BattleScene) -> None:
        self.scene = scene
        self.game_manager = GameManager()
        self._spawn_timer: float = 0.0
        self._wave_timer: float = 0.0

    # ── 현재 웨이브 기반 파생 값 ────────────────────────────────────────

    @property
    def spawn_interval(self) -> float:
        """현재 웨이브에 따른 스폰 간격(초). 웨이브가 높을수록 빠름."""
        wave = self.game_manager.current_wave
        return max(_MIN_SPAWN_INTERVAL, 2.5 - (wave - 1) * _INTERVAL_STEP)

    @property
    def spawn_count(self) -> int:
        """한 번에 전장당 스폰할 적 수."""
        wave = self.game_manager.current_wave
        return min(_MAX_SPAWN_COUNT, 1 + (wave - 1) // _SPAWN_COUNT_STEP)

    def _enemy_stats(self) -> dict:
        """현재 웨이브에 맞게 스케일된 적 스텟 딕셔너리."""
        wave = self.game_manager.current_wave
        scale = 1.0 + (wave - 1) * _STAT_SCALE_PER_WAVE
        # 웨이브 5 이상부터 방어율 소폭 부여 (최대 30%)
        defense = min(0.30, max(0.0, (wave - 4) * 0.03))
        return {
            "hp": 50.0 * scale,
            "speed": min(220.0, 120.0 + (wave - 1) * 6.0),
            "damage": 10.0 * scale,
            "defense_rate": defense,
        }

    # ── 업데이트 ────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """스폰 타이머와 웨이브 타이머를 업데이트한다."""
        # 웨이브 진행 (보상 선택 대기 중에는 타이머 멈춤)
        if not self.game_manager.reward_pending:
            self._wave_timer += dt
        if self._wave_timer >= _WAVE_DURATION:
            self._wave_timer = 0.0
            self.game_manager.current_wave += 1
            self.game_manager.reward_pending = True

        # 적 스폰
        self._spawn_timer += dt
        if self._spawn_timer >= self.spawn_interval:
            self._spawn_timer = 0.0
            self._spawn_enemies()

    def _spawn_enemies(self) -> None:
        """현재 웨이브 스텟으로 양쪽 전장에 적을 스폰한다."""
        stats = self._enemy_stats()
        count = self.spawn_count
        # 화면 상하 여백(TILE_SIZE)을 제외한 Y 범위
        y_min = float(TILE_SIZE)
        y_max = float(SCREEN_HEIGHT - TILE_SIZE * 2)

        for _ in range(count):
            # Field 0: 화면 오른쪽 끝에서 스폰, 왼쪽으로 이동
            y0 = random.uniform(y_min, y_max)
            e0 = Enemy(x=float(SCREEN_WIDTH + 32), y=y0, field_id=0, **stats)

            # Field 1: 화면 왼쪽 끝에서 스폰, 오른쪽으로 이동
            y1 = random.uniform(y_min, y_max)
            e1 = Enemy(x=-32.0, y=y1, field_id=1, **stats)

            self.scene.all_sprites.add(e0, e1)
            self.scene.enemies.add(e0, e1)
