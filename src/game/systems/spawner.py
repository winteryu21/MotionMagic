"""웨이브 기반 적 스폰 시스템.

이 파일은 스폰 테이블과 스폰 흐름만 관리한다.
적 종류별 실제 스탯은 src.game.entities.enemy.ENEMY_STATS에 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from src.game.entities.enemy import ENEMY_STATS, Enemy
from src.game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, WAVE_CLEAR_DELAY


@dataclass(frozen=True, slots=True)
class SpawnEntry:
    enemy_key: str
    weight: int


@dataclass(frozen=True, slots=True)
class SpawnPlan:
    total_count: int
    spawn_interval: float
    entries: tuple[SpawnEntry, ...]

    def choose_enemy_key(self) -> str:
        keys = [entry.enemy_key for entry in self.entries]
        weights = [entry.weight for entry in self.entries]
        selected_key = random.choices(keys, weights=weights, k=1)[0]
        if selected_key not in ENEMY_STATS:
            raise KeyError(f"등록되지 않은 적 종류입니다: {selected_key}")
        return selected_key


SPAWN_TABLES: tuple[SpawnPlan, ...] = (
    # Stage 1: 일반 적만 등장
    SpawnPlan(
        total_count=11,
        spawn_interval=0.95,
        entries=(SpawnEntry("normal", 100),),
    ),
    # Stage 2: 빠른 적 소량 추가
    SpawnPlan(
        total_count=14,
        spawn_interval=0.82,
        entries=(SpawnEntry("normal", 85), SpawnEntry("fast", 15)),
    ),
    # Stage 3: 느린 탱커 적 추가
    SpawnPlan(
        total_count=17,
        spawn_interval=0.70,
        entries=(SpawnEntry("normal", 70), SpawnEntry("fast", 20), SpawnEntry("tank", 10)),
    ),
    # Stage 4: 특수 적 비율 증가
    SpawnPlan(
        total_count=21,
        spawn_interval=0.58,
        entries=(SpawnEntry("normal", 55), SpawnEntry("fast", 30), SpawnEntry("tank", 15)),
    ),
    # Stage 5: 일반 적보다 특수 적 비율이 더 커짐
    SpawnPlan(
        total_count=25,
        spawn_interval=0.48,
        entries=(SpawnEntry("normal", 42), SpawnEntry("fast", 38), SpawnEntry("tank", 20)),
    ),
    # Stage 6 이후: 고난도 반복 테이블
    SpawnPlan(
        total_count=30,
        spawn_interval=0.38,
        entries=(SpawnEntry("normal", 30), SpawnEntry("fast", 45), SpawnEntry("tank", 25)),
    ),
)


@dataclass(slots=True)
class WaveSpawner:
    stage: int = 1
    wave: int = 1
    spawned_count: int = 0
    defeated_count: int = 0
    spawn_timer: float = 0.0
    clear_timer: float = 0.0
    current_plan: SpawnPlan = field(default_factory=lambda: SPAWN_TABLES[0])

    def start_stage(self, stage: int) -> None:
        self.stage = stage
        self.wave = stage
        self.spawned_count = 0
        self.defeated_count = 0
        self.spawn_timer = 0.0
        self.clear_timer = 0.0
        self.current_plan = self._make_plan(stage)

    def _make_plan(self, stage: int) -> SpawnPlan:
        table_index = min(max(stage, 1), len(SPAWN_TABLES)) - 1
        return SPAWN_TABLES[table_index]

    def record_defeated(self, count: int = 1) -> None:
        self.defeated_count = min(self.current_plan.total_count, self.defeated_count + count)

    def update(self, dt: float, fields: list, active_field_index: int) -> bool:
        """스폰 진행. 스테이지 클리어 시 True 반환."""
        self.spawn_timer -= dt
        if self.spawned_count < self.current_plan.total_count and self.spawn_timer <= 0:
            self.spawn_timer = self.current_plan.spawn_interval
            field_index = self.spawned_count % len(fields)
            y_margin = 90
            enemy_key = self.current_plan.choose_enemy_key()
            advance_direction = -1 if field_index == 0 else 1
            spawn_x = SCREEN_WIDTH + 35 if advance_direction < 0 else -35
            enemy = Enemy.from_type(
                enemy_type=enemy_key,
                x=spawn_x,
                y=random.randint(y_margin, SCREEN_HEIGHT - y_margin),
                field_index=field_index,
                advance_direction=advance_direction,
            )
            fields[field_index].enemies.append(enemy)
            self.spawned_count += 1

        all_spawned = self.spawned_count >= self.current_plan.total_count
        all_dead = all(not field.enemies for field in fields)
        if all_spawned and all_dead:
            self.clear_timer += dt
            if self.clear_timer >= WAVE_CLEAR_DELAY:
                return True
        else:
            self.clear_timer = 0.0
        return False

    @property
    def remaining_to_spawn(self) -> int:
        return max(0, self.current_plan.total_count - self.spawned_count)
