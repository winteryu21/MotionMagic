"""게임 전역 상수 및 설정값."""

from __future__ import annotations

# ── 화면 설정 ──
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60
TITLE = "MotionMagic"

# ── 타일맵 ──
TILE_SIZE = 64

# ── 전장 ──
NUM_FIELDS = 2  # 2개 전장 전환

# ── 제스처 ──
GESTURE_LABELS: list[str] = ["fist", "palm", "point", "scissors"]
GESTURE_COMBO_SIZE = 2  # 2개 제스처 조합으로 마법 발동
GESTURE_CONFIDENCE_THRESHOLD = 0.85

# ── AI 모델 ──
NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_FINGER_FEATURES = 5
INPUT_DIM = NUM_LANDMARKS * NUM_COORDS + NUM_FINGER_FEATURES  # 68
NUM_GESTURE_CLASSES = len(GESTURE_LABELS)

# ── 전투 ──
MAX_ENEMIES_PER_WAVE = 20
AUTO_ATTACK_RANGE = 200
AUTO_ATTACK_COOLDOWN_MS = 500

# ── 에임 ──
AIM_SENSITIVITY = 2.5
AIM_EMA_ALPHA = 0.3  # 지수가중이동평균 스무딩 계수
AIM_BUFFER_SIZE = 5   # 반동 발사용 좌표 버퍼

# ── 색상 ──
COLOR_BG = (15, 15, 25)
COLOR_WHITE = (255, 255, 255)
COLOR_HUD_HP = (76, 209, 55)
COLOR_HUD_MP = (72, 126, 255)
