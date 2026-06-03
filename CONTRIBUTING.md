# MotionMagic — 기여 가이드

본 문서는 팀원 간 코드 품질과 일관성을 유지하기 위한 **컨벤션**을 정의합니다.

## 1. 환경 세팅

권장 경로는 프로젝트 루트에서 원클릭 배치 파일을 사용하는 것입니다.

```bat
setup_env.bat
run_game.bat
```

수동 실행이 필요하면 반드시 프로젝트 `.venv`의 Python을 사용합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\python -m src.game.app
```

## 2. 네이밍 규칙

| 대상 | 케이스 | 예시 |
|------|--------|------|
| 모듈 / 패키지 | `snake_case` | `rule_validator.py` |
| 클래스 | `PascalCase` | `GestureCNN`, `EnemySpawner` |
| 함수 / 메서드 | `snake_case` | `extract_finger_states()` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_ENEMIES_PER_WAVE` |
| private | `_접두사` | `_normalize_landmarks()` |
| 타입 별칭 | `PascalCase` | `GestureLabel`, `Landmark` |

## 3. 독스트링 & 주석

- **독스트링**: Google 스타일. 모든 public 클래스·함수에 필수.
- **텐서 차원 주석**: reshape, conv 등 텐서 연산 뒤에 `# (B, C, L)` 형태로 차원을 기록.
- **매직 넘버 금지**: 숫자 리터럴 대신 상수로 추출.

```python
# 좋은 예
x = coords.reshape(batch_size, 3, 21)  # (B, C=3, L=21)

# 나쁜 예
x = coords.reshape(batch_size, 3, 21)
```

## 4. 임포트 순서

```python
# 1. 표준 라이브러리
from __future__ import annotations
import threading

# 2. 서드파티
import numpy as np
import torch

# 3. 로컬 패키지 (절대 경로)
from src.ai.model import GestureCNN
```

## 5. 타입 힌트

- 모든 함수 시그니처에 타입 힌트 필수.
- 파일 상단에 `from __future__ import annotations` 사용.

## 6. 에러 처리

- AI 인식 실패는 **예외가 아닌 `None` 반환** + 로깅.
- `logging.getLogger(__name__)` 사용.

## 7. 의존성 방향

```
game/ → bridge/ → ai/  (단방향만 허용)
```

- `src/ai/`는 `pygame`을 import하지 않는다.
- `src/game/`은 `torch`를 import하지 않는다.

## 8. Git 브랜치 전략

| 브랜치 | 용도 | 규칙 |
|--------|------|------|
| `main` | 안정 릴리스 | 직접 커밋 금지 |
| `develop` | 통합 개발 | PR + 최소 1인 리뷰 후 머지 |
| `feature/*` | 기능 개발 | `develop`에서 분기 → PR로 머지 |
| `hotfix/*` | 긴급 수정 | `main`에서 분기 → 양쪽 머지 |

### 브랜치 네이밍

```
feature/<카테고리>-<설명>
예: feature/ai-cnn-model, feature/game-battle-scene
```

### 커밋 메시지 (Conventional Commits)

```
<type>(<scope>): <description>

type: feat, fix, refactor, docs, test, data, asset, chore
scope: ai, game, bridge, assets, docs

예: feat(ai): 하이브리드 CNN 모델 아키텍처 구현
```

## 9. PR 규칙

- **다른 파트 1명 이상**을 리뷰어로 지정 (AI↔게임 크로스 리뷰)
- 충돌 발생 시 **브랜치 소유자**가 해결
- `develop` 직접 push 금지

## 10. 코드 품질 도구

커밋 전에 아래를 실행하세요:

```bash
black .              # 포매팅
ruff check .         # 린트
pytest               # 테스트
```
