# MotionMagic

**MediaPipe 손동작 인식과 CNN 기반 제스처 분류를 활용한 실시간 마법 전투 디펜스 게임**

웹캠으로 손 제스처를 인식하여 마법을 시전하는 2D 디펜스 게임입니다.

## 기술 스택

- **AI**: MediaPipe Hand Landmarker · PyTorch Conv1d CNN
- **Game**: Pygame-CE
- **Language**: Python 3.11+

## 빠른 시작

```bash
# 1. 가상환경 생성 & 활성화
python -m venv .venv
.venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 게임 실행
python -m src.game.app
```

## 프로젝트 구조

```
src/
├── ai/                     # AI 제스처 인식 시스템
│   ├── collector.py        #   웹캠 → MediaPipe → CSV 수집
│   ├── preprocessor.py     #   정규화, 증강, 피처 추출
│   ├── dataset.py          #   PyTorch Dataset
│   ├── model.py            #   하이브리드 CNN 모델
│   ├── trainer.py          #   학습 루프 & 평가
│   ├── recognizer.py       #   실시간 제스처 인식
│   ├── rule_validator.py   #   규칙 기반 검증 레이어
│   └── aim_tracker.py      #   에임 좌표 매핑 & EMA 스무딩
│
├── game/                   # Pygame-CE 게임 클라이언트
│   ├── app.py              #   메인 게임 루프
│   ├── settings.py         #   게임 상수 (1920×1080)
│   ├── scenes/             #   씬 (title, battle, result)
│   ├── entities/           #   엔티티 (player, enemy, projectile)
│   ├── systems/            #   시스템 (magic, combat, spawner, particles)
│   └── ui/                 #   UI (hud, crosshair)
│
└── bridge/                 # AI ↔ 게임 통합
    ├── gesture_event.py    #   제스처 이벤트 데이터클래스
    └── camera_thread.py    #   웹캠 + 추론 백그라운드 스레드

scripts/                    # CLI 도구 (collect_data, train_model, evaluate_model)
tests/                      # pytest 테스트
assets/                     # 게임 에셋 (sprites, effects, maps, ui, sounds, fonts)
data/                       # AI 학습 데이터 — .gitignore 대상
models/                     # 학습 완료 .pth 체크포인트
docs/                       # 문서 (아키텍처, 제스처 카탈로그)
```

## 팀 컨벤션

[CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
