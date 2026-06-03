# MotionMagic

**MediaPipe 손동작 인식과 CNN 기반 제스처 분류를 활용한 실시간 마법 전투 디펜스 게임**

웹캠으로 손 제스처를 인식하여 마법을 시전하는 2D 디펜스 게임입니다.

## 기술 스택

- **AI**: MediaPipe Hand Landmarker · PyTorch Conv1d CNN
- **Game**: Pygame-CE
- **Language**: Python 3.11+

## 의존성 기준

2026-05-13 기준 PyPI 최신 릴리스를 확인한 뒤, 프로젝트의 최소 권장 버전을
다음과 같이 갱신 대상으로 둡니다. `numpy`는 MediaPipe/OpenCV/PyTorch 호환성
검증 전까지 1.26 계열을 보수적 하한으로 유지합니다.

| 영역 | 패키지 | 최소 권장 버전 |
|------|--------|----------------|
| AI | `mediapipe` | `0.10.35` |
| AI | `torch` | `2.11.0` |
| AI | `numpy` | `1.26.4` |
| AI | `scikit-learn` | `1.8.0` |
| AI | `matplotlib` | `3.10.9` |
| AI | `opencv-python` | `4.13.0.92` |
| Game | `pygame-ce` | `2.5.7` |
| Dev Tools | `black` | `26.3.1` |
| Dev Tools | `ruff` | `0.15.12` |
| Dev Tools | `pytest` | `9.0.3` |

## 빠른 시작

Windows에서는 아래 배치 파일을 권장합니다. 전역 `py`가 아니라 프로젝트의
`.venv` Python을 강제로 사용하므로 `pygame`/`mediapipe` 설치 위치가 꼬이지
않습니다.

```bat
setup_env.bat
run_game.bat
```

웹캠이 열리지 않으면 카메라 번호를 바꿔 실행합니다.

```bat
run_game.bat 1
run_game.bat 1 debug
run_07_gesture_test.bat 1
```

수동 실행이 필요하면 반드시 `.venv` Python을 사용합니다.

```bash
.venv\Scripts\python -m src.game.app
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
