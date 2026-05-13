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
├── ai/       # AI 제스처 인식 시스템
├── game/     # Pygame-CE 게임 클라이언트
└── bridge/   # AI ↔ 게임 통합 레이어
```

## 팀 컨벤션

[CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
