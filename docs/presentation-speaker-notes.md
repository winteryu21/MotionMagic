# MotionMagic 최종 발표 — 슬라이드별 참조 노트

> 대본이 아닌 **발표 시 참고용** 문서입니다.
> 각 슬라이드에서 반드시 언급할 포인트, 기술적 배경, 예상 질문을 정리합니다.
> 전체 발표 시간: **~10분** (15슬라이드, 슬라이드당 평균 40초)

---

## Slide 1 — 타이틀

**시간**: ~15초 (간결하게)

**포인트**
- 프로젝트 이름과 한 줄 소개만 언급
- "MediaPipe 손동작 인식으로 마법을 시전하는 2D 디펜스 게임"
- 하단 키워드 배지는 읽지 말고 시각적 보조로만 활용

---

## Slide 2 — 팀 소개 & 역할 분담

**시간**: ~30초

**포인트**
- 4명, AI 2 + 게임 2 구성
- **핵심**: `game/ → bridge/ → ai/` 단방향 의존성 구조
  - AI 모듈은 게임을 모르고, 게임 모듈은 AI를 모름
  - 독립 개발 → bridge에서 GestureEvent 하나로 통합
- 이 구조 덕분에 AI 쪽 변경(trigger 제거 등)이 게임 코드에 영향을 주지 않았음을 간단히 언급

**기술 배경**
- `src/ai/`는 `torch`, `mediapipe`만 사용 (pygame 절대 import 안 함)
- `src/game/`은 `torch`, `mediapipe` 절대 import 안 함
- `src/bridge/gesture_event.py`의 `GestureEvent` dataclass가 유일한 접점
  - 필드: gesture, confidence, aim_x, aim_y, kind, channel, active

---

## Slide 3 — 프로젝트 소개

**시간**: ~40초

**포인트**
- 표를 가리키며 왼손/오른손/양손의 역할 분리를 설명
  - 왼손: 마법 콤보 입력 (rock, paper, scissors 순서 조합)
  - 오른손: 검지로 조준, 엄지-검지 pinch로 발사
  - 양손: clasp(마나 충전), sonaldo(전장 전환)
- **중간 발표 대비 변화 3가지** 강조
  - 단순 분류 → 모드 시스템
  - trigger 제스처 → aim+pinch 재설계
  - 마우스/제스처 조준 통일

**기술 배경**
- 마법은 1~3개 제스처 순서 조합으로 결정 (1티어~3티어)
- 오른손은 CNN이 아닌 규칙 기반 (landmark 기하학)으로 동작

---

## Slide 4 — 계획에서 최종 아키텍처로

**시간**: ~50초 (핵심 슬라이드)

**포인트**
- 왼쪽(중간 발표)과 오른쪽(최종)을 대비하며 설명
- 중간 발표: "CNN 하나가 모든 것을 분류" 라는 단순한 구조
  - CSV로 데이터 저장 → PyTorch CNN → 규칙 검증 → 게임
- 최종: "모델 + 규칙 + 상태머신 하이브리드"
  - JSON 포맷으로 전환 (구조화된 데이터, 메타데이터 포함 가능)
  - HandObservation 단위로 추상화
  - 손 채널 분리, 모드별 판정, debounce/cooldown
- 하단 파이프라인 다이어그램: Webcam → MediaPipe → **GestureModePipeline** → GestureEvent → Pygame

**기술 배경**
- CSV → JSON 전환 이유: CSV는 flat한 좌표 나열, JSON은 손 별 landmark + 메타데이터를 구조적으로 표현
- `GestureModePipeline`이 핵심 오케스트레이터
  - 내부에서 `LeftHandStackMode`, `RightHandAimMode`, `BothHandSpecialMode` 분기
  - 각 모드가 독립적으로 판정 → 이벤트 생성
- `HandObservation`: landmarks(21×3), finger_states(5,), handedness, timestamp 등을 묶은 구조체

**예상 질문**
- "왜 CNN 하나로 안 하고 모드를 나눴나?" → 왼손/오른손 역할이 다르고, 오른손은 분류가 아니라 연속적 추적이 필요해서
- "JSON이 CSV보다 나은 이유?" → 좌표 외 메타데이터(handedness, timestamp)를 함께 저장, 중첩 구조 표현 가능

---

## Slide 5 — 데이터 표현: 이미지가 아니라 랜드마크

**시간**: ~50초 (핵심 슬라이드)

**포인트**
- "이미지 분류가 아니라 좌표 데이터를 쓴다"는 점을 명확히
- Dual-branch 입력 강조: (21,3) 랜드마크 + (5,) 손가락 상태를 별도 입력으로 처리
- 장점 3가지: 데이터 가벼움, 환경 불변, 구조적 표현
- **한계** 부분이 중요: MediaPipe의 z축 한계 → 다음 슬라이드(trigger 문제)의 복선

**기술 배경**
- MediaPipe Hands: 21개 관절 좌표 (x, y, z), 단 z는 depth camera 없이 RGB에서 추정
- 손가락 상태(5,): 각 손가락의 펴짐(1)/접힘(0), `extract_finger_states()` 함수로 추출
  - 엄지: 다른 4개와 다른 기준 (IP관절 vs MCP관절의 각도)
  - 나머지: TIP이 PIP보다 손바닥에서 먼지 여부
- 정규화: 손목(landmark 0) 기준 상대 좌표, 손 크기로 스케일 정규화

**예상 질문**
- "왜 이미지 CNN을 안 쓰나?" → 배경, 조명, 피부색 변수가 너무 많음. 랜드마크는 MediaPipe가 이미 추출해줌
- "z 좌표를 왜 쓰나? 안 쓰면 안 되나?" → 일부 제스처(예: trigger)에서 깊이 관계가 필요했으나, 결국 z 의존도를 줄이는 방향으로 재설계

---

## Slide 6 — 학습 파이프라인 & GestureCNN

**시간**: ~60초 (기술 핵심)

**포인트**
- 왼쪽: 4단계 파이프라인 (수집 → 전처리 → 학습 → 평가), `08_ai_workflow`로 통합 실행
- 오른쪽: GestureCNN Dual-branch 구조
  - Conv 브랜치: (B,21,3) → permute → (B,3,21) → Conv1d ×3 → GAP → (B,128)
  - Finger 브랜치: (B,5) → Linear → (B,16)
  - concat → (B,144) → FC → 4클래스 출력

**기술 배경**
- Conv1d를 쓴 이유: 21개 관절이 순서가 있음 (0=손목→4=엄지끝→8=검지끝...), 인접 관절 패턴을 캡처
- 3채널(x,y,z)을 Conv1d의 input channel로 사용
- GAP (Global Average Pooling): FC 대비 파라미터 절약 + 과적합 방지
- Finger state를 별도 브랜치로 넣은 이유: Conv가 암묵적으로 학습할 수도 있지만, 명시적 힌트가 분류 안정성을 높임
- Dropout: FC 레이어 사이에 적용, 과적합 방지
- 4클래스: rock, paper, scissors, unknown(또는 idle)

**예상 질문**
- "왜 Conv2d가 아니라 Conv1d?" → 입력이 이미지가 아니라 1D 시퀀스(관절 순서)이므로
- "finger state를 왜 따로?" → 실험 결과 분리했을 때 accuracy가 더 높았음. rock↔scissors 구분에서 특히 효과적
- "4클래스밖에 없는데 CNN이 필요한가?" → 규칙만으로도 가능하지만, 관절 좌표의 미세한 패턴(손바닥 방향 등)은 학습 기반이 더 강건

---

## Slide 7 — 문제 1: 왼손/오른손 혼동

**시간**: ~40초

**포인트**
- 문제 상황을 직관적으로 설명: "사용자는 거울을 보듯 화면을 본다"
- MediaPipe는 원본 카메라 기준으로 handedness를 매기므로, 좌우가 반전됨
- 해결: `CameraThread`에서 mirror + handedness swap을 동시 적용
- 다이어그램으로 흐름 설명: 원본(불일치) → mirror+swap(일치) → 왼손=스택, 오른손=조준

**기술 배경**
- `cv2.flip(frame, 1)` 으로 좌우 반전
- MediaPipe 결과의 `handedness[i].classification[0].label`을 "Left"↔"Right" 스왑
- 테스트: `test_camera_thread.py`에서 handedness 보정 검증

---

## Slide 8 — 문제 2: trigger 제스처 불안정성

**시간**: ~60초 (시행착오의 핵심)

**포인트**
- **AI 과목 평가 기준 "시행착오의 구체성"에 직결되는 슬라이드**
- trigger 제스처(권총 모양)가 왜 실패했는지:
  - 엄지/검지의 깊이(z축) 관계에 의존
  - MediaPipe z는 RGB 추정이라 불안정
  - 손 방향에 따라 관절 구조 자체가 변함
- **핵심 교훈**: 데이터를 더 모아서 해결하려 했으나, 문제는 "제스처 정의 자체가 센서에 맞지 않았다"
- 결정: trigger를 모양 분류로 푸는 것을 포기 → aim + pinch fire로 완전 재설계

**기술 배경**
- trigger: 검지를 펴고 엄지를 세운 권총 모양 → 손목 튕기기로 발사
- 학습 데이터 추가 수집(~500개)으로도 교차 검증 accuracy가 70% 내외로 정체
- 결국 trigger 클래스를 제거하고 4클래스(rock, paper, scissors, unknown)로 축소
- 오른손은 CNN 분류 대상에서 아예 제외 → 규칙 기반 aim tracker로 전환

**예상 질문**
- "학습 데이터를 더 모으면 해결되지 않나?" → 시도했으나, 센서(z축) 자체의 한계가 본질적 원인. 데이터양의 문제가 아님
- "다른 제스처로 대체하는 게 아니라 방식 자체를 바꾼 이유?" → 발사는 "동작"이지 "포즈"가 아님. 정적 분류보다 동적 변화량 감지가 더 적합

---

## Slide 9 — 최종 입력 설계: 3가지 모드

**시간**: ~50초

**포인트**
- 3개 모드 카드를 하나씩 설명
  - 왼손 스택: 왼손 1개 감지 → 손가락 규칙으로 rock/paper/scissors
  - 오른손 조준: 검지 펴짐 + 나머지 접힘 → aim pose → pinch fire
  - 양손 특수: 양손 안정 감지 → clasp/sonaldo 기하학 판정
- 안정화 메커니즘 4가지: stable_seconds, grace_seconds, cooldown, suppress
- UI 조준 통일: 마우스와 제스처가 동일한 aim_pos를 공유

**기술 배경**
- `stable_seconds`: 제스처가 N초 이상 유지되어야 확정 (떨림 방지)
- `grace_seconds`: 짧은 추적 끊김(0.2초 등)은 무시 (MediaPipe frame drop)
- `cooldown`: 같은 제스처 연속 발생 제한 (중복 입력 방지)
- `suppress`: 양손 모드 활성화 시 왼손/오른손 개별 이벤트를 억제
  - 예: clasp 시 왼손이 잠깐 rock으로 인식되는 것 방지
- `aim_pos`: 게임 전역 상태, 마우스 움직임과 제스처 둘 다 이 값을 업데이트

**예상 질문**
- "모드 전환은 어떻게 되나?" → 자동. 감지된 손 개수와 handedness로 파이프라인이 자동 분기
- "stable_seconds를 몇 초로 설정했나?" → 모드마다 다름. 스택 ~0.3초, 양손 ~0.5초

---

## Slide 10 — 조준/발사 알고리즘 & 양손 특수 제스처

**시간**: ~50초

**포인트**
- **Aim + Pinch Fire**: landmark 8(검지 끝) 좌표를 sensitivity 매핑 후 EMA 스무딩
  - 발사: 엄지-검지 거리를 손 크기로 정규화 → open→armed→closed 상태 전이 + 닫힘 속도 체크
  - pinch 시 조준점 흔들림 → **aim history 좌표** 사용으로 해결
- **clasp vs sonaldo**: 둘 다 양손 제스처이지만 pair 관계가 다름
  - clasp: same-pair (검지↔검지, 엄지↔엄지)
  - sonaldo: cross-pair (왼검지↔오른엄지, 왼엄지↔오른검지)

**기술 배경**
- EMA (Exponential Moving Average) 스무딩: `aim = α × new + (1-α) × prev`, α ≈ 0.3
- sensitivity 매핑: MediaPipe 좌표(0~1)를 게임 해상도(1920×1080)에 매핑, 화면 가장자리까지 도달 가능하도록 스케일링
- Pinch fire 상태머신:
  1. `IDLE` → 엄지-검지 거리 > open_threshold → `ARMED`
  2. `ARMED` → 거리 < closed_threshold + 닫힘 속도 > min_velocity → `FIRED`
  3. `FIRED` → cooldown 후 → `IDLE`
- aim history: 최근 N프레임의 aim 좌표 deque → 발사 시점에서 2~3프레임 전 좌표 사용
- clasp/sonaldo 거리 계산: `|L_index - R_index| + |L_thumb - R_thumb|` 등을 손 크기로 정규화

---

## Slide 11 — 시행착오 요약 표

**시간**: ~40초

**포인트**
- 표를 전부 읽지 말 것. 상단 3개(강조된 행)만 짚기:
  1. 왼손/오른손 혼동 → handedness swap
  2. trigger 불안정 → aim+pinch 재설계
  3. idle/unknown 학습 → candidate gate + None 반환
- 나머지 회색 행은 "이 외에도 여러 세부 조정이 있었다" 정도로 넘기기
- 질의응답 시 회색 행에 대한 구체적 질문이 올 수 있음

**기술 배경 (질의응답 대비)**
- idle/unknown 문제: "아무 제스처도 아닌 것"의 분포가 무한 → 학습이 어려움
  - 해결: CNN이 None을 반환할 수 있도록 confidence threshold + candidate gate
  - candidate gate: 손가락 상태가 어떤 제스처의 후보 패턴에도 맞지 않으면 CNN에 보내지 않음
- pinch 조준점 흔들림: aim history deque에서 N프레임 전 좌표 사용 (위 참조)
- clasp/sonaldo 혼동: 초기에는 "양손이 가까운가?"만 체크 → same/cross pair 거리로 분리
- 순간 추적 끊김: grace_seconds 유예 시간 (MediaPipe가 1~2프레임 손을 놓치는 경우)

---

## Slide 12 — 테스트와 평가

**시간**: ~40초

**포인트**
- 혼동 행렬: Test Accuracy **99%** (198/200) — 4클래스 분류
- 학습 곡선: 50 epochs, loss/accuracy 안정적 수렴
- 자동 테스트 90개: AI 전처리, 모델, 규칙, aim tracker, bridge, game UI 전 범위
- **강조**: "정확도 숫자만이 아니라, 실시간 사용감과 시간적 안정성도 평가했다"

**기술 배경**
- 테스트 데이터: 학습에 사용하지 않은 200개 샘플 (각 클래스 50개)
- 혼동 행렬 오분류 2개: rock↔scissors 1개, paper↔scissors 1개
- 학습 설정: Adam optimizer, lr=1e-3, batch=32, 50 epochs, train/val/test = 70/15/15
- pytest 구성: `tests/` 디렉토리, 단위 테스트 + 통합 테스트
- 정량 평가로 보여줄 수 없는 것: 실제 게임 플레이 시 체감 반응 속도, 오입력 빈도 → 플레이테스트로 보완

**예상 질문**
- "99%인데 실제로도 그런가?" → 정적 분류 정확도는 높지만, 실시간 환경에서는 추적 끊김, 모드 전환 지연 등이 있어 체감은 다름. 그래서 규칙 + 상태머신으로 보강
- "테스트 90개는 어떤 것들?" → 전처리 정규화 검증, 데이터셋 로딩, 모델 forward/backward, 규칙 판정, aim 좌표 계산, GestureEvent 생성, 게임 통합 등

---

## Slide 13 — 데모 영상

**시간**: ~60초 (영상 재생)

**포인트**
- 영상 재생 전 간단히: "실제 게임 플레이 영상"
- 영상 중 주요 장면에서 간단히 코멘트:
  - 왼손으로 마법 스택 쌓는 장면
  - 오른손으로 조준/발사하는 장면
  - 양손 clasp(마나 충전) 장면
  - 전장 전환 장면 (있다면)
- 영상 끝난 후 "지금까지 보여드린 기술적 내용이 실제로 이렇게 동작합니다" 정도로 마무리

---

## Slide 14 — 결론

**시간**: ~40초

**포인트**
- 왼쪽 "배운 점" 3가지를 간결하게:
  1. 좋은 모델만으로 실시간 HCI 문제가 해결되지 않음
  2. 데이터~UI까지 전체가 함께 설계되어야 함
  3. 실패한 제스처를 억지로 학습시키기보다, 입력 체계를 재정의하는 것이 나은 해결책
- 오른쪽 "최종 성과" 4가지는 빠르게
- 하단 인용문으로 마무리: "문제 해결 과정을 경험한 프로젝트"

**기술 배경 (정리)**
- 이 프로젝트의 AI 핵심: 모델 accuracy가 아니라, 센서 한계를 인식하고 시스템 설계를 바꾼 것
- 학습시킬 수 없는 것(z축 의존 제스처)을 인정하고, 학습 가능한 범위를 재정의

---

## Slide 15 — Q & A

**시간**: 나머지 전부

**포인트**
- "감사합니다. 질문 받겠습니다."

**대비할 예상 질문 TOP 5**
1. "왜 이미지 기반 CNN이 아니라 landmark 기반인가?"
   → 배경/조명/피부색 변수 제거, MediaPipe가 이미 추출해줌, 데이터가 작고 학습이 빠름
2. "trigger를 더 많은 데이터로 학습시키면 안 되나?"
   → 시도했으나 70% 벽. z축 추정의 본질적 한계. 센서가 안 되는 건 데이터로 안 됨
3. "실시간 성능은?"
   → MediaPipe + CNN 추론이 백그라운드 스레드에서 ~30fps. 게임 루프와 분리되어 게임 프레임에 영향 없음
4. "양손 제스처를 CNN으로 학습시키면 안 되나?"
   → 가능하지만, clasp/sonaldo는 두 손 간 기하학 관계가 핵심. 규칙이 더 직관적이고 설명 가능
5. "다른 제스처를 추가하려면?"
   → 왼손 스택: CNN 클래스 추가 + 학습 데이터 수집. 양손: 규칙 함수 추가. 모듈 구조라 확장 용이

---

## 전체 시간 배분 가이드

| 구간 | 슬라이드 | 시간 |
|------|---------|------|
| 도입 | 1~3 | ~1.5분 |
| 아키텍처 | 4 | ~1분 |
| AI 기술 | 5~6 | ~2분 |
| 시행착오 | 7~8 | ~2분 |
| 최종 설계 | 9~10 | ~1.5분 |
| 평가/데모 | 11~13 | ~2분 |
| 마무리 | 14~15 | ~1분 |
| **합계** | | **~10분** |
