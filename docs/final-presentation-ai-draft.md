# MotionMagic 최종 발표 자료 초안

> HTML 발표자료 제작용 Markdown 초안입니다.  
> 인공지능 과목 발표에 맞춰 AI 파이프라인, 시행착오, 최종 인식 아키텍처,
> 문제 해결 과정을 중심으로 구성했습니다.

---

## 1. 표지

### MotionMagic

**MediaPipe 손동작 인식 기반 실시간 마법 전투 디펜스 게임**

- 과목: 인공지능
- 주제: 웹캠 기반 손 제스처 인식과 게임 입력 시스템
- 핵심 키워드: `MediaPipe`, `Landmark`, `CNN`, `Feature Engineering`, `Rule-based Validation`, `State Machine`, `Real-time HCI`

발표 메모:
- 중간 발표 때는 "CNN 기반 손동작 분류 게임"에 가까운 계획이었다.
- 최종 발표에서는 실제 구현 과정에서 모델 중심 설계가 어떤 한계를 만났고, 이를 어떻게 실시간 인식 시스템으로 재설계했는지를 강조한다.

시각자료 제안:
- 게임 화면 또는 웹캠 디버그 오버레이가 보이는 대표 스크린샷
- 왼손 스택, 오른손 조준, 양손 특수 제스처 3개 장면을 한 화면에 배치

---

## 2. 프로젝트 소개: 최종 결과물

### 웹캠만으로 조작하는 2D 마법 전투 게임

최종 결과물은 웹캠으로 손을 인식하여 다음 입력을 게임에 전달하는 시스템이다.

| 입력 채널 | 최종 역할 | 게임 동작 |
|---|---|---|
| 왼손 | 마법 스택 입력 | `rock`, `paper`, `scissors` 콤보 누적 |
| 오른손 | 조준 및 발사 | 검지 기반 조준, 엄지-검지 pinch로 발사 |
| 양손 | 특수 조작 | `diamond/clasp` 마나 충전, `sonaldo` 전장 전환 |

중간 발표 이후 주요 변화:
- 단순 제스처 분류 모델에서 **실시간 제스처 모드 시스템**으로 확장
- `trigger` 제스처의 불안정성을 분석하고, 오른손 `aim + pinch fire` 방식으로 변경
- UI 조작도 마우스 커서가 아니라 게임 내부 조준점으로 통일
- 원클릭 실행 배치와 디버그 오버레이를 추가하여 시연 안정성 개선

발표 메모:
- "처음 계획과 다르게 바뀐 것"을 실패가 아니라 AI 시스템을 실제 환경에 맞춘 개선 과정으로 설명한다.

시각자료 제안:
- 최종 조작 체계 다이어그램
- 게임 HUD에 콤보 스택과 빨간 조준점이 표시된 화면

---

## 3. 중간 발표 계획과 최종 구현의 차이

### 계획: CNN이 제스처를 분류하고 게임이 이를 사용

중간 발표 기반 문서인 `docs/proposal.md`의 핵심 계획:
- MediaPipe Hand Landmarker로 21개 손 관절 좌표 추출
- 21개 관절 x 3좌표 = 63차원 좌표 데이터 구성
- 손가락 상태 5차원 피처를 추가한 68차원 입력
- PyTorch Conv1d CNN으로 `rock`, `paper`, `scissors`, `trigger` 분류
- 모델 결과를 규칙 기반 검증 레이어로 보정

### 최종: 모델 + 규칙 + 상태머신의 하이브리드 실시간 시스템

최종 구현에서 얻은 결론:
- 단일 프레임 분류만으로는 실시간 게임 입력에 필요한 안정성이 부족했다.
- 특히 `trigger`, `idle`, 손바닥/손등 방향 변화, 양손 제스처에서 문제가 컸다.
- 최종 런타임은 **손 채널 분리 + 기하학 규칙 + debouncing + cooldown + aim history** 중심으로 재설계했다.

발표 메모:
- "CNN 모델을 만들었지만 최종 게임 입력은 모두 CNN에만 맡기지 않았다"는 점이 중요하다.
- AI 과목 발표에서는 이 선택을 "모델을 버린 것"이 아니라 "데이터/센서 한계를 분석하고 보완한 AI 시스템 설계"로 설명한다.

시각자료 제안:
- 왼쪽: 중간 계획 아키텍처
- 오른쪽: 최종 아키텍처

---

## 4. 전체 시스템 아키텍처

### AI 인식부터 게임 이벤트까지

```text
Webcam
  -> MediaPipe Hand Landmarker
  -> HandObservation(label, landmarks, score)
  -> GestureModePipeline
       - left stack debouncer
       - right aim tracker
       - pinch fire detector
       - two-hand special detector
  -> GestureEvent(kind, gesture, aim_x, aim_y, channel)
  -> Bridge Queue
  -> Pygame Battle/UI Scene
```

핵심 모듈:
- `src/bridge/camera_thread.py`: 웹캠 캡처와 MediaPipe 추론을 백그라운드 스레드에서 실행
- `src/bridge/gesture_mode_pipeline.py`: 손 관측값을 게임 이벤트로 변환
- `src/bridge/gesture_event.py`: AI와 게임 사이의 데이터 계약
- `src/game/scenes/battle.py`: 제스처 이벤트를 게임 조작으로 변환

발표 메모:
- 게임 루프가 웹캠 추론 때문에 멈추지 않도록 카메라 인식은 별도 스레드에서 수행한다.
- 게임은 `torch`나 `mediapipe`를 직접 import하지 않고 `GestureEvent`만 받는다.

시각자료 제안:
- 파이프라인 블록 다이어그램
- `GestureEvent` 필드 표: `kind`, `gesture`, `aim_x`, `aim_y`, `channel`

---

## 5. 데이터 표현: 이미지가 아니라 랜드마크

### 왜 이미지 CNN이 아니라 landmark 기반인가?

MediaPipe가 제공하는 손 관절 좌표를 직접 사용했다.

입력 데이터:
- 관절 수: 21개
- 좌표: `x`, `y`, `z`
- 기본 좌표 벡터: `(21, 3)`
- 손가락 상태 피처: 5개
- 모델 실험 입력: landmark 63차원 + finger state 5차원

장점:
- 이미지보다 데이터가 작고 학습이 빠르다.
- 배경, 피부색, 조명 변화의 영향을 일부 줄일 수 있다.
- 손가락 구조가 이미 정제된 좌표로 표현된다.

한계:
- MediaPipe의 `z`는 단일 RGB 카메라 기반 추정값이라 깊이 방향 제스처에 약하다.
- 손바닥/손등 방향이 바뀌면 관절 상대 구조가 달라져 인식 안정성이 떨어진다.

발표 메모:
- `trigger` 실패 원인을 여기와 연결한다. 엄지/검지의 깊이 관계가 중요한 제스처는 2D 카메라/랜드마크만으로 안정적 분류가 어렵다.

시각자료 제안:
- MediaPipe 손 관절 21개 인덱스 그림
- `(21, 3) + finger_states(5)` 형태의 입력 벡터 도식

---

## 6. 학습 파이프라인

### 수집 -> 전처리 -> 학습 -> 평가

구현된 학습 흐름:

```text
00_collect_data.py
  -> raw landmark JSON 수집

01_preprocess.py
  -> landmark 정규화
  -> finger state 추출
  -> train/val/test split

02_train_model.py
  -> GestureCNN 학습
  -> gesture_cnn_best.pth 저장

03_export_onnx.py
  -> ONNX 내보내기

04_evaluate_model.py
  -> 혼동 행렬, 평가 결과 생성

08_ai_workflow.py
  -> 위 과정을 한 프로그램에서 실행
```

수집 편의 개선:
- 기본 제스처와 양손 특수 제스처 수집 지원
- 수집 중 space 일시정지
- 재개 시 3초 카운트다운
- `08_ai_workflow.py`로 여러 단계를 한 번에 진행

발표 메모:
- 단순히 모델만 만든 것이 아니라, 데이터를 반복적으로 수집하고 다시 학습할 수 있는 실험 파이프라인을 만들었다는 점을 강조한다.

시각자료 제안:
- 데이터 수집 화면 또는 `07` 진단 화면
- `models/training_history.png`, `models/confusion_matrix.png`

---

## 7. 모델 설계: GestureCNN

### 좌표 Conv1d + 손가락 상태 MLP

`src/ai/model.py`의 `GestureCNN` 구조:

```text
landmarks: (B, 21, 3)
  -> permute -> (B, 3, 21)
  -> Conv1d(3 -> 32)
  -> Conv1d(32 -> 64)
  -> Conv1d(64 -> 128)
  -> Global Average Pooling
  -> 128-dim feature

finger_states: (B, 5)
  -> Linear(5 -> 16)

concat: 128 + 16 = 144
  -> Linear(144 -> 64)
  -> Dropout
  -> Linear(64 -> num_classes)
```

설계 의도:
- Conv1d: 손 관절 순서에 따른 공간적 패턴을 학습
- finger state MLP: 어떤 손가락이 펴졌는지 명시적으로 전달
- 두 표현을 결합하여 좌표 기반 모델의 혼동을 줄임

최종 발표에서의 해석:
- CNN은 기본 제스처 분류 실험과 평가에는 유효했다.
- 그러나 실시간 게임 입력은 단일 프레임 분류보다 **시간적 안정화와 명시적 모드 구분**이 더 중요했다.

발표 메모:
- 모델 아키텍처를 보여주되, 최종 시스템에서 모델만으로 모든 입력을 처리하지 않은 이유까지 이어서 설명한다.

시각자료 제안:
- 모델 구조 그림
- Conv branch와 finger branch가 합쳐지는 다이어그램

---

## 8. 문제 1: 왼손/오른손 혼동

### 원인: 미러링된 웹캠 화면과 MediaPipe handedness

초기 문제:
- 사용자는 거울처럼 좌우 반전된 화면을 보고 손을 움직인다.
- MediaPipe handedness 라벨은 원본 카메라 기준으로 나온다.
- 그 결과 왼손/오른손이 반대로 인식되는 문제가 발생했다.

해결:
- 카메라 프레임을 좌우 반전할 때 handedness 라벨도 함께 보정
- `CameraThread`에 `mirror=True`, `swap_handedness=True` 기본값 적용
- 테스트로 좌우 라벨 보정 검증

최종 효과:
- 왼손은 마법 스택 전용
- 오른손은 조준/발사 전용
- 양손은 특수 모드 전용

발표 메모:
- 단순한 좌표 문제처럼 보이지만, HCI에서는 사용자가 보는 좌표계와 모델이 보는 좌표계를 맞추는 것이 중요했다.

시각자료 제안:
- 원본 카메라 좌표계 vs 거울 화면 좌표계 그림

---

## 9. 문제 2: `trigger` 제스처의 불안정성

### 모델 한계와 센서 한계가 동시에 나타난 사례

초기 계획:
- `trigger` 제스처를 학습시켜 오른손 발사 입력으로 사용

실제 문제:
- `trigger`는 엄지/검지의 깊이 방향 관계가 중요하다.
- MediaPipe의 `z`는 RGB 카메라 기반 추정값이라 안정적이지 않았다.
- 정면 trigger, 손등/손바닥 방향 변화에서 정확도가 크게 흔들렸다.
- 손목을 튕기는 발사 방식도 의도치 않은 발사/미발사가 반복됐다.

수정 방향:
- `trigger`를 "모양 분류"로 해결하려 하지 않음
- 오른손 모드는 **검지 기반 조준 pose + 엄지-검지 pinch motion**으로 재설계
- 발사는 정적 포즈가 아니라 **pinch 거리 변화량과 속도**로 판단

발표 메모:
- 이 슬라이드는 시행착오의 핵심이다.
- "학습 데이터를 더 넣으면 될 것"이라고 생각했지만, 실제로는 제스처 정의 자체가 센서에 맞지 않았다.

시각자료 제안:
- trigger 포즈와 pinch 포즈 비교
- 깊이 방향 움직임이 RGB 카메라에서 불안정하다는 설명 그림

---

## 10. 최종 입력 설계: 3가지 모드

### 손 역할을 분리하여 오인식 범위를 줄임

최종 구조:

| 모드 | 조건 | 출력 이벤트 |
|---|---|---|
| 왼손 스택 모드 | 왼손 1개, 손가락 상태 규칙 | `stack: rock/paper/scissors` |
| 오른손 조준 모드 | 검지 펴짐, 약지/새끼 접힘 | `aim`, `fire` |
| 양손 특수 모드 | 양손 모두 안정적으로 감지 | `special: clasp/sonaldo` |

중요한 설계:
- 모든 입력은 즉시 확정하지 않고 `stable_seconds`를 거친다.
- 짧은 추적 끊김은 `grace_seconds`로 흡수한다.
- 스택/발사/특수 이벤트는 cooldown으로 중복 입력을 제한한다.
- 양손 특수 모드가 잡히면 왼손 스택/오른손 조준으로 새지 않도록 suppress한다.

발표 메모:
- 이 구조는 "모르는 것을 idle로 학습"하는 접근보다 안정적이었다.
- `None`을 명시적으로 반환하는 후보 게이트가 unknown 문제를 더 잘 처리했다.

시각자료 제안:
- 세 모드 상태 머신 다이어그램
- 이벤트 종류: `stack`, `aim`, `fire`, `special`

---

## 11. 오른손 조준과 발사 알고리즘

### 조준은 검지 끝, 발사는 pinch motion

조준:
- `is_aim_pose`: 검지가 펴지고 약지/새끼가 접힌 상태
- 조준점 anchor는 최종적으로 검지 끝 landmark 8 중심
- 화면 중심 기준 sensitivity 적용
- EMA smoothing으로 손떨림 완화

발사:
- 엄지-검지 끝 거리 `pinch_distance`를 손 크기로 정규화
- `open_threshold`, `closed_threshold`로 armed 상태 관리
- 닫히는 속도와 프레임 간 거리 변화량을 함께 사용
- 발사 순간 좌표가 아니라 **발사 직전 조준 이력**을 사용

왜 직전 좌표를 쓰는가:
- pinch를 하려면 검지 위치가 움직일 수 있다.
- 발사 순간 좌표를 쓰면 의도한 목표점에서 빗나간다.
- 그래서 `pre_fire_seconds` 전의 aim history에서 좌표를 가져온다.

발표 메모:
- "손가락을 접는 동작 자체가 조준점을 흔든다"는 실제 플레이테스트 문제를 해결한 부분이다.

시각자료 제안:
- 시간축 그래프: aim history -> pinch close -> fire event
- 조준점이 유지된 채 발사되는 데모 GIF/영상

---

## 12. 양손 특수 제스처

### clasp와 sonaldo를 기하학 규칙으로 분리

초기 문제:
- 합장 `clasp`는 z축 깊이와 손 겹침 때문에 불안정했다.
- 다이아몬드 형태가 오히려 `sonaldo`처럼 인식되는 혼동도 있었다.

최종 정의:
- `clasp`: 양손 검지끼리, 엄지끼리 가까운 다이아몬드 형태
- `sonaldo`: 왼손 검지-오른손 엄지, 왼손 엄지-오른손 검지가 가까운 cross-pair 형태

판정 방식:
- 손 크기로 정규화한 landmark pair distance 사용
- clasp는 same-pair touch 우선
- sonaldo는 cross-pair touch와 양손 aim-like pose 확인
- 안정화된 special 이벤트는 유지 중에도 계속 방출하여 마나 충전에 사용

게임 동작:
- `clasp`: 유지하는 동안 마나 충전
- `sonaldo`: 전장 전환, 같은 hold 중 반복 전환 방지

발표 메모:
- 여기서는 이미지 분류가 아니라 손 구조의 기하학을 직접 활용한 것이 포인트다.

시각자료 제안:
- clasp same-pair와 sonaldo cross-pair를 선으로 표시한 손 그림

---

## 13. UI 조작 통합

### 마우스 커서가 아니라 게임 내부 조준점으로 조작

문제:
- 초기 통합에서는 제스처 조준 이벤트가 OS 마우스 커서를 이동시켰다.
- 마우스 커서가 창 밖에 있거나 OS 상태에 따라 조작감이 흔들렸다.
- 게임 조준점과 실제 마우스 커서가 별개라 사용자가 혼란을 느꼈다.

최종 수정:
- OS 마우스 위치를 강제로 움직이지 않음
- 모든 씬이 내부 `aim_pos`를 갖도록 통일
- 마우스 이동과 제스처 aim이 같은 `aim_pos`를 업데이트
- 오른손 fire는 내부 조준점 위치에서 UI 버튼을 선택
- 전투, 보상 선택, 해금 UI, 타이틀/설명/결과 화면 모두 같은 방식 사용

발표 메모:
- AI 입력 시스템도 결국 사용자 경험과 결합되어야 한다.
- 인식 알고리즘이 맞아도 UI 좌표계가 분리되면 전체 시스템은 불안정하게 느껴진다.

시각자료 제안:
- "Before: OS mouse cursor" vs "After: internal aim point" 비교

---

## 14. 디버그와 시연 안정성

### 실제 발표에서 실패 가능성을 줄이기 위한 장치

추가한 디버그 기능:
- `07_gesture_mode_test.py`: 왼손/오른손/양손 모드 진단
- 게임 내 F3: 좌측 하단 웹캠 디버그 오버레이
- MediaPipe landmark와 handedness 라벨 표시
- `run_game.bat`, `run_07_gesture_test.bat`: `.venv` 고정 실행
- 카메라 번호 인자 지원: `run_game.bat 1`

해결한 환경 문제:
- `py -m src.game.app`가 전역 Python을 타면서 `pygame`이 없다는 오류 발생
- 실행 배치에서 `.venv\Scripts\python`을 강제하여 설치 위치 불일치 방지
- `hand_landmarker.task` 누락 시 자동 다운로드

발표 메모:
- AI 데모는 알고리즘만큼 환경 세팅이 중요하다.
- 시연 실패 가능성 자체를 시스템적으로 줄인 점을 협업/완성도 측면에서 말할 수 있다.

시각자료 제안:
- `run_game.bat 1 debug` 실행 화면
- F3 디버그 오버레이 스크린샷

---

## 15. 시행착오와 수정 과정 요약

| 문제 | 원인 분석 | 수정 |
|---|---|---|
| 왼손으로 학습하면 오른손 인식 불안정 | mirrored camera와 handedness mismatch | handedness swap, 손 역할 분리 |
| trigger 인식 불안정 | z축/깊이 정보 한계, 손 방향 변화 | trigger 제거, aim + pinch fire로 재설계 |
| idle/unknown 학습 애매함 | "모르는 것"의 분포가 너무 넓음 | unknown 학습 대신 candidate gate와 `None` 처리 |
| 손바닥/손등 방향 변화 | 단일 포즈 데이터로 일반화 어려움 | 손가락 상태 규칙, 모드 안정화, 명시적 조건 |
| pinch 시 조준점이 움직임 | 발사 동작이 검지 위치를 흔듦 | 발사 직전 aim history 좌표 사용 |
| clasp/sonaldo 혼동 | 두 손가락 pair 관계가 유사 | same-pair/cross-pair 거리 규칙 분리 |
| 순간 추적 끊김 | 웹캠/MediaPipe frame noise | stable delay, grace time, cooldown |
| UI 조작과 조준점 분리 | OS 마우스와 게임 aim 좌표 분리 | 내부 aim point로 전 씬 통일 |

발표 메모:
- 교수님 평가 항목의 "시행착오와 수정 과정의 구체성"에 직접 대응하는 슬라이드다.
- 각 문제를 "무엇을 관찰했고, 왜 그렇게 판단했고, 어떻게 바꿨는가" 순서로 설명한다.

시각자료 제안:
- 표를 크게 보여주고, 발표자는 2~3개 대표 사례만 깊게 설명

---

## 16. 테스트와 평가

### 정량 평가 + 실시간 사용성 검증

정량 평가 후보:
- `models/confusion_matrix.png`
- `models/training_history.png`
- `pytest` 기반 회귀 테스트

현재 자동 테스트:
- AI 전처리, 데이터셋, 모델, 규칙 판정, aim tracker 테스트
- gesture mode pipeline 테스트
- bridge/game integration 테스트
- UI gesture control 테스트

최근 검증 예:

```text
pytest
-> 90 passed
```

실시간 검증:
- `scripts/07_gesture_mode_test.py`
- `run_07_gesture_test.bat`
- `run_game.bat 0 debug`
- 다양한 카메라 번호와 조명 환경에서 플레이테스트

발표 메모:
- 정확도 숫자만 말하기보다, "실시간 게임 입력이므로 시간적 안정성, 잘못된 입력 억제, 사용감"도 평가 기준이었다고 설명한다.
- 혼동 행렬과 테스트 개수를 같이 보여주면 AI 과목 발표와 소프트웨어 완성도를 동시에 보여줄 수 있다.

시각자료 제안:
- 혼동 행렬 이미지
- 테스트 통과 캡처
- 07 진단 화면

---

## 17. 팀원 역할 분담과 협업

### AI 팀과 게임 팀을 병렬로 나누고, bridge에서 통합

역할 분담 구조:

| 영역 | 주요 담당 내용 |
|---|---|
| AI 파이프라인 | 데이터 수집, 전처리, CNN 학습, 평가, rule/mode pipeline, 07 진단 도구 |
| 게임 클라이언트 | Pygame 전투 시스템, 마법/투사체, 적 스폰, UI/HUD, 스테이지 흐름 |
| 통합/브릿지 | `GestureEvent` 계약, `CameraThread`, 게임 씬 입력 연결, 디버그 오버레이 |
| 운영/시연 | 원클릭 세팅, 배치 실행, 카메라 ID 대응, 발표용 데모 준비 |

협업 방식:
- AI와 게임을 독립적으로 개발한 뒤 `src/bridge`에서 결합
- `game -> bridge -> ai` 단방향 의존성 유지
- PR 단위로 main에 병합
- 테스트를 통해 AI 변경이 게임 입력을 깨지 않도록 관리

발표 메모:
- 교수님 평가 항목 중 "팀원 간 역할 분담과 협업"에 대응한다.
- "AI 팀 2명/게임 팀 2명"처럼 실제 이름/인원이 확정되어 있으면 표에 이름을 추가하면 좋다.

시각자료 제안:
- 팀 역할 표
- GitHub PR 흐름 또는 모듈 의존성 그림

---

## 18. 실현 가능한 범위와 최종 구현 범위

### 중간 이후 논의한 범위 안에서 안정성 중심으로 완성

최종 구현한 것:
- 웹캠 기반 손 관절 추출
- 왼손 마법 스택
- 오른손 조준과 pinch 발사
- 양손 특수 제스처
- Pygame 전투 게임과 실시간 연결
- 디버그 오버레이와 원클릭 실행
- UI 조작까지 제스처 입력으로 통일

범위를 조정한 것:
- `trigger` 제스처는 최종 발사 입력에서 제외
- `idle`을 별도 학습 클래스로 두기보다, 후보 조건 불만족 시 이벤트를 내지 않는 구조로 변경
- 모든 제스처를 CNN 하나로 처리하기보다, 실시간 조작에 맞춰 규칙/상태머신을 결합

의미:
- 중간 발표의 핵심 목표였던 "AI 손동작 인식으로 게임을 조작"은 달성
- 동시에 실제 웹캠 환경의 한계를 반영하여 실현 가능한 입력 체계로 조정

발표 메모:
- "계획 대비 축소"가 아니라 "실제 동작하는 결과물을 위해 범위를 재정의"했다는 식으로 말한다.

시각자료 제안:
- Done / Adjusted / Deferred 3열 표

---

## 19. 데모 시나리오

### 발표 현장 시연 순서

사전 세팅:

```bat
setup_env.bat
run_game.bat 0 debug
```

웹캠이 열리지 않을 경우:

```bat
run_game.bat 1 debug
run_07_gesture_test.bat 1
```

시연 흐름:
1. F3 디버그 오버레이로 손 landmark가 잡히는지 확인
2. 타이틀/설명 화면을 오른손 조준점 + pinch로 조작
3. 전투 화면에서 왼손 `rock/paper/scissors` 스택 입력
4. 오른손 검지로 조준점 이동
5. 오른손 pinch로 마법 발사
6. 양손 diamond/clasp로 마나 충전
7. sonaldo 제스처로 전장 전환
8. 보상/해금 UI도 조준점으로 선택

백업:
- 카메라/조명 문제에 대비해 같은 흐름의 녹화 영상 준비
- 07 진단 화면 캡처 또는 짧은 GIF 준비

발표 메모:
- 현장 시연 전에는 카메라 ID와 조명, 배경, 손 위치를 반드시 확인한다.
- 데모에서 모든 기능을 오래 보여주기보다, "AI 이벤트가 게임 조작으로 연결되는 순간"을 명확히 보여준다.

시각자료 제안:
- 데모 체크리스트
- 실행 명령과 fallback 명령

---

## 20. 결론

### 모델을 만드는 것에서, 동작하는 AI 시스템을 만드는 것으로

이번 프로젝트를 통해 배운 점:
- 좋은 모델 하나만으로 실시간 HCI 문제가 해결되지는 않는다.
- 데이터 수집, 전처리, 모델, 규칙 검증, 상태머신, UI 통합이 모두 함께 설계되어야 한다.
- 실제 환경에서는 센서 한계, 사용자 동작 편차, 지연, 흔들림, 설치 환경 문제가 모두 성능에 영향을 준다.
- 실패한 제스처를 억지로 학습시키는 것보다, 입력 체계를 다시 정의하는 것이 더 좋은 해결책일 수 있다.

최종 성과:
- MediaPipe landmark 기반 AI 입력 시스템 구축
- CNN 학습/평가 파이프라인 구현
- 실시간 제스처 모드 파이프라인 구현
- Pygame 게임과 AI 입력 통합
- 실제 시연 가능한 완성형 데모 확보

마무리 문장 후보:
> MotionMagic은 단순한 게임 구현이 아니라, AI 모델을 실제 사용자 입력 시스템으로 배포할 때 필요한 문제 해결 과정을 경험한 프로젝트입니다.

시각자료 제안:
- 최종 게임 화면 + 핵심 파이프라인 한 줄 요약

---

## 부록 A. 발표에 넣을 수 있는 코드 키워드

- `GestureCNN`: Conv1d + finger state hybrid classifier
- `compute_finger_states`: landmark 기반 손가락 펼침 상태 계산
- `GestureModePipeline`: 최종 실시간 인식 상태머신
- `StackGestureDebouncer`: 왼손 스택 안정화
- `AimModeTracker`: 오른손 조준 활성 안정화
- `PinchFireDetector`: pinch 거리/속도 기반 발사
- `SpecialGestureDebouncer`: 양손 특수 제스처 안정화
- `EmaAimTracker`: 조준점 감도 매핑 + EMA smoothing
- `GestureEvent`: AI와 게임 사이의 bridge contract
- `CameraThread`: MediaPipe 추론 백그라운드 스레드

---

## 부록 B. 발표자료 제작 시 이미지 후보

프로젝트 내부 자료:
- `models/confusion_matrix.png`
- `models/training_history.png`
- `scripts/07_gesture_mode_test.py` 실행 화면 캡처
- `run_game.bat 0 debug` 실행 화면 캡처
- 게임 전투 화면: 콤보 스택, 빨간 조준점, 발사 장면
- diamond/clasp, sonaldo 손 포즈 캡처

직접 제작하면 좋은 그림:
- 최종 AI 파이프라인 블록 다이어그램
- 3모드 입력 구조도
- pinch fire 시간축 다이어그램
- clasp same-pair / sonaldo cross-pair 거리 규칙 그림

