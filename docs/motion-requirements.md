# MotionMagic AI Gesture Recognition Requirements (Finalized)

이 문서는 **MotionMagic** 프로젝트의 AI 제스처 인식 모듈의 최종 확정된 사양 및 아키텍처 요구사항 정의서입니다. 사용자 인터뷰(/grill-me)를 거쳐 모든 설계 의사결정이 완료되었으며, 이 내용을 바탕으로 AI 하위 모듈 및 브릿지 연동 작업을 시작합니다.

---

## 1. 프로젝트 아키텍처 개요
MotionMagic은 실시간 웹캠 입력을 처리하는 **AI 모듈**, 이를 게임과 연결하는 **브릿지 모듈**, 그리고 게임 비주얼과 플레이 규칙을 실행하는 **게임 모듈**로 나뉩니다.

```mermaid
graph LR
    subgraph AI Module (src/ai)
        Collector --> Preprocessor
        Preprocessor --> Dataset
        Dataset --> Model
        Model --> Trainer
        Recognizer
        RuleValidator
        AimTracker
    end

    subgraph Bridge Module (src/bridge)
        CameraThread
        GestureEvent
    end

    subgraph Game Module (src/game)
        GameLoop
    end

    CameraThread --> Recognizer
    CameraThread --> AimTracker
    Recognizer --> RuleValidator
    RuleValidator --> GestureEvent
    AimTracker --> GestureEvent
    GestureEvent --> GameLoop
```

> [!IMPORTANT]
> **의존성 단방향 제약:**
> AI 모듈(`src/ai/`) 내의 그 어떤 코드도 `src/game/`이나 `pygame` 라이브러리를 직접 가져와서(Import) 사용해서는 안 됩니다. 모든 데이터 전달은 `src/bridge/` 인터페이스를 거칩니다.

---

## 2. 합의된 핵심 설계 의사결정 사항 (Finalized Decisions)

### ① 데이터 저장 포맷 및 관리
* **포맷:** **JSON 형식**
* **범위:** **한 손(Single Hand)** 관절 정보 기반 저장. 
* **구조:** 프레임별로 손의 좌표 정보와 손가락 상태 힌트 정보를 다음과 같은 JSON 형식으로 누적 관리합니다.
  ```json
  [
    {
      "label": "scissors",
      "handedness": "Right",
      "landmarks": [
        [0.52, 0.61],
        [0.55, 0.59]
      ]
    }
  ]
  ```

### ② 관절 범위 및 좌표 차원
* **차원:** **X, Y 2D 좌표**만 사용 (`z`축 깊이 정보 제외)
* **이유:** 웹캠 환경에서 pseudo-3D인 Z축 값의 흔들림 오차를 최소화하고, 가위/바위/보/트리거 평면 실루엣 판정의 안정성을 극대화하기 위함입니다.

### ③ 전처리 및 정규화
* **위치 정규화:** 손목(Landmark 0)을 원점 `(0, 0)`으로 하는 평행 이동(Zero-centering).
* **크기 정규화:** 손목(0번)에서 중지 시작 관절(9번)까지의 거리를 기준척도로 삼아 스케일링(Scaling) 수행.
* **회전 보정전략:** 인위적인 수식 정렬을 생략하는 대신, **데이터 수집 시 다양한 각도 촬영 및 학습 시 랜덤 회전 데이터 증강(Augmentation, ±30도 내외)**을 적용하여 모델이 스스로 회전 강인성을 학습하게 유도합니다. (이를 통해 손을 가로로 눕혀 쏘는 총 모양 등 물리 방향성 훼손을 방지합니다.)

### ④ 5차원 손가락 상태(Fingers State) 기하 규칙
* **판정 규칙:** **손목 대비 관절 거리 비율 방식 (회전 강인성 확보)**
* **수식:** 각 손가락에 대해 `(손목 0번 ~ 손가락 끝 Landmark 거리) / (손목 0번 ~ 손가락 뿌리 MCP Landmark 거리)` 비율을 구합니다.
  * 이 비율이 임계값(Threshold, 기본값 `1.3` 이상)을 초과하면 해당 손가락은 완전히 **펴진 상태(1)**, 미만이면 **접힌 상태(0)**로 판정하여 `[1, 0, 0, 0, 0]` 형태의 5차원 피처 힌트를 도출합니다.
  * 예: 엄지(Landmark 4 vs 2), 검지(Landmark 8 vs 5), 중지(Landmark 12 vs 9), 약지(Landmark 16 vs 13), 새끼(Landmark 20 vs 17)

### ⑤ 모델 아키텍처 (대안 A)
* **구조:** **1D CNN + 손가락 상태 5차원 결합 모델**
* **입력:** 2D 랜드마크 좌표 피처 `(B, 21, 2)` ➡️ 1D CNN 가중치 추출 ➡️ 5차원 Fingers State 힌트 벡터와 병렬 결합(Concatenate) ➡️ Fully-Connected Layer ➡️ 최종 분류 소프트맥스 출력 `(B, 5)`
* **출력 클래스 (5개):**
  * `0: Rock` (바위 - Shield)
  * `1: Paper` (보 - Fire Blast)
  * `2: Scissors` (가위 - Ice Bolt)
  * `3: Trigger` (트리거 - Cast)
  * `4: Idle/Default` (기타/평소상태 - 오탐 예방용 예외 처리)

### ⑥ 모델 배포 및 서빙
* **방식:** **ONNX 포맷 변환 및 ONNX Runtime 추론**
* **파이프라인:** PyTorch로 학습 완료 시 즉시 `.onnx` 파일로 내보내는 변환 스크립트(`export_onnx.py`)를 가동합니다.
* **실시간 추론:** `recognizer.py`는 무겁고 뚱뚱한 PyTorch 라이브러리 전체를 불러오지 않고, **초경량 ONNX Runtime**만 로드하여 실시간 웹캠 비디오 프레임에 대해 15ms 이하의 초고속 저지연 추론을 수행합니다. (가상환경 패키지 및 최종 빌드 시 약 2GB 이상의 메모리/라이브러리 덩치를 획기적으로 차단합니다.)

### ⑦ 조준선 트래킹 및 스무딩
* **필터:** **원-유로 필터 (1€ Filter)**
* **목적:** 손가락 떨림을 줄이기 위해 조준점 이동 속도가 느릴 때는 필터 강도를 높이고, 빠른 화면 조준 시에는 지연(Lag)이 생기지 않도록 반응성을 유동적으로 조절합니다.
* **추적 관절:** 검지 끝(Landmark 8) 또는 손바닥 중심의 변환 좌표를 게임 화면 해상도(1920x1080)에 맵핑합니다.

---

## 3. 학습 및 성능 요구사항

* **데이터셋 규모:** 각 클래스당 **최소 1,000 프레임** 분량의 랜드마크 데이터 수집 목표.
* **데이터셋 분할:** 수집된 JSON 데이터셋을 훈련/검증/테스트 용도로 **8 : 1 : 1 비율로 무작위 분할**하여 평가 신뢰성을 확보합니다.
* **정확도 목표:** Test dataset 기준 분류 Accuracy **96% 이상** 달성.
* **실시간성 요건:** 추론 런타임 지연시간 **15ms 이하** 유지 (ONNX 가동으로 보장).
