"""05 — 실시간 제스처 인식 데모.

웹캠에서 실시간으로 제스처를 인식하고 화면에 오버레이한다.
ONNX 모델과 규칙 검증기를 결합하여 최종 판정을 시각화한다.

사용법:
    python scripts/05_live_demo.py
    python scripts/05_live_demo.py --model models/gesture_cnn.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.aim_tracker import AimTracker
from src.ai.preprocessor import (
    extract_2d_landmarks,
    extract_finger_states,
    normalize_landmarks,
)
from src.ai.rule_validator import validate_gesture

# MediaPipe 설정
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles

# 제스처별 표시 색상 (BGR)
GESTURE_COLORS: dict[str, tuple[int, int, int]] = {
    "rock": (0, 0, 255),      # 빨강
    "paper": (0, 255, 0),     # 초록
    "scissors": (255, 0, 0),  # 파랑
    "trigger": (0, 255, 255), # 노랑
    "idle": (128, 128, 128),  # 회색
}

# 제스처별 한국어 표시
GESTURE_KOR: dict[str, str] = {
    "rock": "바위 ✊",
    "paper": "보 ✋",
    "scissors": "가위 ✌",
    "trigger": "트리거 🔫",
    "idle": "대기 ⏳",
}


def run_demo(model_path: Path, camera_id: int = 0) -> None:
    """실시간 데모 실행.

    Args:
        model_path: ONNX 또는 PyTorch 모델 경로.
        camera_id: 카메라 디바이스 ID.
    """
    # 모델 로드
    is_onnx = model_path.suffix == ".onnx"

    if is_onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_names = [inp.name for inp in session.get_inputs()]
        print(f"ONNX 모델 로드: {model_path}")
    else:
        import torch

        from src.ai.model import GestureCNN
        from src.ai.preprocessor import NUM_CLASSES

        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        model = GestureCNN(
            num_classes=checkpoint.get("num_classes", NUM_CLASSES)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"PyTorch 모델 로드: {model_path}")

    # 조준선 추적기
    aim_tracker = AimTracker(game_width=640, game_height=480)

    # 카메라 열기
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"카메라(ID={camera_id})를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    from src.ai.preprocessor import GESTURE_LABELS

    fps_history: list[float] = []

    with MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        print("\n실시간 데모 시작! (종료: 'q' 또는 ESC)")

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_text = "None"
            confidence = 0.0
            gesture_color = (128, 128, 128)

            if results.multi_hand_landmarks:
                # 다중 손 감지: Y축 가장 높은 손 선택
                best_hand = None
                best_y = float("inf")

                for hand_lm in results.multi_hand_landmarks:
                    wrist_y = hand_lm.landmark[0].y
                    if wrist_y < best_y:
                        best_y = wrist_y
                        best_hand = hand_lm

                if best_hand is not None:
                    # 랜드마크 그리기
                    MP_DRAWING.draw_landmarks(
                        frame,
                        best_hand,
                        MP_HANDS.HAND_CONNECTIONS,
                        MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                        MP_DRAWING_STYLES.get_default_hand_connections_style(),
                    )

                    # 좌표 추출 및 전처리
                    raw_coords = np.array(
                        [[lm.x, lm.y] for lm in best_hand.landmark],
                        dtype=np.float32,
                    )

                    # 조준점 업데이트
                    aim_x, aim_y = aim_tracker.update(raw_coords)
                    cv2.circle(
                        frame,
                        (int(aim_x), int(aim_y)),
                        8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.drawMarker(
                        frame,
                        (int(aim_x), int(aim_y)),
                        (0, 255, 255),
                        cv2.MARKER_CROSS,
                        20,
                        2,
                    )

                    # 정규화
                    normalized = normalize_landmarks(raw_coords)
                    if normalized is not None:
                        finger_states = extract_finger_states(normalized)

                        # 추론
                        if is_onnx:
                            outputs = session.run(
                                None,
                                {
                                    input_names[0]: normalized[
                                        np.newaxis, :, :
                                    ],
                                    input_names[1]: finger_states[
                                        np.newaxis, :
                                    ],
                                },
                            )
                            logits = outputs[0][0]
                        else:
                            with torch.no_grad():
                                lm_t = torch.from_numpy(
                                    normalized
                                ).unsqueeze(0)
                                fs_t = torch.from_numpy(
                                    finger_states
                                ).unsqueeze(0)
                                logits = model(lm_t, fs_t).numpy()[0]

                        # Softmax
                        exp_l = np.exp(logits - np.max(logits))
                        probs = exp_l / exp_l.sum()
                        pred_idx = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        raw_label = GESTURE_LABELS.get(pred_idx, "idle")

                        # 규칙 검증
                        validated = validate_gesture(
                            raw_label,
                            normalized,
                            confidence,
                            min_confidence=0.6,
                        )

                        if validated:
                            gesture_text = GESTURE_KOR.get(
                                validated, validated
                            )
                            gesture_color = GESTURE_COLORS.get(
                                validated, (255, 255, 255)
                            )
                        else:
                            gesture_text = f"({raw_label}?) 검증 실패"
                            gesture_color = (0, 0, 200)

            # FPS 계산
            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 화면 표시
            # 배경 패널
            cv2.rectangle(frame, (0, 0), (400, 90), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Gesture: {gesture_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                gesture_color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1%}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.0f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("MotionMagic Live Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("데모 종료.")


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="MotionMagic 실시간 제스처 인식 데모"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 파일 (.onnx 또는 .pth). 미지정 시 자동 감지.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 디바이스 ID (기본: 0)",
    )

    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        onnx_path = Path("models/gesture_cnn.onnx")
        pth_path = Path("models/gesture_cnn_best.pth")
        if onnx_path.exists():
            model_path = onnx_path
        elif pth_path.exists():
            model_path = pth_path
        else:
            print("모델 파일 없음. 먼저 02_train_model.py를 실행하세요.")
            return

    run_demo(model_path, camera_id=args.camera)


if __name__ == "__main__":
    main()
