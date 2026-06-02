"""04 — 모델 평가.

테스트 세트를 사용하여 학습된 모델의 정확도, 혼동 행렬,
클래스별 Precision/Recall/F1을 계산하고 시각화한다.

사용법:
    python scripts/04_evaluate_model.py
    python scripts/04_evaluate_model.py --model models/gesture_cnn.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.preprocessor import (
    GESTURE_LABELS,
    LABEL_TO_INDEX,
    NUM_CLASSES,
    NUM_COORDS,
    extract_finger_states,
    extract_landmarks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(path: str) -> Path:
    """상대경로를 프로젝트 루트 기준 경로로 변환한다.

    Args:
        path: CLI에서 받은 경로 문자열.

    Returns:
        절대경로 또는 프로젝트 루트 기준 경로.
    """
    result = Path(path)
    if result.is_absolute():
        return result
    return PROJECT_ROOT / result


def _checkpoint_num_coords(checkpoint: dict) -> int | None:
    """체크포인트에서 모델 입력 좌표 차원을 추정한다.

    Args:
        checkpoint: PyTorch 체크포인트 딕셔너리.

    Returns:
        입력 좌표 차원. 알 수 없으면 ``None``.
    """
    if "num_coords" in checkpoint:
        return int(checkpoint["num_coords"])

    first_weight = checkpoint["model_state_dict"].get("conv_block.0.weight")
    if first_weight is None:
        return None
    return int(first_weight.shape[1])


def load_test_data(
    test_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """테스트 세트 로드 및 전처리.

    Args:
        test_path: 테스트 JSON 파일 경로.

    Returns:
        ``(landmarks_list, finger_states_list, labels_list)`` 튜플.
    """
    with open(test_path, encoding="utf-8") as f:
        samples = json.load(f)

    landmarks_list = []
    fingers_list = []
    labels_list = []

    for sample in samples:
        if sample["label"] not in LABEL_TO_INDEX:
            continue

        lm = extract_landmarks(np.array(sample["landmarks"], dtype=np.float32))
        fs = extract_finger_states(lm)
        label = LABEL_TO_INDEX[sample["label"]]

        landmarks_list.append(lm)
        fingers_list.append(fs)
        labels_list.append(label)

    return landmarks_list, fingers_list, labels_list


def evaluate_onnx(
    model_path: Path,
    landmarks_list: list[np.ndarray],
    fingers_list: list[np.ndarray],
    labels_list: list[int],
) -> tuple[float, np.ndarray]:
    """ONNX 모델로 테스트 세트 평가.

    Args:
        model_path: ``.onnx`` 모델 경로.
        landmarks_list: 랜드마크 리스트.
        fingers_list: 손가락 상태 리스트.
        labels_list: 정답 라벨 리스트.

    Returns:
        ``(accuracy, confusion_matrix)`` 튜플.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]

    preds = []
    for lm, fs in zip(landmarks_list, fingers_list):
        outputs = session.run(
            None,
            {
                input_names[0]: lm[np.newaxis, :, :],
                input_names[1]: fs[np.newaxis, :],
            },
        )
        logits = outputs[0][0]
        preds.append(int(np.argmax(logits)))

    # 정확도
    correct = sum(pred == label for pred, label in zip(preds, labels_list))
    accuracy = correct / len(labels_list)

    # 혼동 행렬
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(preds, labels_list):
        cm[label][pred] += 1

    return accuracy, cm


def evaluate_pytorch(
    model_path: Path,
    landmarks_list: list[np.ndarray],
    fingers_list: list[np.ndarray],
    labels_list: list[int],
) -> tuple[float, np.ndarray]:
    """PyTorch 모델로 테스트 세트 평가 (ONNX 미변환 시 대안).

    Args:
        model_path: ``.pth`` 체크포인트 경로.
        landmarks_list: 랜드마크 리스트.
        fingers_list: 손가락 상태 리스트.
        labels_list: 정답 라벨 리스트.

    Returns:
        ``(accuracy, confusion_matrix)`` 튜플.
    """
    import torch

    from src.ai.model import GestureCNN

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    checkpoint_num_coords = _checkpoint_num_coords(checkpoint)
    if checkpoint_num_coords != NUM_COORDS:
        message = (
            f"체크포인트 입력 차원({checkpoint_num_coords})이 "
            f"현재 코드({NUM_COORDS})와 다릅니다. "
        )
        raise ValueError(message + "3D 전처리로 다시 학습한 뒤 평가하세요.")

    model = GestureCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for lm, fs in zip(landmarks_list, fingers_list):
            lm_tensor = torch.from_numpy(lm).unsqueeze(0)
            fs_tensor = torch.from_numpy(fs).unsqueeze(0)
            logits = model(lm_tensor, fs_tensor)
            preds.append(int(logits.argmax(dim=1).item()))

    correct = sum(pred == label for pred, label in zip(preds, labels_list))
    accuracy = correct / len(labels_list)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(preds, labels_list):
        cm[label][pred] += 1

    return accuracy, cm


def print_report(accuracy: float, cm: np.ndarray) -> None:
    """평가 결과를 포맷하여 출력.

    Args:
        accuracy: 전체 정확도.
        cm: 혼동 행렬 ``(C, C)``.
    """
    class_names = [GESTURE_LABELS[i] for i in range(NUM_CLASSES)]

    print(f"\n  전체 정확도: {accuracy:.2%}")
    print(f"\n  {'='*60}")
    print("  혼동 행렬:")
    print(f"  {'':>12}", end="")
    for name in class_names:
        print(f" {name:>10}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"  {name:>12}", end="")
        for j in range(NUM_CLASSES):
            print(f" {cm[i][j]:>10}", end="")
        print()

    # 클래스별 Precision / Recall / F1
    print(f"\n  {'='*60}")
    print(f"  {'Class':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")

    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(f"  {name:>12} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%}")


def save_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """혼동 행렬을 이미지로 저장.

    Args:
        cm: 혼동 행렬 ``(C, C)``.
        save_path: 저장 경로.
    """
    import matplotlib.pyplot as plt

    class_names = [GESTURE_LABELS[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # 셀 값 표시
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  혼동 행렬 이미지 저장: {save_path}")


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="MotionMagic 모델 평가")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 파일 (.onnx 또는 .pth). 미지정 시 자동 감지.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.json",
        help="테스트 데이터 경로 (기본: data/processed/test.json)",
    )

    args = parser.parse_args()
    test_path = _project_path(args.test_data)

    if not test_path.exists():
        print(f"테스트 데이터 없음: {test_path}")
        print("먼저 01_preprocess.py를 실행하세요.")
        return

    # 모델 파일 자동 감지
    if args.model:
        model_path = _project_path(args.model)
    else:
        onnx_path = PROJECT_ROOT / "models" / "gesture_cnn.onnx"
        pth_path = PROJECT_ROOT / "models" / "gesture_cnn_best.pth"
        if onnx_path.exists():
            model_path = onnx_path
        elif pth_path.exists():
            model_path = pth_path
        else:
            print("모델 파일 없음. 먼저 02_train_model.py를 실행하세요.")
            return

    print(f"\n{'='*60}")
    print(f" 모델 평가: {model_path}")
    print(f"{'='*60}")

    # 데이터 로드
    print(f"\n  테스트 데이터 로드: {test_path}")
    landmarks_list, fingers_list, labels_list = load_test_data(test_path)
    print(f"  테스트 샘플 수: {len(labels_list)}")

    # 평가
    if model_path.suffix == ".onnx":
        accuracy, cm = evaluate_onnx(
            model_path, landmarks_list, fingers_list, labels_list
        )
    else:
        accuracy, cm = evaluate_pytorch(
            model_path, landmarks_list, fingers_list, labels_list
        )

    # 결과 출력
    print_report(accuracy, cm)

    # 혼동 행렬 저장
    cm_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
    save_confusion_matrix(cm, cm_path)

    print(f"\n{'='*60}")
    print(f" 평가 완료! 정확도: {accuracy:.2%}")
    print(f"{'='*60}")
    print("\n다음 단계: python scripts/05_live_demo.py")


if __name__ == "__main__":
    main()
