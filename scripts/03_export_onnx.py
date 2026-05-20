"""03 — PyTorch → ONNX 모델 변환.

학습된 ``.pth`` 체크포인트를 ONNX 포맷으로 내보내어
게임 런타임에서 PyTorch 없이 ONNX Runtime으로 추론할 수 있게 한다.

사용법:
    python scripts/03_export_onnx.py
    python scripts/03_export_onnx.py --checkpoint models/gesture_cnn_best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.ai.model import GestureCNN
from src.ai.preprocessor import NUM_CLASSES, NUM_COORDS, NUM_LANDMARKS


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """PyTorch 체크포인트를 ONNX로 변환.

    Args:
        checkpoint_path: ``.pth`` 체크포인트 파일 경로.
        output_path: 출력 ``.onnx`` 파일 경로.
        opset_version: ONNX opset 버전.
    """
    # 체크포인트 로드
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    model = GestureCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  모델 로드 완료: {checkpoint_path}")
    print(f"  에포크: {checkpoint.get('epoch', '?')}")
    print(f"  검증 정확도: {checkpoint.get('val_acc', 0):.2%}")

    # 더미 입력 생성
    dummy_landmarks = torch.randn(
        1, NUM_LANDMARKS, NUM_COORDS
    )  # (1, 21, 2)
    dummy_fingers = torch.randn(1, 5)  # (1, 5)

    # ONNX 내보내기
    torch.onnx.export(
        model,
        (dummy_landmarks, dummy_fingers),
        str(output_path),
        opset_version=opset_version,
        input_names=["landmarks", "finger_states"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "finger_states": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"  ONNX 내보내기 완료: {output_path}")

    # 파일 크기 확인
    size_kb = output_path.stat().st_size / 1024
    print(f"  모델 크기: {size_kb:.1f} KB")


def verify_onnx(onnx_path: Path) -> bool:
    """ONNX 모델의 유효성을 검증.

    Args:
        onnx_path: ``.onnx`` 파일 경로.

    Returns:
        검증 성공 여부.
    """
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        # 테스트 추론
        test_landmarks = np.random.randn(
            1, NUM_LANDMARKS, NUM_COORDS
        ).astype(np.float32)
        test_fingers = np.random.randn(1, 5).astype(np.float32)

        outputs = session.run(
            None,
            {
                "landmarks": test_landmarks,
                "finger_states": test_fingers,
            },
        )

        logits = outputs[0]
        print(f"  검증 성공! 출력 형상: {logits.shape}")
        return True

    except ImportError:
        print("  ⚠ onnxruntime 미설치. 검증 생략.")
        print("    설치: pip install onnxruntime")
        return True

    except Exception as e:
        print(f"  ✗ 검증 실패: {e}")
        return False


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="MotionMagic PyTorch → ONNX 변환"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/gesture_cnn_best.pth",
        help="입력 .pth 체크포인트 (기본: models/gesture_cnn_best.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/gesture_cnn.onnx",
        help="출력 .onnx 파일 (기본: models/gesture_cnn.onnx)",
    )

    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"체크포인트 파일 없음: {checkpoint_path}")
        print("먼저 02_train_model.py를 실행하세요.")
        return

    print(f"\n{'='*50}")
    print(" PyTorch → ONNX 변환")
    print(f"{'='*50}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(checkpoint_path, output_path)

    print(f"\n  ONNX 모델 검증 중...")
    verify_onnx(output_path)

    print(f"\n{'='*50}")
    print(" 변환 완료!")
    print(f"{'='*50}")
    print("\n다음 단계: python scripts/04_evaluate_model.py")


if __name__ == "__main__":
    main()
