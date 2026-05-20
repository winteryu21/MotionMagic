"""02 — 모델 학습.

전처리된 데이터셋으로 1D CNN + Finger State 하이브리드 모델을 학습하고,
최고 성능 체크포인트를 ``models/`` 에 저장한다.

사용법:
    python scripts/02_train_model.py
    python scripts/02_train_model.py --epochs 100 --batch-size 32 --lr 0.001
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.ai.dataset import GestureDataset
from src.ai.model import GestureCNN
from src.ai.trainer import GestureTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="MotionMagic 제스처 분류 모델 학습"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed",
        help="전처리된 데이터 디렉터리 (기본: data/processed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="최대 에포크 수 (기본: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="배치 크기 (기본: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="학습률 (기본: 0.001)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="드롭아웃 비율 (기본: 0.3)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="모델 저장 디렉터리 (기본: models)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data)
    save_dir = Path(args.save_dir)

    # 데이터셋 로드
    print(f"\n{'='*60}")
    print(" 데이터셋 로드 중...")
    print(f"{'='*60}")

    train_dataset = GestureDataset(data_dir, augment=True)
    val_dataset = GestureDataset(data_dir, augment=False)

    # train.json과 val.json이 분리되지 않은 경우를 대비하여
    # data_dir 내에 train.json, val.json이 있으면 각각 로드
    train_dir = data_dir
    val_dir = data_dir

    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"

    if train_json.exists() and val_json.exists():
        # 분할된 파일이 있으면 임시 디렉터리로 분리 로드
        import json
        import tempfile
        import shutil

        # train 데이터를 임시 디렉터리에 복사
        train_tmp = Path(tempfile.mkdtemp())
        val_tmp = Path(tempfile.mkdtemp())

        shutil.copy(train_json, train_tmp / "train.json")
        shutil.copy(val_json, val_tmp / "val.json")

        train_dataset = GestureDataset(train_tmp, augment=True)
        val_dataset = GestureDataset(val_tmp, augment=False)

        # 임시 디렉터리 정리
        shutil.rmtree(train_tmp)
        shutil.rmtree(val_tmp)

    if len(train_dataset) == 0:
        print("학습 데이터가 없습니다. 먼저 01_preprocess.py를 실행하세요.")
        return

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # 클래스별 통계
    train_counts = train_dataset.get_class_counts()
    print(f"\n  클래스별 학습 데이터:")
    for label, count in train_counts.items():
        print(f"    {label}: {count}")

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 모델 및 트레이너 생성
    model = GestureCNN(dropout_rate=args.dropout)
    trainer = GestureTrainer(
        model=model,
        lr=args.lr,
        epochs=args.epochs,
    )

    # 학습 실행
    best_path = trainer.train(train_loader, val_loader, save_dir=save_dir)

    # 학습 이력 그래프 저장
    trainer.plot_history(save_dir / "training_history.png")

    print(f"\n최고 모델: {best_path}")
    print("다음 단계: python scripts/03_export_onnx.py")


if __name__ == "__main__":
    main()
