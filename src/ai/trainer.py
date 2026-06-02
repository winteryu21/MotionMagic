"""모델 학습 및 검증.

PyTorch 학습 루프를 실행하고, 에포크별 Loss/Accuracy를 기록하며,
최적 성능의 체크포인트를 ``models/`` 에 저장한다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ai.model import GestureCNN
from src.ai.preprocessor import NUM_CLASSES, NUM_COORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# 학습기 클래스
# ---------------------------------------------------------------------------


class GestureTrainer:
    """제스처 분류 모델 학습기.

    Args:
        model: 학습할 ``GestureCNN`` 모델 인스턴스.
        device: 학습 디바이스 (``"cpu"`` 또는 ``"cuda"``).
        lr: 학습률.
        epochs: 최대 에포크 수.
    """

    def __init__(
        self,
        model: GestureCNN | None = None,
        device: str | None = None,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if model is None:
            model = GestureCNN(num_classes=NUM_CLASSES)
        self._model = model.to(self._device)

        self._lr = lr
        self._epochs = epochs

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.5, patience=5
        )

        # 학습 이력
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path | None = None,
    ) -> Path:
        """모델 학습 실행.

        Args:
            train_loader: 학습 데이터 로더.
            val_loader: 검증 데이터 로더.
            save_dir: 체크포인트 저장 디렉터리.

        Returns:
            최적 모델 체크포인트 파일 경로.
        """
        if save_dir is None:
            save_dir = MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        best_model_path = save_dir / "gesture_cnn_best.pth"

        logger.info(
            "학습 시작: device=%s, epochs=%d, lr=%.4f",
            self._device,
            self._epochs,
            self._lr,
        )
        print(f"\n{'='*60}")
        print(f" 학습 시작: {self._device} | {self._epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(1, self._epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            self._scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            elapsed = time.time() - t0
            lr_now = self._optimizer.param_groups[0]["lr"]

            log_msg = (
                f"Epoch {epoch:3d}/{self._epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                f"LR: {lr_now:.6f} | {elapsed:.1f}s"
            )
            print(log_msg)
            logger.info(log_msg)

            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self._model.state_dict(),
                        "optimizer_state_dict": self._optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "num_classes": self._model.num_classes,
                        "num_coords": NUM_COORDS,
                    },
                    best_model_path,
                )
                logger.info("최고 모델 저장: epoch=%d, val_acc=%.4f", epoch, val_acc)
                print(f"  ★ 최고 모델 저장 (val_acc={val_acc:.2%})")

        print(f"\n{'='*60}")
        print(f" 학습 완료! 최고 검증 정확도: {best_val_acc:.2%}")
        print(f" 모델 저장: {best_model_path}")
        print(f"{'='*60}")

        return best_model_path

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """한 에포크 학습.

        Returns:
            ``(평균 loss, 정확도)`` 튜플.
        """
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for landmarks, finger_states, labels in loader:
            landmarks = landmarks.to(self._device)  # (B, 21, 3)
            finger_states = finger_states.to(self._device)  # (B, 5)
            labels = labels.to(self._device)  # (B,)

            self._optimizer.zero_grad()
            logits = self._model(landmarks, finger_states)  # (B, C)
            loss = self._criterion(logits, labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)  # (B,)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        """검증 세트 평가.

        Returns:
            ``(평균 loss, 정확도)`` 튜플.
        """
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for landmarks, finger_states, labels in loader:
            landmarks = landmarks.to(self._device)
            finger_states = finger_states.to(self._device)
            labels = labels.to(self._device)

            logits = self._model(landmarks, finger_states)
            loss = self._criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def plot_history(self, save_path: Path | str | None = None) -> None:
        """학습 이력 그래프를 생성하고 저장.

        Args:
            save_path: 그래프 저장 경로. ``None``이면 ``models/training_history.png``.
        """
        if save_path is None:
            save_path = MODELS_DIR / "training_history.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss 그래프
        ax1.plot(epochs, self.train_losses, "b-", label="Train Loss")
        ax1.plot(epochs, self.val_losses, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy 그래프
        ax2.plot(epochs, self.train_accs, "b-", label="Train Acc")
        ax2.plot(epochs, self.val_accs, "r-", label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"학습 이력 그래프 저장: {save_path}")
        logger.info("학습 이력 그래프 저장: %s", save_path)
