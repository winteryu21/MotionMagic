"""하이브리드 CNN 모델 정의.

좌표 63차원 → Conv1d 브랜치, 손가락 상태 5차원 → Linear 브랜치.
두 브랜치를 결합하여 제스처를 4클래스로 분류한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.game.settings import NUM_COORDS, NUM_FINGER_FEATURES, NUM_GESTURE_CLASSES, NUM_LANDMARKS


class GestureCNN(nn.Module):
    """하이브리드 CNN 제스처 분류 모델.

    Args:
        num_classes: 분류할 제스처 클래스 수.
    """

    def __init__(self, num_classes: int = NUM_GESTURE_CLASSES) -> None:
        super().__init__()

        # Conv1d 브랜치: 좌표 (B, 3, 21)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(NUM_COORDS, 64, kernel_size=3),   # (B, 64, 19)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),           # (B, 128, 17)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                     # (B, 128, 1)
            nn.Flatten(),                                # (B, 128)
        )

        # Linear 브랜치: 손가락 상태 (B, 5)
        self.finger_branch = nn.Sequential(
            nn.Linear(NUM_FINGER_FEATURES, 32),
            nn.ReLU(),
        )

        # 결합 분류기
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        coords: torch.Tensor,
        finger_states: torch.Tensor,
    ) -> torch.Tensor:
        """순전파.

        Args:
            coords: 관절 좌표 텐서 (B, 3, 21).
            finger_states: 손가락 펼침/접힘 상태 (B, 5).

        Returns:
            로짓 텐서 (B, num_classes).
        """
        conv_out = self.conv_branch(coords)          # (B, 128)
        finger_out = self.finger_branch(finger_states)  # (B, 32)
        combined = torch.cat([conv_out, finger_out], dim=1)  # (B, 160)
        return self.classifier(combined)             # (B, num_classes)
