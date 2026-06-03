"""1D CNN + 손가락 상태 결합 제스처 분류 모델.

3D 랜드마크 좌표에서 1D 합성곱으로 지역 특징을 추출한 뒤,
5차원 손가락 상태 힌트 벡터와 결합하여 최종 분류한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ai.preprocessor import NUM_CLASSES, NUM_COORDS, NUM_FINGERS


class GestureCNN(nn.Module):
    """1D CNN + Finger-State Hybrid 제스처 분류 신경망.

    아키텍처:
        1. 1D Conv 블록 (3층): ``(B, 3, 21)`` → 특징 벡터
        2. Finger State MLP: ``(B, 5)`` → 중간 표현
        3. Concat → FC → Softmax 분류

    Args:
        num_classes: 출력 클래스 수 (기본 4).
        dropout_rate: 드롭아웃 비율 (기본 0.3).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1D Conv 블록: 입력 (B, C=3, L=21)
        self.conv_block = nn.Sequential(
            # Conv1: (B, 3, 21) → (B, 32, 21)
            nn.Conv1d(
                in_channels=NUM_COORDS,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Conv2: (B, 32, 21) → (B, 64, 21)
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Conv3: (B, 64, 21) → (B, 128, 21)
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Global Average Pooling: (B, 128, 21) → (B, 128)
            nn.AdaptiveAvgPool1d(1),
        )

        # 손가락 상태 MLP: (B, 5) → (B, 16)
        self.finger_mlp = nn.Sequential(
            nn.Linear(NUM_FINGERS, 16),
            nn.ReLU(inplace=True),
        )

        # 결합 분류기: (B, 128 + 16) → (B, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        finger_states: torch.Tensor,
    ) -> torch.Tensor:
        """순전파.

        Args:
            landmarks: ``(B, 21, 3)`` 정규화된 3D 좌표 텐서.
            finger_states: ``(B, 5)`` 손가락 상태 힌트 텐서.

        Returns:
            ``(B, num_classes)`` 로짓(logits) 텐서.
        """
        # (B, 21, 3) → (B, 3, 21) for Conv1d
        x = landmarks.permute(0, 2, 1)  # (B, C=3, L=21)

        # 1D Conv 특징 추출
        x = self.conv_block(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)

        # 손가락 상태 MLP
        f = self.finger_mlp(finger_states)  # (B, 16)

        # 결합 및 분류
        combined = torch.cat([x, f], dim=1)  # (B, 144)
        logits = self.classifier(combined)  # (B, num_classes)

        return logits
