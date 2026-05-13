"""CNN 모델 입출력 형상 테스트."""

from __future__ import annotations

import torch

from src.ai.model import GestureCNN
from src.game.settings import NUM_GESTURE_CLASSES


class TestGestureCNN:
    """GestureCNN 모델 형상 검증."""

    def setup_method(self) -> None:
        self.model = GestureCNN()
        self.model.eval()

    def test_output_shape(self) -> None:
        """출력 shape이 (B, num_classes)인지 확인."""
        batch_size = 4
        coords = torch.randn(batch_size, 3, 21)
        finger_states = torch.randn(batch_size, 5)

        output = self.model(coords, finger_states)
        assert output.shape == (batch_size, NUM_GESTURE_CLASSES)

    def test_single_sample(self) -> None:
        """단일 샘플 추론이 정상 동작하는지 확인."""
        coords = torch.randn(1, 3, 21)
        finger_states = torch.randn(1, 5)

        output = self.model(coords, finger_states)
        assert output.shape == (1, NUM_GESTURE_CLASSES)
