"""model 모듈 테스트."""

from __future__ import annotations

import torch

from src.ai.model import GestureCNN
from src.ai.preprocessor import NUM_CLASSES


class TestGestureCNN:
    """GestureCNN 모델 테스트."""

    def test_output_shape(self) -> None:
        """순전파 출력 형상 검증."""
        model = GestureCNN(num_classes=NUM_CLASSES)
        landmarks = torch.randn(4, 21, 3)  # (B=4, 21, 3)
        finger_states = torch.randn(4, 5)  # (B=4, 5)

        logits = model(landmarks, finger_states)
        assert logits.shape == (4, NUM_CLASSES)

    def test_single_sample(self) -> None:
        """단일 샘플 추론."""
        model = GestureCNN()
        landmarks = torch.randn(1, 21, 3)
        finger_states = torch.randn(1, 5)

        logits = model(landmarks, finger_states)
        assert logits.shape == (1, NUM_CLASSES)

    def test_custom_num_classes(self) -> None:
        """커스텀 클래스 수."""
        model = GestureCNN(num_classes=3)
        landmarks = torch.randn(2, 21, 3)
        finger_states = torch.randn(2, 5)

        logits = model(landmarks, finger_states)
        assert logits.shape == (2, 3)

    def test_gradient_flow(self) -> None:
        """역전파 그래디언트가 흐르는지 확인."""
        model = GestureCNN()
        landmarks = torch.randn(2, 21, 3, requires_grad=True)
        finger_states = torch.randn(2, 5, requires_grad=True)

        logits = model(landmarks, finger_states)
        loss = logits.sum()
        loss.backward()

        assert landmarks.grad is not None
        assert finger_states.grad is not None

    def test_eval_mode_deterministic(self) -> None:
        """eval 모드에서 동일 입력은 동일 출력."""
        model = GestureCNN()
        model.eval()

        landmarks = torch.randn(1, 21, 3)
        finger_states = torch.randn(1, 5)

        with torch.no_grad():
            out1 = model(landmarks, finger_states)
            out2 = model(landmarks, finger_states)

        torch.testing.assert_close(out1, out2)
