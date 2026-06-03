"""dataset 모듈 테스트."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.ai.dataset import GestureDataset
from src.ai.preprocessor import LABEL_TO_INDEX


def test_dataset_pads_2d_landmarks_and_skips_idle() -> None:
    """기존 2D 샘플은 3D로 보정하고 idle 샘플은 제외."""
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        data_dir = Path(tmpdir)
        landmarks_2d = np.zeros((21, 2), dtype=np.float32).tolist()
        samples = [
            {"label": "rock", "landmarks": landmarks_2d},
            {"label": "idle", "landmarks": landmarks_2d},
        ]
        with open(data_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(samples, f)

        dataset = GestureDataset(data_dir)

        assert len(dataset) == 1
        landmarks, finger_states, label = dataset[0]
        assert landmarks.shape == (21, 3)
        assert finger_states.shape == (5,)
        assert int(label.item()) == LABEL_TO_INDEX["rock"]
