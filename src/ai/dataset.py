"""PyTorch 제스처 데이터셋.

전처리된 JSON 파일을 읽어 ``(좌표 텐서, 손가락 상태 텐서, 라벨)``
삼중 항으로 반환하는 ``torch.utils.data.Dataset`` 구현체.
학습 시 데이터 증강(회전, 노이즈)을 선택적으로 적용한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ai.preprocessor import (
    LABEL_TO_INDEX,
    NUM_COORDS,
    NUM_FINGERS,
    NUM_LANDMARKS,
    augment_noise,
    augment_rotation,
    extract_finger_states,
)

logger = logging.getLogger(__name__)


class GestureDataset(Dataset):
    """제스처 인식 학습용 PyTorch 데이터셋.

    전처리가 완료된 JSON 파일들을 로드하여 학습에 필요한
    텐서 형태로 변환한다.

    각 샘플은 다음을 포함한다:
    - ``landmarks``: ``(21, 2)`` 정규화된 2D 좌표
    - ``finger_states``: ``(5,)`` 손가락 상태 힌트
    - ``label``: 정수 라벨 (0~4)

    Args:
        data_dir: 전처리된 JSON 파일이 있는 디렉터리 경로.
        augment: 학습 시 데이터 증강 적용 여부.
        max_rotation_deg: 최대 회전 각도 (증강 시 사용).
        noise_std: 가우시안 노이즈 표준편차 (증강 시 사용).
    """

    def __init__(
        self,
        data_dir: Path | str,
        augment: bool = False,
        max_rotation_deg: float = 30.0,
        noise_std: float = 0.01,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._augment = augment
        self._max_rotation_deg = max_rotation_deg
        self._noise_std = noise_std
        self._rng = np.random.default_rng()

        self._landmarks: list[np.ndarray] = []
        self._labels: list[int] = []

        self._load_data()

    def _load_data(self) -> None:
        """디렉터리 내 모든 JSON 파일을 읽어 메모리에 적재."""
        json_files = sorted(self._data_dir.glob("*.json"))
        if not json_files:
            logger.warning("데이터 파일 없음: %s", self._data_dir)
            return

        for fpath in json_files:
            with open(fpath, encoding="utf-8") as f:
                samples = json.load(f)

            for sample in samples:
                label_str = sample["label"]
                if label_str not in LABEL_TO_INDEX:
                    logger.warning("알 수 없는 라벨 '%s' 무시", label_str)
                    continue

                landmarks = np.array(
                    sample["landmarks"], dtype=np.float32
                )  # (21, 2)

                if landmarks.shape != (NUM_LANDMARKS, NUM_COORDS):
                    logger.warning(
                        "잘못된 landmark 형상 %s, 무시", landmarks.shape
                    )
                    continue

                self._landmarks.append(landmarks)
                self._labels.append(LABEL_TO_INDEX[label_str])

        logger.info(
            "데이터 로드 완료: %d samples from %s",
            len(self._landmarks),
            self._data_dir,
        )

    def __len__(self) -> int:
        """데이터셋 크기."""
        return len(self._landmarks)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """인덱스에 해당하는 샘플 반환.

        Args:
            idx: 샘플 인덱스.

        Returns:
            ``(landmarks, finger_states, label)`` 튜플.
            - landmarks: ``(21, 2)`` float32 텐서
            - finger_states: ``(5,)`` float32 텐서
            - label: ``()`` int64 텐서
        """
        landmarks = self._landmarks[idx].copy()  # (21, 2)
        label = self._labels[idx]

        # 데이터 증강 (학습 시만)
        if self._augment:
            landmarks = augment_rotation(
                landmarks,
                max_angle_deg=self._max_rotation_deg,
                rng=self._rng,
            )
            landmarks = augment_noise(
                landmarks,
                noise_std=self._noise_std,
                rng=self._rng,
            )

        # 증강 후 손가락 상태 재계산
        finger_states = extract_finger_states(landmarks)  # (5,)

        return (
            torch.from_numpy(landmarks),  # (21, 2)
            torch.from_numpy(finger_states),  # (5,)
            torch.tensor(label, dtype=torch.long),  # ()
        )

    def get_class_counts(self) -> dict[str, int]:
        """각 클래스별 샘플 수 반환.

        Returns:
            ``{"rock": 1000, "paper": 950, ...}`` 형태의 딕셔너리.
        """
        from src.ai.preprocessor import GESTURE_LABELS

        counts: dict[str, int] = {name: 0 for name in GESTURE_LABELS.values()}
        for label_idx in self._labels:
            counts[GESTURE_LABELS[label_idx]] += 1
        return counts
