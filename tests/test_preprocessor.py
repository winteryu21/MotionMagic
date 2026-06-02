"""preprocessor 모듈 테스트."""

from __future__ import annotations

import numpy as np

from src.ai.preprocessor import (
    augment_mirror,
    augment_noise,
    augment_rotation,
    extract_2d_landmarks,
    extract_finger_states,
    extract_landmarks,
    normalize_landmarks,
    preprocess_single,
)

# ---------------------------------------------------------------------------
# 테스트 유틸: 가짜 랜드마크 생성
# ---------------------------------------------------------------------------


def _make_fake_landmarks(spread: float = 0.5) -> np.ndarray:
    """테스트용 가짜 21개 관절 좌표 ``(21, 3)`` 생성."""
    rng = np.random.default_rng(42)
    # 손목을 (0.5, 0.5)에 놓고, 다른 관절은 주변에 분포
    base = np.full((21, 3), 0.5, dtype=np.float32)
    offsets = rng.uniform(-spread, spread, size=(21, 3)).astype(np.float32)
    offsets[0] = [0.0, 0.0, 0.0]  # 손목은 중심
    return base + offsets


def _make_open_hand_landmarks() -> np.ndarray:
    """모든 손가락이 펴진 손 모양의 랜드마크 ``(21, 2)``.

    손목(0)을 원점으로 하고, 각 손가락의 tip이 MCP보다
    확실히 멀리 위치하도록 배치한다.
    """
    lm = np.zeros((21, 2), dtype=np.float32)
    # 손목
    lm[0] = [0.0, 0.0]

    # 엄지: base=2, tip=4
    lm[1] = [0.3, -0.1]
    lm[2] = [0.5, -0.2]  # MCP
    lm[3] = [0.7, -0.3]
    lm[4] = [0.9, -0.4]  # tip

    # 검지: base=5, tip=8
    lm[5] = [0.2, -0.4]
    lm[6] = [0.2, -0.6]
    lm[7] = [0.2, -0.8]
    lm[8] = [0.2, -1.0]

    # 중지: base=9, tip=12
    lm[9] = [0.0, -0.5]
    lm[10] = [0.0, -0.7]
    lm[11] = [0.0, -0.9]
    lm[12] = [0.0, -1.1]

    # 약지: base=13, tip=16
    lm[13] = [-0.2, -0.4]
    lm[14] = [-0.2, -0.6]
    lm[15] = [-0.2, -0.8]
    lm[16] = [-0.2, -1.0]

    # 새끼: base=17, tip=20
    lm[17] = [-0.4, -0.3]
    lm[18] = [-0.4, -0.5]
    lm[19] = [-0.4, -0.7]
    lm[20] = [-0.4, -0.9]

    return lm


def _make_fist_landmarks() -> np.ndarray:
    """모든 손가락이 접힌 주먹 랜드마크 ``(21, 2)``.

    tip이 MCP보다 손목에 가깝게 위치한다.
    """
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = [0.0, 0.0]

    # 엄지: tip이 MCP보다 손목에 가까움
    lm[2] = [0.4, -0.2]
    lm[4] = [0.3, -0.1]

    # 검지
    lm[5] = [0.2, -0.4]
    lm[8] = [0.15, -0.25]

    # 중지
    lm[9] = [0.0, -0.5]
    lm[12] = [0.0, -0.3]

    # 약지
    lm[13] = [-0.2, -0.4]
    lm[16] = [-0.15, -0.25]

    # 새끼
    lm[17] = [-0.4, -0.3]
    lm[20] = [-0.3, -0.2]

    return lm


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


class TestExtractLandmarks:
    """extract_landmarks 함수 테스트."""

    def test_from_dict_list(self) -> None:
        """딕셔너리 리스트에서 3D 좌표 추출."""
        dicts = [
            {"x": float(i), "y": float(i + 1), "z": float(i + 2)} for i in range(21)
        ]
        result = extract_landmarks(dicts)
        assert result.shape == (21, 3)
        assert result[0][0] == 0.0
        assert result[0][1] == 1.0
        assert result[0][2] == 2.0

    def test_from_3d_array(self) -> None:
        """(21, 3) 배열을 그대로 반환."""
        arr_3d = np.random.randn(21, 3).astype(np.float32)
        result = extract_landmarks(arr_3d)
        assert result.shape == (21, 3)
        np.testing.assert_array_equal(result, arr_3d)

    def test_from_2d_array(self) -> None:
        """(21, 2) 배열은 Z축 0으로 패딩."""
        arr_2d = np.random.randn(21, 2).astype(np.float32)
        result = extract_landmarks(arr_2d)
        assert result.shape == (21, 3)
        np.testing.assert_array_equal(result[:, :2], arr_2d)
        np.testing.assert_array_equal(result[:, 2], np.zeros(21))

    def test_extract_2d_compatibility(self) -> None:
        """기존 2D 추출 함수는 X/Y만 반환."""
        arr_3d = np.random.randn(21, 3).astype(np.float32)
        result = extract_2d_landmarks(arr_3d)
        assert result.shape == (21, 2)
        np.testing.assert_array_equal(result, arr_3d[:, :2])


class TestNormalizeLandmarks:
    """normalize_landmarks 함수 테스트."""

    def test_wrist_at_origin(self) -> None:
        """정규화 후 손목이 원점에 위치."""
        lm = _make_fake_landmarks()
        result = normalize_landmarks(lm)
        assert result is not None
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0, 0.0])

    def test_scale_normalization(self) -> None:
        """정규화 후 손목→중지MCP 거리가 1.0."""
        lm = _make_open_hand_landmarks()
        # 원래 좌표에 오프셋 추가 (손목이 원점이 아니도록)
        lm_shifted = lm + np.array([2.0, 3.0], dtype=np.float32)
        result = normalize_landmarks(lm_shifted)
        assert result is not None
        dist = np.linalg.norm(result[0] - result[9])
        np.testing.assert_almost_equal(dist, 1.0, decimal=5)

    def test_zero_distance_returns_none(self) -> None:
        """모든 점이 같은 위치면 None 반환."""
        lm = np.full((21, 2), 0.5, dtype=np.float32)
        result = normalize_landmarks(lm)
        assert result is None


class TestExtractFingerStates:
    """extract_finger_states 함수 테스트."""

    def test_open_hand(self) -> None:
        """모든 손가락이 펴진 상태 = [1, 1, 1, 1, 1]."""
        lm = _make_open_hand_landmarks()
        states = extract_finger_states(lm)
        assert states.shape == (5,)
        np.testing.assert_array_equal(states, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_fist(self) -> None:
        """주먹 = [0, 0, 0, 0, 0]."""
        lm = _make_fist_landmarks()
        states = extract_finger_states(lm)
        assert states.shape == (5,)
        np.testing.assert_array_equal(states, [0.0, 0.0, 0.0, 0.0, 0.0])


class TestAugmentation:
    """데이터 증강 함수 테스트."""

    def test_rotation_preserves_shape(self) -> None:
        """회전 증강 후 형상 유지."""
        lm = _make_fake_landmarks()
        result = augment_rotation(lm, max_angle_deg=30.0)
        assert result.shape == (21, 3)

    def test_rotation_changes_values(self) -> None:
        """회전 증강 후 값이 변함 (0이 아닌 각도 시)."""
        lm = _make_open_hand_landmarks()
        rng = np.random.default_rng(123)
        result = augment_rotation(lm, max_angle_deg=30.0, rng=rng)
        # 손목(원점)은 회전에 무관하게 그대로
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        # 다른 점은 변해야 함
        assert not np.allclose(result[8], lm[8])

    def test_noise_preserves_shape(self) -> None:
        """노이즈 증강 후 형상 유지."""
        lm = _make_fake_landmarks()
        result = augment_noise(lm, noise_std=0.01)
        assert result.shape == (21, 3)

    def test_mirror_flips_x_only(self) -> None:
        """좌우 반전 증강은 X축만 반전."""
        lm = _make_fake_landmarks()
        result = augment_mirror(lm)
        np.testing.assert_array_equal(result[:, 0], -lm[:, 0])
        np.testing.assert_array_equal(result[:, 1:], lm[:, 1:])


class TestPreprocessSingle:
    """preprocess_single 통합 테스트."""

    def test_returns_tuple(self) -> None:
        """정상 입력에 대해 (landmarks, finger_states) 튜플 반환."""
        dicts = [
            {"x": float(i) * 0.05, "y": float(i) * 0.03, "z": 0.0} for i in range(21)
        ]
        result = preprocess_single(dicts)
        assert result is not None
        landmarks, finger_states = result
        assert landmarks.shape == (21, 3)
        assert finger_states.shape == (5,)
