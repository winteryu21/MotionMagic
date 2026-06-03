"""01 — 데이터 전처리.

``data/raw/`` 의 원시 JSON 파일들을 읽어 정규화(위치·크기) 처리 후
``data/processed/`` 에 저장하고, 8:1:1 비율로 Train/Val/Test 분할한다.

사용법:
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.preprocessor import (
    LABEL_TO_INDEX,
    extract_landmarks,
    normalize_landmarks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # Train, Val, Test
SPLIT_NAMES = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(path: str) -> Path:
    """상대경로를 프로젝트 루트 기준 경로로 변환한다.

    Args:
        path: CLI에서 받은 경로 문자열.

    Returns:
        절대경로 또는 프로젝트 루트 기준 경로.
    """
    result = Path(path)
    if result.is_absolute():
        return result
    return PROJECT_ROOT / result


def load_raw_data(raw_dir: Path) -> list[dict]:
    """``data/raw/`` 디렉터리의 모든 JSON 파일을 읽어 통합.

    Args:
        raw_dir: 원시 데이터 디렉터리 경로.

    Returns:
        전체 샘플 리스트.
    """
    all_samples: list[dict] = []
    json_files = sorted(raw_dir.glob("*.json"))

    if not json_files:
        logger.warning("원시 데이터 없음: %s", raw_dir)
        return all_samples

    for fpath in json_files:
        with open(fpath, encoding="utf-8") as f:
            samples = json.load(f)
        all_samples.extend(samples)
        logger.info("  로드: %s (%d samples)", fpath.name, len(samples))

    return all_samples


def preprocess_samples(samples: list[dict]) -> list[dict]:
    """원시 샘플들을 정규화하여 전처리 결과를 반환.

    Args:
        samples: 원시 샘플 리스트 (각각 ``label``, ``landmarks`` 포함).

    Returns:
        정규화된 샘플 리스트.
    """
    processed: list[dict] = []
    skipped = 0

    for sample in samples:
        label = sample["label"]
        if label not in LABEL_TO_INDEX:
            skipped += 1
            continue

        if "landmarks" not in sample:
            skipped += 1
            continue

        raw_landmarks = np.array(sample["landmarks"], dtype=np.float32)
        coords = extract_landmarks(raw_landmarks)
        normalized = normalize_landmarks(coords)

        if normalized is None:
            skipped += 1
            continue

        processed.append(
            {
                "label": label,
                "landmarks": normalized.tolist(),
                "handedness": sample.get("handedness"),
            }
        )

    if skipped > 0:
        logger.info("건너뛴 샘플: %d개 (정규화 실패 또는 잘못된 라벨)", skipped)

    return processed


def split_and_save(
    samples: list[dict],
    output_dir: Path,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> None:
    """전처리된 샘플을 셔플 후 Train/Val/Test로 분할 저장.

    Args:
        samples: 전처리된 샘플 리스트.
        output_dir: 출력 디렉터리 (``data/processed/``).
        ratios: (train, val, test) 분할 비율.
    """
    rng = np.random.default_rng(42)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    n = len(samples)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": [samples[i] for i in indices[:n_train]],
        "val": [samples[i] for i in indices[n_train : n_train + n_val]],
        "test": [samples[i] for i in indices[n_train + n_val :]],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        filepath = output_dir / f"{split_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False)
        print(f"  {split_name}: {len(split_data)} samples → {filepath}")
        logger.info("%s: %d samples → %s", split_name, len(split_data), filepath)


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="MotionMagic 데이터 전처리 및 분할")
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw",
        help="원시 데이터 디렉터리 (기본: data/raw)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="출력 디렉터리 (기본: data/processed)",
    )

    args = parser.parse_args()
    raw_dir = _project_path(args.raw)
    output_dir = _project_path(args.out)

    print(f"\n{'='*50}")
    print(" 데이터 전처리 시작")
    print(f"{'='*50}")

    # 1) 원시 데이터 로드
    print(f"\n[1/3] 원시 데이터 로드: {raw_dir}")
    raw_samples = load_raw_data(raw_dir)
    if not raw_samples:
        print("원시 데이터가 없습니다. 먼저 00_collect_data.py를 실행하세요.")
        return

    print(f"  총 {len(raw_samples)}개 샘플 로드 완료")

    # 2) 전처리
    print("\n[2/3] 정규화 전처리 중...")
    processed = preprocess_samples(raw_samples)
    print(f"  전처리 완료: {len(processed)}개 샘플")

    # 3) 분할 및 저장
    print(f"\n[3/3] Train/Val/Test 분할 ({SPLIT_RATIOS})...")
    split_and_save(processed, output_dir)

    # 클래스별 통계 출력
    from collections import Counter

    label_counts = Counter(s["label"] for s in processed)
    print("\n클래스별 통계:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"\n{'='*50}")
    print(" 전처리 완료!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
