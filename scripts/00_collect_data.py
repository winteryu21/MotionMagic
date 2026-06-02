"""00 — 제스처 데이터 수집 CLI.

웹캠을 열어 실시간으로 손 관절 좌표를 수집하고 JSON 파일로 저장한다.
자동 캡처 모드가 기본이며, 한 손 제스처 4개 클래스를 순차적으로 수집한다.

사용법:
    python scripts/00_collect_data.py
    python scripts/00_collect_data.py --gesture rock --samples 500
    python scripts/00_collect_data.py --all --samples 1000
    python scripts/00_collect_data.py --specials --samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.collector import TWO_HAND_GESTURES, GestureCollector
from src.ai.preprocessor import GESTURE_LABELS


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="MotionMagic 제스처 데이터 수집기")
    parser.add_argument(
        "--gesture",
        type=str,
        default=None,
        help="수집할 제스처 라벨 (rock, paper, scissors, trigger, clasp, sonaldo)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="한 손 기본 제스처를 순차적으로 수집",
    )
    parser.add_argument(
        "--specials",
        action="store_true",
        help="양손 특수 제스처(clasp, sonaldo)를 순차적으로 수집",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="클래스당 수집할 프레임 수 (기본: 500)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 디바이스 ID (기본: 0)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        choices=(1, 2),
        default=None,
        help="감지할 손 개수. 미지정 시 양손 제스처는 2, 나머지는 1",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="자동 캡처 간격 밀리초 (기본: 30ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="출력 디렉터리 (기본: data/raw)",
    )

    args = parser.parse_args()
    selected_modes = sum(
        bool(value) for value in (args.gesture, args.all, args.specials)
    )
    if selected_modes > 1:
        parser.error("--gesture, --all, --specials 중 하나만 선택하세요.")

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if args.all or args.specials:
        gestures = (
            list(GESTURE_LABELS.values()) if args.all else sorted(TWO_HAND_GESTURES)
        )
        mode_name = "한 손 기본" if args.all else "양손 특수"
        print(f"\n{mode_name} 제스처 수집 모드 ({len(gestures)}개 클래스)")
        print(f"클래스당 {args.samples} 프레임 수집 예정\n")

        for i, gesture in enumerate(gestures, 1):
            print(f"\n{'='*50}")
            print(f" [{i}/{len(gestures)}] 제스처: {gesture}")
            print(f"{'='*50}")
            input(f"  '{gesture}' 제스처를 준비한 뒤 Enter를 누르세요...")

            collector = GestureCollector(
                camera_id=args.camera,
                max_num_hands=args.hands or (2 if gesture in TWO_HAND_GESTURES else 1),
            )
            collector.collect(
                gesture=gesture,
                num_samples=args.samples,
                output_dir=output_dir,
                auto_capture=True,
                capture_interval_ms=args.interval,
            )

        print(f"\n{mode_name} 데이터 수집 완료! → {output_dir}")

    elif args.gesture:
        max_num_hands = args.hands or (2 if args.gesture in TWO_HAND_GESTURES else 1)
        collector = GestureCollector(
            camera_id=args.camera,
            max_num_hands=max_num_hands,
        )
        collector.collect(
            gesture=args.gesture,
            num_samples=args.samples,
            output_dir=output_dir,
            auto_capture=True,
            capture_interval_ms=args.interval,
        )

    else:
        parser.print_help()
        print("\n예시:")
        print("  python scripts/00_collect_data.py --all --samples 1000")
        print("  python scripts/00_collect_data.py --specials --samples 500")
        print("  python scripts/00_collect_data.py --gesture rock --samples 500")


if __name__ == "__main__":
    main()
