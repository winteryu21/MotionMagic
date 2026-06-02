"""06 — 현재 구조의 오른손 조준/핀치 발사 테스트.

기존 trigger + 손목 튕김 테스트는 현재 입력 구조와 맞지 않는다.
이 스크립트는 07 제스처 모드 진단 화면을 오른손 집중 모드로 실행한다.

사용법:
    python scripts/06_trigger_shoot_test.py
    python scripts/06_trigger_shoot_test.py --camera 1 --ema-alpha 0.25
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_mode_test_module() -> ModuleType:
    script_path = Path(__file__).with_name("07_gesture_mode_test.py")
    spec = importlib.util.spec_from_file_location("gesture_mode_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"제스처 모드 테스트 모듈을 불러올 수 없습니다: {script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="MotionMagic 오른손 조준/핀치 발사 테스트"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 디바이스 ID (기본: 0)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="오른손 조준점 EMA 현재 입력 반영 비율 (기본: 0.3)",
    )
    parser.add_argument(
        "--aim-sensitivity",
        type=float,
        default=3.0,
        help="화면 중심 기준 조준 감도 (기본: 3.0)",
    )
    parser.add_argument(
        "--aim-sensitivity-x",
        type=float,
        default=None,
        help="X축 조준 감도. 미지정 시 --aim-sensitivity 사용",
    )
    parser.add_argument(
        "--aim-sensitivity-y",
        type=float,
        default=None,
        help="Y축 조준 감도. 미지정 시 --aim-sensitivity 사용",
    )
    parser.add_argument(
        "--aim-center-x",
        type=float,
        default=0.5,
        help="화면 중앙으로 매핑할 입력 X 좌표 (기본: 0.5)",
    )
    parser.add_argument(
        "--aim-center-y",
        type=float,
        default=0.5,
        help="화면 중앙으로 매핑할 입력 Y 좌표 (기본: 0.5)",
    )
    parser.add_argument(
        "--aim-anchor",
        choices=("index", "pinch"),
        default="index",
        help="조준점 기준: 검지 끝(index) 또는 엄지/검지 중심(pinch)",
    )
    parser.set_defaults(swap_handedness=True)
    parser.add_argument(
        "--no-swap-handedness",
        action="store_false",
        dest="swap_handedness",
        help="MediaPipe handedness 라벨을 그대로 사용",
    )
    parser.add_argument(
        "--pinch-open",
        type=float,
        default=0.70,
        help="재장전으로 볼 엄지-검지 거리 (기본: 0.70)",
    )
    parser.add_argument(
        "--pinch-closed",
        type=float,
        default=0.55,
        help="발사로 볼 엄지-검지 닫힘 거리 (기본: 0.55)",
    )
    parser.add_argument(
        "--pinch-velocity",
        type=float,
        default=0.60,
        help="발사로 볼 닫힘 속도 임계값 (기본: 0.60)",
    )
    parser.add_argument(
        "--pinch-delta",
        type=float,
        default=0.12,
        help="발사로 볼 프레임 간 닫힘 변화량 (기본: 0.12)",
    )

    args = parser.parse_args()
    module = _load_mode_test_module()
    module.run_test(
        camera_id=args.camera,
        ema_alpha=args.ema_alpha,
        aim_sensitivity=args.aim_sensitivity,
        aim_sensitivity_x=args.aim_sensitivity_x,
        aim_sensitivity_y=args.aim_sensitivity_y,
        aim_center_x=args.aim_center_x,
        aim_center_y=args.aim_center_y,
        aim_anchor=args.aim_anchor,
        swap_handedness=args.swap_handedness,
        right_only=True,
        pinch_open_threshold=args.pinch_open,
        pinch_closed_threshold=args.pinch_closed,
        pinch_close_velocity=args.pinch_velocity,
        pinch_close_delta=args.pinch_delta,
    )


if __name__ == "__main__":
    main()
