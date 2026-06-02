"""08 — AI 데이터 수집/전처리/학습 워크플로우 런처.

기존 ``00_collect_data.py`` → ``01_preprocess.py`` → ``02_train_model.py`` →
``03_export_onnx.py`` → ``04_evaluate_model.py`` 흐름을 한 진입점에서 실행한다.

사용법:
    python scripts/08_ai_workflow.py
    python scripts/08_ai_workflow.py --collect-basic --samples 500
    python scripts/08_ai_workflow.py --collect-specials --samples 500
    python scripts/08_ai_workflow.py --full-basic --epochs 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def _script_path(script_name: str) -> Path:
    """스크립트 파일의 절대 경로를 반환한다.

    Args:
        script_name: ``scripts/`` 아래의 Python 파일명.

    Returns:
        스크립트 절대 경로.
    """
    return SCRIPT_DIR / script_name


def _run_step(title: str, command: list[str]) -> None:
    """워크플로우 한 단계를 실행한다.

    Args:
        title: 사용자에게 표시할 단계 이름.
        command: ``sys.executable`` 뒤에 붙일 명령 인자 목록.
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)


def _collect_command(args: argparse.Namespace, collect_args: list[str]) -> list[str]:
    """수집 스크립트 명령을 조립한다.

    Args:
        args: CLI 인자 네임스페이스.
        collect_args: ``00_collect_data.py``에 전달할 수집 모드 인자.

    Returns:
        ``subprocess.run``에 전달할 명령 인자 목록.
    """
    command = [
        str(_script_path("00_collect_data.py")),
        *collect_args,
        "--samples",
        str(args.samples),
        "--camera",
        str(args.camera),
        "--interval",
        str(args.interval),
        "--output",
        args.raw,
    ]
    if args.hands is not None:
        command.extend(["--hands", str(args.hands)])
    return command


def _run_collect_basic(args: argparse.Namespace) -> None:
    """한 손 기본 제스처 수집을 실행한다."""
    _run_step("한 손 기본 제스처 수집", _collect_command(args, ["--all"]))


def _run_collect_specials(args: argparse.Namespace) -> None:
    """양손 특수 제스처 수집을 실행한다."""
    print(
        "\n참고: 현재 CNN 학습 파이프라인은 단일 손 4클래스 전용이라 "
        "clasp/sonaldo 원시 데이터는 아직 전처리/학습에 포함되지 않아요."
    )
    _run_step("양손 특수 제스처 수집", _collect_command(args, ["--specials"]))


def _run_collect_gesture(args: argparse.Namespace, gesture: str) -> None:
    """지정 제스처 수집을 실행한다.

    Args:
        args: CLI 인자 네임스페이스.
        gesture: 수집할 제스처 라벨.
    """
    _run_step(
        f"제스처 수집: {gesture}",
        _collect_command(args, ["--gesture", gesture]),
    )


def _run_preprocess(args: argparse.Namespace) -> None:
    """원시 데이터를 전처리한다."""
    _run_step(
        "데이터 전처리",
        [
            str(_script_path("01_preprocess.py")),
            "--raw",
            args.raw,
            "--out",
            args.processed,
        ],
    )


def _run_train(args: argparse.Namespace) -> None:
    """전처리 데이터를 사용해 모델을 학습한다."""
    _run_step(
        "모델 학습",
        [
            str(_script_path("02_train_model.py")),
            "--data",
            args.processed,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--save-dir",
            args.models,
        ],
    )


def _run_export(args: argparse.Namespace) -> None:
    """최신 PyTorch 체크포인트를 ONNX로 변환한다."""
    _run_step(
        "ONNX 변환",
        [
            str(_script_path("03_export_onnx.py")),
            "--checkpoint",
            str(Path(args.models) / "gesture_cnn_best.pth"),
            "--output",
            str(Path(args.models) / "gesture_cnn.onnx"),
        ],
    )


def _run_evaluate(args: argparse.Namespace) -> None:
    """테스트 세트로 모델을 평가한다."""
    _run_step(
        "모델 평가",
        [
            str(_script_path("04_evaluate_model.py")),
            "--model",
            str(Path(args.models) / "gesture_cnn.onnx"),
            "--test-data",
            str(Path(args.processed) / "test.json"),
        ],
    )


def _run_full_basic(args: argparse.Namespace) -> None:
    """한 손 기본 제스처의 전체 파이프라인을 실행한다."""
    _run_collect_basic(args)
    _run_preprocess(args)
    _run_train(args)
    _run_export(args)
    _run_evaluate(args)


def _run_selected_actions(args: argparse.Namespace) -> None:
    """CLI에서 선택된 작업들을 순서대로 실행한다.

    Args:
        args: CLI 인자 네임스페이스.
    """
    if args.full_basic:
        _run_full_basic(args)
        return

    if args.collect_basic:
        _run_collect_basic(args)
    if args.collect_specials:
        _run_collect_specials(args)
    for gesture in args.gesture or []:
        _run_collect_gesture(args, gesture)
    if args.preprocess:
        _run_preprocess(args)
    if args.train:
        _run_train(args)
    if args.export:
        _run_export(args)
    if args.evaluate:
        _run_evaluate(args)


def _has_selected_action(args: argparse.Namespace) -> bool:
    """CLI에서 실행 작업이 선택되었는지 반환한다.

    Args:
        args: CLI 인자 네임스페이스.

    Returns:
        하나 이상의 작업 선택 여부.
    """
    return any(
        (
            args.collect_basic,
            args.collect_specials,
            args.gesture,
            args.preprocess,
            args.train,
            args.export,
            args.evaluate,
            args.full_basic,
        )
    )


def _run_menu(args: argparse.Namespace) -> None:
    """텍스트 메뉴 기반 워크플로우를 실행한다.

    Args:
        args: CLI 인자 네임스페이스.
    """
    while True:
        print("\nMotionMagic AI Workflow")
        print(
            f"기본값: samples={args.samples}, camera={args.camera}, "
            f"epochs={args.epochs}, raw={args.raw}, processed={args.processed}"
        )
        print("  1. 한 손 기본 제스처 수집 (rock/paper/scissors/trigger)")
        print("  2. 양손 특수 제스처 수집 (clasp/sonaldo)")
        print("  3. 특정 제스처 수집")
        print("  4. 전처리")
        print("  5. 학습")
        print("  6. ONNX 변환")
        print("  7. 평가")
        print("  8. 한 손 기본 전체 실행 (수집→전처리→학습→변환→평가)")
        print("  q. 종료")

        choice = input("선택: ").strip().lower()
        if choice == "1":
            _run_collect_basic(args)
        elif choice == "2":
            _run_collect_specials(args)
        elif choice == "3":
            gesture = input("제스처 라벨: ").strip()
            if gesture:
                _run_collect_gesture(args, gesture)
        elif choice == "4":
            _run_preprocess(args)
        elif choice == "5":
            _run_train(args)
        elif choice == "6":
            _run_export(args)
        elif choice == "7":
            _run_evaluate(args)
        elif choice == "8":
            _run_full_basic(args)
        elif choice in {"q", "quit", "exit"}:
            return
        else:
            print("알 수 없는 선택입니다.")


def main() -> None:
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="MotionMagic AI 워크플로우 런처")
    parser.add_argument(
        "--collect-basic",
        action="store_true",
        help="한 손 기본 제스처를 순차 수집",
    )
    parser.add_argument(
        "--collect-specials",
        action="store_true",
        help="양손 특수 제스처(clasp, sonaldo)를 순차 수집",
    )
    parser.add_argument(
        "--gesture",
        action="append",
        help="특정 제스처를 수집. 여러 번 지정 가능",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="전처리 실행",
    )
    parser.add_argument("--train", action="store_true", help="학습 실행")
    parser.add_argument("--export", action="store_true", help="ONNX 변환 실행")
    parser.add_argument("--evaluate", action="store_true", help="평가 실행")
    parser.add_argument(
        "--full-basic",
        action="store_true",
        help="한 손 기본 전체 파이프라인 실행",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="수집할 클래스당 프레임 수 (기본: 500)",
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
        help="감지할 손 개수. 미지정 시 수집 스크립트 기본값 사용",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="자동 캡처 간격 밀리초 (기본: 30ms)",
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw",
        help="원시 데이터 디렉터리 (기본: data/raw)",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/processed",
        help="전처리 데이터 디렉터리 (기본: data/processed)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        help="모델 저장 디렉터리 (기본: models)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="학습 에포크 수 (기본: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="학습 배치 크기 (기본: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (기본: 0.001)",
    )

    args = parser.parse_args()
    if _has_selected_action(args):
        _run_selected_actions(args)
    else:
        _run_menu(args)


if __name__ == "__main__":
    main()
