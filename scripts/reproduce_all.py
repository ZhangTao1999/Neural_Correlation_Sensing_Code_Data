"""Unified entry point for regenerating the six manuscript figures."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(root: Path, *arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def copy_schematics(root: Path, output_dir: Path) -> None:
    source_dir = root / "assets" / "schematics"
    for name in ("Fig1_Illustration.pdf", "Fig3_Two_Stage_Sensing.pdf"):
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)


def quick_retrain(root: Path) -> tuple[Path, Path, Path]:
    run(
        root,
        "scripts/run_fig4.py",
        "--profile",
        "quick",
        "--systems",
        "2x2",
        "3x3",
        "--data",
        "data/processed/fig4_calibration.npz",
    )
    run(root, "scripts/run_fig5.py", "--profile", "quick")
    run(root, "scripts/run_fig6a.py", "--profile", "quick")
    run(root, "scripts/run_fig6b.py", "--profile", "quick")
    fig6b = root / "outputs" / "fig6b" / "fig6b_statistics_quick.npz"
    run(
        root,
        "scripts/merge_fig6b.py",
        "--pattern",
        "fig6b_quick_start*_n*.npz",
        "--output",
        str(fig6b),
    )
    return (
        root / "outputs" / "fig4" / "fig4_nn_results_quick.npz",
        root / "outputs" / "fig5" / "fig5_nn_results_quick.npz",
        fig6b,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("paper", "quick"),
        default="paper",
        help="Use frozen paper-scale neural results or retrain a small smoke test.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "paper"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_schematics(root, output_dir)

    if args.mode == "quick":
        fig4_data, fig5_data, fig6b_data = quick_retrain(root)
        fig6a_data = root / "outputs" / "fig6a" / "fig6a_pattern275_quick.npz"
    else:
        result_dir = root / "data" / "results"
        fig4_data = result_dir / "fig4_nn_results_paper.npz"
        fig5_data = result_dir / "fig5_nn_results_paper.npz"
        fig6a_data = root / "data" / "processed" / "fig6a_pattern275.npz"
        fig6b_data = root / "data" / "processed" / "fig6b_statistics.npz"
        for path in (fig4_data, fig5_data, fig6a_data, fig6b_data):
            if not path.exists():
                raise FileNotFoundError(
                    f"required frozen result is missing: {path}. "
                    "Run the corresponding training script first."
                )

    run(
        root,
        "scripts/plot_fig2.py",
        "--output",
        str(output_dir / "Fig2_QFI_CFI"),
    )
    run(
        root,
        "scripts/plot_fig4.py",
        "--data",
        str(fig4_data),
        "--output",
        str(output_dir / "Fig4_Sensing_CRB"),
    )
    run(
        root,
        "scripts/plot_fig5.py",
        "--data",
        str(fig5_data),
        "--output",
        str(output_dir / "Fig5_Robustness_gamma"),
    )
    run(
        root,
        "scripts/plot_fig6.py",
        "--single",
        str(fig6a_data),
        "--aggregate",
        str(fig6b_data),
        "--output",
        str(output_dir / "Fig6_Robustness_Mask"),
    )
    print(f"All figures are available in {output_dir}")


if __name__ == "__main__":
    main()
