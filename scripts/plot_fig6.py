"""Regenerate Fig. 6 from compact neural-network results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ncl_sensing.plot_style import apply_paper_style, save_figure


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def draw_pattern_inset(axis, bits: str) -> None:
    inset = axis.inset_axes([0.06, 0.02, 0.20, 0.30])
    inset.set_aspect("equal")
    pattern = np.asarray([int(bit) for bit in bits]).reshape(3, 3)
    for row in range(3):
        for column in range(3):
            circle = plt.Circle(
                (column, 2 - row),
                0.31,
                facecolor="black" if pattern[row, column] else "white",
                edgecolor="black",
                linewidth=0.9,
            )
            inset.add_patch(circle)
    inset.set_xlim(-0.5, 2.5)
    inset.set_ylim(-0.5, 2.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)


def make_figure(single: dict[str, np.ndarray], aggregate: dict[str, np.ndarray]):
    apply_paper_style()
    figure = plt.figure(figsize=(3.4, 4.55))
    layout = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.50)

    top = figure.add_subplot(layout[0])
    grid = single["grid"]
    top.plot(grid, single["uniform"], color="#888888", linestyle=":", linewidth=1.1, label="Uniform")
    top.plot(grid, single["raw_post"], color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.75, label="NN (raw)")
    top.plot(grid, single["smoothed_post"], color=BLUE, linewidth=1.5, label="NN (smoothed)")
    top.plot(grid, single["true_post"], color="black", linewidth=1.7, label="Ground truth")
    top.set_xlim(float(grid[0]), float(grid[-1]))
    top.set_xlabel(r"Detuning $\Delta$", labelpad=1)
    top.set_ylabel("Posterior probability")
    top.set_yticks([0.01, 0.02])
    top.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.18),
        mode="expand",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.7,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    draw_pattern_inset(top, str(single["pbits"].item()))
    top.text(-0.17, 1.29, "(a)", transform=top.transAxes, fontsize=13, va="top")

    bottom = figure.add_subplot(layout[1])
    ratios = aggregate["x_ratios"] * 100.0
    mse = bottom.errorbar(
        ratios,
        aggregate["y_mse_means"],
        yerr=aggregate["y_mse_sems"],
        color=BLUE,
        marker="o",
        markersize=3.5,
        capsize=2,
        label="MSE",
    )
    bottom.set_yscale("log")
    bottom.set_ylim(5.5e-6, 1.1e-4)
    bottom.set_xlabel("Masked Patterns (%)", labelpad=1)
    bottom.set_ylabel("MSE", color=BLUE, labelpad=2)
    bottom.tick_params(axis="y", colors=BLUE)

    right = bottom.twinx()
    bias = right.errorbar(
        ratios,
        aggregate["y_bias_means"],
        yerr=aggregate["y_bias_sems"],
        color=ORANGE,
        linestyle="--",
        marker="s",
        markersize=3.5,
        capsize=2,
        label="Bias",
    )
    right.axhline(0.0, color="#999999", linestyle=":", linewidth=1.0)
    right.set_ylim(-0.012, 0.012)
    right.set_yticks([-0.01, 0.0, 0.01])
    right.set_ylabel("Bias", color=ORANGE, labelpad=2)
    right.tick_params(axis="y", colors=ORANGE)

    upper = bottom.twiny()
    upper.set_xlim(bottom.get_xlim())
    upper.set_xticks(ratios)
    upper.set_xticklabels([f"{100 * value:.1f}" for value in aggregate["x_mass_means"]])
    upper.set_xlabel("Masked Samples (%)", labelpad=2)
    upper.tick_params(axis="x", labelsize=8, pad=2)

    bottom.legend(
        [mse, bias],
        ["MSE", "Bias"],
        loc="upper left",
        handlelength=2.0,
        handletextpad=0.5,
    )
    bottom.text(-0.17, 1.22, "(b)", transform=bottom.transAxes, fontsize=13, va="top")
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        type=Path,
        default=root / "data" / "processed" / "fig6a_pattern275.npz",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=root / "data" / "processed" / "fig6b_statistics.npz",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs" / "Fig6_Robustness_Mask"
    )
    args = parser.parse_args()
    with np.load(args.single, allow_pickle=False) as archive:
        single = {key: archive[key] for key in archive.files}
    with np.load(args.aggregate, allow_pickle=False) as archive:
        aggregate = {key: archive[key] for key in archive.files}
    figure = make_figure(single, aggregate)
    save_figure(figure, args.output)
    plt.close(figure)
    print(f"Saved {args.output}.pdf/.svg/.png")


if __name__ == "__main__":
    main()

