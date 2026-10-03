"""Plot Fig. 4 from neural-estimator results produced by run_fig4.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

from ncl_sensing.plot_style import apply_paper_style, save_figure


BLUE = "#1f77b4"
RED = "#e45756"


def make_figure(data: dict[str, np.ndarray]):
    apply_paper_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.4, 4.75), gridspec_kw={"hspace": 0.24}
    )

    grid = data["2x2_grid"]
    colors = ["#440154", "#31688e", "#35b779", "#fde725"]
    checkpoints = [
        checkpoint
        for checkpoint in (10, 100, 500, 1000)
        if f"2x2_posterior_m{checkpoint}" in data
    ]
    for color, checkpoint in zip(colors, checkpoints):
        top.plot(
            grid,
            data[f"2x2_posterior_m{checkpoint}"],
            color=color,
            label=fr"$M={checkpoint}$",
        )
    top.axvline(
        float(data["2x2_ground_truth"]), color=RED, linestyle="--", linewidth=1.1
    )
    top.set_xlim(float(grid[0]), float(grid[-1]))
    top.set_ylim(bottom=0)
    top.set_xlabel(r"Detuning $\Delta$", labelpad=1)
    top.set_ylabel(r"$P(\Delta\mid\mathbf{s}^{(M)})$")
    top.legend(loc="upper right")
    top.text(-0.17, 1.02, "(a)", transform=top.transAxes, fontsize=13, va="top")

    steps = np.arange(1, int(data["shots_per_run"]) + 1)
    bottom.loglog(steps, data["3x3_mse_pattern"], color=BLUE, linewidth=1.7, label="Pattern")
    bottom.loglog(steps, data["3x3_mse_nr"], color=RED, linewidth=1.5, label=r"$N_r$")
    bottom.loglog(steps, data["3x3_crb"], color="black", linestyle="--", linewidth=1.1, label="CRB")
    bottom.set_xlim(10, steps[-1])
    bottom.set_ylim(1e-6, 3e-4)
    bottom.set_xlabel(r"Measurements $M$", labelpad=1)
    bottom.set_ylabel("MSE")
    bottom.legend(loc="upper right")
    bottom.text(-0.17, 1.02, "(b)", transform=bottom.transAxes, fontsize=13, va="top")

    inset = bottom.inset_axes([0.09, 0.08, 0.40, 0.42])
    inset.loglog(steps, data["2x2_mse_pattern"], color=BLUE, linewidth=1.3)
    inset.loglog(steps, data["2x2_mse_nr"], color=RED, linewidth=1.2)
    inset.loglog(steps, data["2x2_crb"], color="black", linestyle="--", linewidth=1.0)
    inset.set_xlim(10, steps[-1])
    inset.set_ylim(2e-6, 1e-3)
    inset.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    inset.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3]))
    inset.xaxis.set_minor_locator(NullLocator())
    inset.yaxis.set_minor_locator(NullLocator())
    inset.xaxis.set_major_formatter(LogFormatterMathtext())
    inset.yaxis.set_major_formatter(LogFormatterMathtext())
    inset.tick_params(labelsize=7, pad=1)
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data is None:
        args.data = root / "outputs" / "fig4" / f"fig4_nn_results_{args.profile}.npz"
    if args.output is None:
        args.output = root / "outputs" / f"Fig4_Sensing_CRB_{args.profile}"
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    figure = make_figure(data)
    save_figure(figure, args.output)
    plt.close(figure)
    print(f"Saved {args.output}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
