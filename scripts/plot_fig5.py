"""Plot Fig. 5 from neural-estimator robustness results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ncl_sensing.plot_style import apply_paper_style, save_figure


BLUE = "#1f77b4"
RED = "#d62728"


def make_figure(data: dict[str, np.ndarray]):
    apply_paper_style()
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.4, 4.45), gridspec_kw={"hspace": 0.30}
    )
    measurements = data["measurements"]

    top.loglog(
        measurements,
        data["trace_ideal_mean"],
        color=RED,
        linestyle="--",
        label="Ideal Model",
    )
    top.fill_between(
        measurements,
        np.maximum(data["trace_ideal_mean"] - data["trace_ideal_sem"], 1e-12),
        data["trace_ideal_mean"] + data["trace_ideal_sem"],
        color=RED,
        alpha=0.18,
        linewidth=0,
    )
    top.loglog(
        measurements,
        data["trace_matched_mean"],
        color=BLUE,
        label="Noise-Aware",
    )
    top.fill_between(
        measurements,
        np.maximum(data["trace_matched_mean"] - data["trace_matched_sem"], 1e-12),
        data["trace_matched_mean"] + data["trace_matched_sem"],
        color=BLUE,
        alpha=0.18,
        linewidth=0,
    )
    top.set_xlim(max(5, int(measurements[0])), int(measurements[-1]))
    top.set_xlabel(r"Measurements $M$", labelpad=1)
    top.set_ylabel("MSE")
    top.legend(loc="upper right")
    top.text(
        0.05,
        0.06,
        fr"$\gamma={float(data['target_gamma']):g}$",
        transform=top.transAxes,
    )
    top.text(-0.17, 1.02, "(a)", transform=top.transAxes, fontsize=13, va="top")

    gamma = data["gamma_values"]
    bottom.errorbar(
        gamma,
        data["final_ideal_mean"],
        yerr=data["final_ideal_sem"],
        color=RED,
        linestyle="--",
        marker="o",
        markersize=3.5,
        capsize=2,
        label="Ideal Model",
    )
    bottom.errorbar(
        gamma,
        data["final_matched_mean"],
        yerr=data["final_matched_sem"],
        color=BLUE,
        marker="o",
        markersize=3.5,
        capsize=2,
        label="Noise-Aware",
    )
    bottom.set_xscale("log")
    bottom.set_yscale("log")
    bottom.set_xlabel(r"$\gamma$", labelpad=1)
    bottom.set_ylabel("MSE")
    bottom.legend(loc="upper left")
    bottom.text(-0.17, 1.02, "(b)", transform=bottom.transAxes, fontsize=13, va="top")
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data is None:
        args.data = root / "outputs" / "fig5" / f"fig5_nn_results_{args.profile}.npz"
    if args.output is None:
        args.output = root / "outputs" / f"Fig5_Robustness_gamma_{args.profile}"
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    figure = make_figure(data)
    save_figure(figure, args.output)
    plt.close(figure)
    print(f"Saved {args.output}.pdf/.svg/.png")


if __name__ == "__main__":
    main()

