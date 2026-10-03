"""Regenerate Fig. 2 from the compact scaling summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from ncl_sensing.plot_style import apply_paper_style, save_figure


BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#aaaaaa"


def configure_x_axis(axis, show_label: bool) -> None:
    axis.set_xscale("log")
    axis.set_xlim(0.9, 13.5)
    axis.xaxis.set_major_locator(FixedLocator([1, 2, 4, 6, 8, 10, 12]))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel(r"$N$" if show_label else "")


def plot_points(axis, n, values, is_1d, color, labels=True) -> None:
    one_d = is_1d
    two_d = ~is_1d
    axis.plot(
        n[one_d],
        values[one_d],
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.4,
        markersize=5.4,
    )
    axis.plot(
        n[two_d],
        values[two_d],
        linestyle="none",
        marker="*",
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.2,
    )
    if labels:
        n_rows = CURRENT_DATA["n_row"]
        n_cols = CURRENT_DATA["n_col"]
        for index in np.flatnonzero(two_d):
            geometry = f"{int(n_rows[index])}x{int(n_cols[index])}"
            vertical = 1.25
            if geometry in {"3x3", "3x4"}:
                vertical = 0.72
            axis.text(
                n[index],
                values[index] * vertical,
                geometry,
                color=color,
                fontsize=7.5,
                ha="center",
                va="bottom" if vertical > 1 else "top",
            )


def reference_lines(axis, scale, include_exponent=False, labels=True) -> None:
    n_line = np.linspace(1.0, 13.0, 300)
    axis.plot(
        n_line,
        scale * n_line,
        color=GRAY,
        linestyle="--",
        linewidth=1.0,
        label=r"SQL ($\propto N$)" if labels else None,
    )
    axis.plot(
        n_line,
        scale * n_line**2,
        color=GRAY,
        linestyle="-.",
        linewidth=1.0,
        label=r"HL ($\propto N^2$)" if labels else None,
    )
    if include_exponent:
        axis.plot(
            n_line,
            scale * n_line**1.7,
            color=GRAY,
            linestyle=":",
            linewidth=1.1,
            label=r"$\propto N^{1.7}$" if labels else None,
        )


def marker_legend(axis, color, qfi: bool) -> None:
    axis.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.4,
        label="1D Chain" if qfi else r"CFI$_{pat}$ (1D)",
    )
    axis.plot(
        [],
        [],
        linestyle="none",
        marker="*",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=9,
        label="2D Array" if qfi else r"CFI$_{pat}$ (2D)",
    )


CURRENT_DATA = None


def make_figure(data: dict[str, np.ndarray]):
    global CURRENT_DATA
    CURRENT_DATA = data
    apply_paper_style()
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(3.4, 5.0),
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )

    n = data["n_atoms"].astype(float)
    is_1d = (data["n_row"] == 1) | (data["n_col"] == 1)

    reference_lines(top, 24.0)
    plot_points(top, n, data["qfi_t10"], is_1d, BLUE)
    marker_legend(top, BLUE, qfi=True)
    top.set_yscale("log")
    top.set_ylabel("QFI")
    top.set_ylim(20, 6.0e3)
    top.legend(loc="lower right", handlelength=2.2, handletextpad=0.5)
    configure_x_axis(top, show_label=False)
    top.text(-0.17, 0.98, "(a)", transform=top.transAxes, fontsize=13, va="top")

    inset = top.inset_axes([0.095, 0.58, 0.39, 0.38])
    reference_lines(inset, 2300.0, labels=False)
    plot_points(inset, n, data["qfi_t100"], is_1d, BLUE, labels=False)
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlim(0.9, 13.5)
    inset.set_ylim(1.8e3, 5.0e5)
    inset.xaxis.set_major_locator(FixedLocator([1, 4, 8, 12]))
    inset.xaxis.set_minor_locator(NullLocator())
    inset.xaxis.set_major_formatter(ScalarFormatter())
    inset.tick_params(labelsize=7, pad=1)

    reference_lines(bottom, 2350.0, include_exponent=True)
    plot_points(bottom, n, data["cfipattern_t100"], is_1d, RED)
    marker_legend(bottom, RED, qfi=False)
    bottom.set_yscale("log")
    bottom.set_ylim(1.8e3, 4.2e5)
    bottom.set_ylabel("CFI")
    configure_x_axis(bottom, show_label=True)
    bottom.legend(loc="upper left", handlelength=2.2, handletextpad=0.5)
    bottom.text(-0.17, 0.98, "(b)", transform=bottom.transAxes, fontsize=13, va="top")
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "processed" / "fig2_scaling.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs" / "Fig2_QFI_CFI"
    )
    args = parser.parse_args()
    with np.load(args.data) as archive:
        data = {key: archive[key] for key in archive.files if key != "source_files_json"}
    figure = make_figure(data)
    save_figure(figure, args.output)
    plt.close(figure)
    print(f"Saved {args.output}.pdf/.svg/.png")


if __name__ == "__main__":
    main()

