"""Regenerate the Lindblad probability tables used by Fig. 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ncl_sensing.physics import LindbladRydbergArray, RydbergConfig


PAPER_GAMMAS = np.asarray(
    [0.0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--gamma", type=float, nargs="*")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "fig5" / "fig5_probabilities_regenerated.npz",
    )
    args = parser.parse_args()
    gamma_values = (
        np.asarray(args.gamma, dtype=np.float64)
        if args.gamma
        else (np.asarray([0.0, 0.01, 0.1]) if args.profile == "quick" else PAPER_GAMMAS)
    )
    grid = (
        np.linspace(0.0, 0.1, 11)
        if args.profile == "quick"
        else np.linspace(0.0, 0.1, 101)
    )
    simulator = LindbladRydbergArray(RydbergConfig(2, 2, 0.9, 1.0))
    probabilities = np.empty(
        (gamma_values.size, grid.size, simulator.dimension), dtype=np.float64
    )
    for gamma_index, gamma in enumerate(gamma_values):
        for grid_index, shift in enumerate(grid):
            probabilities[gamma_index, grid_index] = (
                simulator.probabilities_with_decay(
                    detuning=2.0 + shift,
                    time=10.0,
                    gamma=float(gamma),
                )
            )
        print(
            f"computed gamma={gamma:g} ({gamma_index + 1}/{gamma_values.size})",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gamma_values=gamma_values,
        grid_points=grid,
        probabilities=probabilities,
        ground_truth=np.asarray(0.05),
        n_rows=np.asarray(2),
        n_cols=np.asarray(2),
        spacing=np.asarray(0.9),
        omega=np.asarray(1.0),
        base_detuning=np.asarray(2.0),
        evolution_time=np.asarray(10.0),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
