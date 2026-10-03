"""Regenerate the exact coherent probability table used by Fig. 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ncl_sensing.physics import RydbergArray, RydbergConfig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "fig6_probabilities.npz",
    )
    args = parser.parse_args()

    grid = np.linspace(0.15, 0.25, 101)
    simulator = RydbergArray(RydbergConfig(3, 3, spacing=0.9, omega=1.0))
    probabilities = np.empty((grid.size, simulator.dimension), dtype=np.float64)
    for index, shift in enumerate(grid):
        probabilities[index] = simulator.probabilities(
            detuning=2.0 + shift,
            time=10.0,
        )
        if (index + 1) % 10 == 0 or index + 1 == grid.size:
            print(f"computed {index + 1}/{grid.size} parameter points")

    gt_index = int(np.argmin(np.abs(grid - 0.2)))
    cfi = simulator.cfi(detuning=2.0 + grid[gt_index], time=10.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        grid_points=grid,
        probabilities=probabilities,
        ground_truth=np.asarray(grid[gt_index]),
        ground_truth_index=np.asarray(gt_index),
        ground_truth_cfi=np.asarray(cfi),
        n_rows=np.asarray(3),
        n_cols=np.asarray(3),
        spacing=np.asarray(0.9),
        omega=np.asarray(1.0),
        base_detuning=np.asarray(2.0),
        evolution_time=np.asarray(10.0),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
