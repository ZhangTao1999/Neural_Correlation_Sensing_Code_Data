"""Merge independently generated Fig. 6(b) realization shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=root / "outputs" / "fig6b" / "parts"
    )
    parser.add_argument("--pattern", default="fig6b_paper_start*_n*.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "fig6b" / "fig6b_statistics_network.npz",
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no shards matched {args.input_dir / args.pattern}")

    ratios = None
    arrays = {key: [] for key in ("raw_mses", "raw_biases", "raw_mass_lost")}
    seeds = []
    for path in paths:
        with np.load(path) as shard:
            if ratios is None:
                ratios = shard["x_ratios"]
            else:
                np.testing.assert_allclose(shard["x_ratios"], ratios)
            for key in arrays:
                arrays[key].append(shard[key])
            seeds.append(shard["realization_seeds"])
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    realization_seeds = np.concatenate(seeds)
    if np.unique(realization_seeds).size != realization_seeds.size:
        raise ValueError("duplicate realization seeds detected across shards")
    mse_mean, mse_sem = mean_and_sem(merged["raw_mses"])
    bias_mean, bias_sem = mean_and_sem(merged["raw_biases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_ratios=ratios,
        y_mse_means=mse_mean,
        y_mse_sems=mse_sem,
        y_bias_means=bias_mean,
        y_bias_sems=bias_sem,
        x_mass_means=merged["raw_mass_lost"].mean(axis=0),
        raw_mses=merged["raw_mses"],
        raw_biases=merged["raw_biases"],
        raw_mass_lost=merged["raw_mass_lost"],
        realization_seeds=realization_seeds,
    )
    print(f"Merged {len(paths)} shards and {realization_seeds.size} realizations")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
