"""Train the held-out-pattern neural estimator used in Fig. 6(a)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ncl_sensing.inference import network_log_posteriors
from ncl_sensing.models import NetworkConfig, PosteriorNetwork
from ncl_sensing.training import TrainingConfig, seed_everything, train_network


PROFILES = {
    "quick": {
        "hidden_size": 256,
        "hidden_blocks": 2,
        "batch_norm": True,
        "learning_rate": 5e-4,
        "max_epochs": 2000,
        "patience": 500,
        "target_kl": 1e-4,
    },
    "paper": {
        "hidden_size": 1024,
        "hidden_blocks": 5,
        "batch_norm": True,
        "learning_rate": 1e-4,
        "max_epochs": 5000,
        "patience": 200,
        "target_kl": 0.0,
    },
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--pattern", type=int, default=275)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--training-mode",
        choices=("pattern-kl", "snapshot-ce"),
        default="pattern-kl",
        help=(
            "Use the original equal-pattern KL objective or the algebraically "
            "grouped form of snapshot-level cross entropy."
        ),
    )
    parser.add_argument("--smoothing-sigma", type=float, default=10.0)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=root / "data" / "processed" / "fig6_calibration.npz",
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=root / "data" / "processed" / "fig6_probabilities.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "fig6a"
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.calibration) as calibration:
        patterns = calibration["x_train"].astype(np.float32)
        targets = calibration["y_train"].astype(np.float32)
        frequencies = calibration["pattern_frequencies"].astype(np.float64)
        grid = calibration["grid_points"].astype(np.float64)
    with np.load(args.probabilities) as physics:
        physical_probabilities = physics["probabilities"].astype(np.float64)

    if not 0 <= args.pattern < patterns.shape[0]:
        raise ValueError("held-out pattern index is outside the pattern space")
    keep = np.ones(patterns.shape[0], dtype=bool)
    keep[args.pattern] = False
    if args.training_mode == "snapshot-ce":
        n_bins = targets.shape[1]
        reconstructed = np.rint(
            targets.astype(np.float64) * (frequencies[:, None] + n_bins) - 1.0
        )
        counts = np.clip(reconstructed, 0.0, None)
        if not np.allclose(counts.sum(axis=1), frequencies):
            raise ValueError("could not reconstruct snapshot counts from calibration data")
        train_targets = counts / frequencies[:, None]
        train_weights = frequencies.astype(np.float32)
    else:
        train_targets = targets
        train_weights = np.ones(patterns.shape[0], dtype=np.float32)
    network_config = NetworkConfig(
        input_size=patterns.shape[1],
        output_size=targets.shape[1],
        hidden_size=profile["hidden_size"],
        hidden_blocks=profile["hidden_blocks"],
        batch_norm=profile["batch_norm"],
    )
    training_config = TrainingConfig(
        learning_rate=profile["learning_rate"],
        max_epochs=profile["max_epochs"],
        patience=profile["patience"],
        target_kl=profile["target_kl"],
        seed=args.seed,
        device="auto",
        log_every=500,
    )
    seed_everything(args.seed)
    model = PosteriorNetwork(network_config)
    result = train_network(
        model,
        patterns[keep],
        train_targets[keep],
        train_weights[keep],
        training_config,
    )

    raw = np.exp(network_log_posteriors(model, patterns.shape[1])[args.pattern])
    raw /= raw.sum()
    true = physical_probabilities[:, args.pattern]
    true /= true.sum()
    uniform = np.full(grid.size, 1.0 / grid.size)
    smoothed = gaussian_filter1d(raw, sigma=args.smoothing_sigma)
    smoothed /= smoothed.sum()
    mse_uniform = float(np.mean((uniform - true) ** 2))
    mse_raw = float(np.mean((raw - true) ** 2))
    mse_smoothed = float(np.mean((smoothed - true) ** 2))
    pbits = format(args.pattern, f"0{patterns.shape[1]}b")
    kl_div = float(
        np.sum(true * np.log(np.clip(true / uniform, np.finfo(float).tiny, None)))
    )

    mode_suffix = "" if args.training_mode == "pattern-kl" else "_snapshot-ce"
    stem = f"fig6a_pattern{args.pattern}_{args.profile}{mode_suffix}"
    result_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        result_path,
        pidx=np.asarray(args.pattern),
        pbits=np.asarray(pbits),
        kl_div=np.asarray(kl_div),
        grid=grid,
        true_post=true,
        raw_post=raw,
        smoothed_post=smoothed,
        uniform=uniform,
        mse_uniform=np.asarray(mse_uniform),
        mse_raw=np.asarray(mse_raw),
        mse_smooth=np.asarray(mse_smoothed),
        impr_raw=np.asarray(mse_uniform / mse_raw),
        impr_smooth=np.asarray(mse_uniform / mse_smoothed),
    )
    metadata = {
        "pattern": args.pattern,
        "pattern_bits": pbits,
        "training_mode": args.training_mode,
        "network_config": asdict(network_config),
        "training_config": asdict(training_config),
        "training_result": {
            "best_loss": result.best_loss,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "converged": result.converged,
        },
        "mse_uniform": mse_uniform,
        "mse_raw": mse_raw,
        "mse_smoothed": mse_smoothed,
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    checkpoint_path = args.output_dir / "checkpoints" / args.profile / f"{stem}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(metadata), checkpoint_path)
    print(f"Saved {result_path}")
    print(f"Saved {metadata_path}")
    print(f"Raw posterior MSE: {mse_raw:.6e}")


if __name__ == "__main__":
    main()
