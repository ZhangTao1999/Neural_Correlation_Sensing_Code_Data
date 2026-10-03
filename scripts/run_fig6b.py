"""Run neural generalization tests with low-frequency patterns masked."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ncl_sensing.calibration import grouped_calibration, sample_counts
from ncl_sensing.inference import network_log_posteriors
from ncl_sensing.models import NetworkConfig, PosteriorNetwork
from ncl_sensing.patterns import all_bit_patterns
from ncl_sensing.training import TrainingConfig, seed_everything, train_network


PROFILES = {
    "quick": {
        "ratios": [0.0, 0.45, 0.9],
        "repeats": 2,
        "calibration_shots": 10000,
        "sensing_shots": 1000,
        "hidden_size": 128,
        "hidden_blocks": 1,
        "batch_norm": True,
        "learning_rate": 5e-4,
        "max_epochs": 1500,
        "patience": 500,
        "target_kl": 1e-2,
        "require_convergence": True,
    },
    "paper": {
        "ratios": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
        "repeats": 1000,
        "calibration_shots": 10000,
        "sensing_shots": 1000,
        "hidden_size": 768,
        "hidden_blocks": 3,
        "batch_norm": True,
        "learning_rate": 1e-4,
        "max_epochs": 8000,
        "patience": 2000,
        "target_kl": 1e-6,
        "require_convergence": True,
    },
}


def posterior_estimate(log_map: np.ndarray, shots: np.ndarray, grid: np.ndarray) -> float:
    accumulated = log_map[shots].sum(axis=0)
    posterior = np.exp(accumulated - accumulated.max())
    posterior /= posterior.sum()
    return float(posterior @ grid)


def mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--training-mode",
        choices=("pattern-kl", "snapshot-ce"),
        default="pattern-kl",
        help=(
            "Use the original equal-pattern KL objective or the algebraically "
            "grouped form of snapshot-level cross entropy."
        ),
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=root / "data" / "processed" / "fig6_probabilities.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "fig6b" / "parts"
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    requested_repeats = args.repeats or profile["repeats"]
    ratios = np.asarray(profile["ratios"], dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.probabilities) as archive:
        grid = archive["grid_points"].astype(np.float64)
        probabilities = archive["probabilities"].astype(np.float64)
        gt = float(archive["ground_truth"])
        gt_index = int(archive["ground_truth_index"])
    n_bins, n_patterns = probabilities.shape
    n_qubits = int(np.log2(n_patterns))
    patterns = all_bit_patterns(n_qubits)
    network_config = NetworkConfig(
        input_size=n_qubits,
        output_size=n_bins,
        hidden_size=profile["hidden_size"],
        hidden_blocks=profile["hidden_blocks"],
        batch_norm=profile["batch_norm"],
    )

    raw_mses = []
    raw_biases = []
    raw_masses = []
    raw_losses = []
    realization_seeds = []
    attempt = 0
    max_attempts = requested_repeats * 3
    while len(raw_mses) < requested_repeats and attempt < max_attempts:
        global_index = args.start + len(raw_mses)
        realization_seed = args.seed + global_index * 1009 + attempt
        attempt += 1
        counts = sample_counts(
            probabilities,
            profile["calibration_shots"],
            realization_seed,
        )
        pseudocount = 1.0 if args.training_mode == "pattern-kl" else 0.0
        calibration = grouped_calibration(patterns, counts, pseudocount=pseudocount)
        if args.training_mode == "snapshot-ce":
            training_weights = calibration.weights
        else:
            training_weights = np.ones(n_patterns, dtype=np.float32)
        frequencies = counts.sum(axis=1)
        ranked = np.argsort(frequencies, kind="stable")
        rng = np.random.default_rng(realization_seed + 500000)
        sensing = rng.choice(
            n_patterns,
            size=profile["sensing_shots"],
            p=probabilities[gt_index],
        )

        realization_mses = []
        realization_biases = []
        realization_masses = []
        realization_losses = []
        valid = True
        for ratio_index, ratio in enumerate(ratios):
            n_masked = int(n_patterns * ratio)
            keep = np.ones(n_patterns, dtype=bool)
            if n_masked:
                keep[ranked[:n_masked]] = False
            training_config = TrainingConfig(
                learning_rate=profile["learning_rate"],
                max_epochs=profile["max_epochs"],
                patience=profile["patience"],
                target_kl=profile["target_kl"],
                seed=realization_seed + ratio_index,
                device="auto",
                log_every=0,
            )
            seed_everything(training_config.seed)
            model = PosteriorNetwork(network_config)
            result = train_network(
                model,
                calibration.patterns[keep],
                calibration.targets[keep],
                training_weights[keep],
                training_config,
            )
            if profile["require_convergence"] and not result.converged:
                valid = False
                print(
                    f"realization={global_index} ratio={ratio:g} failed "
                    f"with KL={result.best_loss:.3e}; retrying"
                )
                break
            log_map = network_log_posteriors(model, n_qubits)
            estimate = posterior_estimate(log_map, sensing, grid)
            bias = estimate - gt
            realization_biases.append(bias)
            realization_mses.append(bias**2)
            realization_masses.append(
                float(frequencies[ranked[:n_masked]].sum() / frequencies.sum())
                if n_masked
                else 0.0
            )
            realization_losses.append(result.best_loss)
            print(
                f"realization={global_index} ratio={ratio:g} "
                f"KL={result.best_loss:.3e} bias={bias:.3e}"
            )

        if not valid:
            continue
        raw_mses.append(realization_mses)
        raw_biases.append(realization_biases)
        raw_masses.append(realization_masses)
        raw_losses.append(realization_losses)
        realization_seeds.append(realization_seed)

    if len(raw_mses) != requested_repeats:
        raise RuntimeError(
            f"obtained {len(raw_mses)} valid realizations out of "
            f"{requested_repeats} requested"
        )

    raw_mses = np.asarray(raw_mses)
    raw_biases = np.asarray(raw_biases)
    raw_masses = np.asarray(raw_masses)
    raw_losses = np.asarray(raw_losses)
    mse_mean, mse_sem = mean_and_sem(raw_mses)
    bias_mean, bias_sem = mean_and_sem(raw_biases)
    mode_suffix = "" if args.training_mode == "pattern-kl" else "_snapshot-ce"
    stem = (
        f"fig6b_{args.profile}_start{args.start:04d}_n{requested_repeats:04d}"
        f"{mode_suffix}"
    )
    output_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        output_path,
        x_ratios=ratios,
        y_mse_means=mse_mean,
        y_mse_sems=mse_sem,
        y_bias_means=bias_mean,
        y_bias_sems=bias_sem,
        x_mass_means=raw_masses.mean(axis=0),
        raw_mses=raw_mses,
        raw_biases=raw_biases,
        raw_mass_lost=raw_masses,
        raw_training_losses=raw_losses,
        realization_seeds=np.asarray(realization_seeds),
    )
    metadata = {
        "profile": args.profile,
        "training_mode": args.training_mode,
        "start": args.start,
        "repeats": requested_repeats,
        "attempts": attempt,
        "ratios": ratios.tolist(),
        "network_config": asdict(network_config),
        "training": {
            key: profile[key]
            for key in (
                "calibration_shots",
                "sensing_shots",
                "learning_rate",
                "max_epochs",
                "patience",
                "target_kl",
            )
        },
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
