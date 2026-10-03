"""Train ideal and noise-aware neural estimators for Fig. 5."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

from ncl_sensing.calibration import grouped_calibration, sample_counts
from ncl_sensing.data import load_fig5_probabilities
from ncl_sensing.inference import network_log_posteriors
from ncl_sensing.models import NetworkConfig, PosteriorNetwork
from ncl_sensing.patterns import all_bit_patterns
from ncl_sensing.training import TrainingConfig, seed_everything, train_network
from ncl_sensing.validation import posterior_agreement


PROFILES = {
    "quick": {
        "gamma_values": [0.0, 0.01, 0.1],
        "calibration_shots": 5000,
        "repeats": 25,
        "shots": 500,
        "final_window": 50,
        "hidden_size": 64,
        "hidden_blocks": 1,
        "max_epochs": 1200,
        "target_kl": 2e-5,
    },
    "paper": {
        "gamma_values": None,
        "calibration_shots": 1000000,
        "repeats": 500,
        "shots": 5000,
        "final_window": 100,
        "hidden_size": 128,
        "hidden_blocks": 2,
        "max_epochs": 5000,
        "target_kl": 1e-7,
    },
}


def train_estimator(
    probabilities: np.ndarray,
    calibration_shots: int,
    seed: int,
    profile: dict,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_bins, n_patterns = probabilities.shape
    n_qubits = int(np.log2(n_patterns))
    counts = sample_counts(probabilities, calibration_shots, seed)
    calibration = grouped_calibration(
        all_bit_patterns(n_qubits), counts, pseudocount=0.0
    )
    network_config = NetworkConfig(
        input_size=n_qubits,
        output_size=n_bins,
        hidden_size=profile["hidden_size"],
        hidden_blocks=profile["hidden_blocks"],
        batch_norm=False,
    )
    training_config = TrainingConfig(
        learning_rate=1e-3,
        max_epochs=profile["max_epochs"],
        patience=max(500, profile["max_epochs"] // 5),
        target_kl=profile["target_kl"],
        seed=seed,
        device="auto",
        log_every=0,
    )
    seed_everything(seed)
    model = PosteriorNetwork(network_config)
    result = train_network(
        model,
        calibration.patterns,
        calibration.targets,
        calibration.weights,
        training_config,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    network_logs = network_log_posteriors(model, n_qubits)
    empirical_targets = calibration.targets.astype(np.float64)
    exact_targets = probabilities.T.astype(np.float64)
    exact_targets /= exact_targets.sum(axis=1, keepdims=True)
    physical_weights = probabilities.mean(axis=0)
    metadata = {
        "network_config": asdict(network_config),
        "training_config": asdict(training_config),
        "training_result": {
            "best_loss": result.best_loss,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "converged": result.converged,
        },
        "calibration_shots_per_bin": calibration_shots,
        "network_vs_empirical_calibration": posterior_agreement(
            empirical_targets, network_logs, calibration.weights
        ).to_dict(),
        "network_vs_exact_posterior": posterior_agreement(
            exact_targets, network_logs, physical_weights
        ).to_dict(),
        "empirical_vs_exact_posterior": posterior_agreement(
            exact_targets,
            np.log(np.clip(empirical_targets, np.finfo(float).tiny, None)),
            physical_weights,
        ).to_dict(),
    }
    torch.save(model.checkpoint(metadata), checkpoint_path)
    empirical_logs = np.log(
        np.clip(empirical_targets, np.finfo(float).tiny, None)
    )
    exact_logs = np.log(np.clip(exact_targets, np.finfo(float).tiny, None))
    return network_logs, empirical_logs, exact_logs, metadata


def normalize_log_rows(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def trace_errors(log_table: np.ndarray, shots: np.ndarray, grid: np.ndarray, gt: float) -> np.ndarray:
    cumulative = np.cumsum(log_table[shots], axis=0)
    posterior = normalize_log_rows(cumulative)
    estimates = posterior @ grid
    return (estimates - gt) ** 2


def final_window_error(
    log_table: np.ndarray,
    shots: np.ndarray,
    grid: np.ndarray,
    gt: float,
    window: int,
) -> float:
    prefix = shots[:-window]
    if prefix.size:
        counts = np.bincount(prefix, minlength=log_table.shape[0])
        base = counts @ log_table
    else:
        base = np.zeros(log_table.shape[1])
    cumulative = base + np.cumsum(log_table[shots[-window:]], axis=0)
    posterior = normalize_log_rows(cumulative)
    estimates = posterior @ grid
    return float(np.mean((estimates - gt) ** 2))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "processed" / "fig5_probabilities.h5"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "fig5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]

    grid, gt, available, all_tables = load_fig5_probabilities(args.data)
    gamma_values = available if profile["gamma_values"] is None else np.asarray(profile["gamma_values"], dtype=float)
    missing = [float(g) for g in gamma_values if float(g) not in all_tables]
    if missing:
        raise ValueError(f"Requested decay rates absent from input: {missing}; regenerate those rates or select the paper profile to use all available rates")
    tables = {float(g): all_tables[float(g)] for g in gamma_values}

    checkpoint_dir = args.output_dir / "checkpoints" / args.profile
    ideal_logs, ideal_table_logs, ideal_exact_logs, ideal_info = train_estimator(
        tables[0.0],
        profile["calibration_shots"],
        args.seed,
        profile,
        checkpoint_dir / "fig5_ideal_network.pt",
    )
    matched_logs = {0.0: ideal_logs}
    matched_table_logs = {0.0: ideal_table_logs}
    matched_exact_logs = {0.0: ideal_exact_logs}
    training_info = {"ideal": ideal_info, "matched": {}}
    for index, gamma in enumerate(gamma_values):
        gamma = float(gamma)
        if gamma == 0.0:
            training_info["matched"][str(gamma)] = ideal_info
            continue
        logs, table_logs, exact_logs, info = train_estimator(
            tables[gamma],
            profile["calibration_shots"],
            args.seed + index + 1,
            profile,
            checkpoint_dir / f"fig5_matched_gamma_{gamma:g}.pt",
        )
        matched_logs[gamma] = logs
        matched_table_logs[gamma] = table_logs
        matched_exact_logs[gamma] = exact_logs
        training_info["matched"][str(gamma)] = info
        print(
            f"gamma={gamma:g}: matched KL={info['training_result']['best_loss']:.3e}, "
            f"converged={info['training_result']['converged']}"
        )

    rng = np.random.default_rng(args.seed + 10000)
    target_gamma = 0.01
    trace_ideal = []
    trace_matched = []
    final_ideal = []
    final_matched = []
    final_ideal_table = []
    final_matched_table = []
    final_ideal_exact = []
    final_matched_exact = []
    plotted_gammas = []

    for gamma in gamma_values:
        gamma = float(gamma)
        if gamma == 0.0:
            continue
        probability_gt = tables[gamma][int(np.argmin(np.abs(grid - gt)))]
        errors_ideal = []
        errors_matched = []
        errors_ideal_table = []
        errors_matched_table = []
        errors_ideal_exact = []
        errors_matched_exact = []
        for _ in range(profile["repeats"]):
            shots = rng.choice(
                probability_gt.size,
                size=profile["shots"],
                replace=True,
                p=probability_gt,
            )
            errors_ideal.append(
                final_window_error(
                    ideal_logs,
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            errors_matched.append(
                final_window_error(
                    matched_logs[gamma],
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            errors_ideal_table.append(
                final_window_error(
                    ideal_table_logs,
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            errors_matched_table.append(
                final_window_error(
                    matched_table_logs[gamma],
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            errors_ideal_exact.append(
                final_window_error(
                    ideal_exact_logs,
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            errors_matched_exact.append(
                final_window_error(
                    matched_exact_logs[gamma],
                    shots,
                    grid,
                    gt,
                    profile["final_window"],
                )
            )
            if np.isclose(gamma, target_gamma):
                trace_ideal.append(trace_errors(ideal_logs, shots, grid, gt))
                trace_matched.append(trace_errors(matched_logs[gamma], shots, grid, gt))

        plotted_gammas.append(gamma)
        final_ideal.append(errors_ideal)
        final_matched.append(errors_matched)
        final_ideal_table.append(errors_ideal_table)
        final_matched_table.append(errors_matched_table)
        final_ideal_exact.append(errors_ideal_exact)
        final_matched_exact.append(errors_matched_exact)

    trace_ideal = np.asarray(trace_ideal)
    trace_matched = np.asarray(trace_matched)
    final_ideal = np.asarray(final_ideal)
    final_matched = np.asarray(final_matched)
    final_ideal_table = np.asarray(final_ideal_table)
    final_matched_table = np.asarray(final_matched_table)
    final_ideal_exact = np.asarray(final_ideal_exact)
    final_matched_exact = np.asarray(final_matched_exact)
    repeats = profile["repeats"]
    arrays = {
        "profile": np.asarray(args.profile),
        "grid": grid,
        "ground_truth": np.asarray(gt),
        "gamma_values": np.asarray(plotted_gammas),
        "target_gamma": np.asarray(target_gamma),
        "measurements": np.arange(1, profile["shots"] + 1),
        "trace_ideal_mean": trace_ideal.mean(axis=0),
        "trace_ideal_sem": trace_ideal.std(axis=0, ddof=1) / np.sqrt(repeats),
        "trace_matched_mean": trace_matched.mean(axis=0),
        "trace_matched_sem": trace_matched.std(axis=0, ddof=1) / np.sqrt(repeats),
        "final_ideal_mean": final_ideal.mean(axis=1),
        "final_ideal_sem": final_ideal.std(axis=1, ddof=1) / np.sqrt(repeats),
        "final_matched_mean": final_matched.mean(axis=1),
        "final_matched_sem": final_matched.std(axis=1, ddof=1) / np.sqrt(repeats),
        "validation_final_ideal_table_mean": final_ideal_table.mean(axis=1),
        "validation_final_matched_table_mean": final_matched_table.mean(axis=1),
        "validation_final_ideal_network_over_table": (
            final_ideal.mean(axis=1) / final_ideal_table.mean(axis=1)
        ),
        "validation_final_matched_network_over_table": (
            final_matched.mean(axis=1) / final_matched_table.mean(axis=1)
        ),
        "validation_final_ideal_exact_table_mean": final_ideal_exact.mean(axis=1),
        "validation_final_matched_exact_table_mean": final_matched_exact.mean(axis=1),
        "validation_final_ideal_network_over_exact_table": (
            final_ideal.mean(axis=1) / final_ideal_exact.mean(axis=1)
        ),
        "validation_final_matched_network_over_exact_table": (
            final_matched.mean(axis=1) / final_matched_exact.mean(axis=1)
        ),
        "repeats": np.asarray(repeats),
        "shots_per_run": np.asarray(profile["shots"]),
        "final_window": np.asarray(profile["final_window"]),
        "calibration_shots_per_bin": np.asarray(profile["calibration_shots"]),
    }
    result_path = args.output_dir / f"fig5_nn_results_{args.profile}.npz"
    np.savez_compressed(result_path, **arrays)
    info_path = args.output_dir / f"fig5_nn_results_{args.profile}.json"
    info_path.write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    print(f"Saved {info_path}")


if __name__ == "__main__":
    main()
