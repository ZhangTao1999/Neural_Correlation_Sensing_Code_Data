"""Train neural estimators and regenerate the sensing-performance data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ncl_sensing.calibration import grouped_calibration
from ncl_sensing.inference import (
    network_log_posteriors,
    posterior_means,
    posterior_trace,
)
from ncl_sensing.models import NetworkConfig, PosteriorNetwork
from ncl_sensing.patterns import all_bit_patterns, excitation_counts
from ncl_sensing.training import TrainingConfig, seed_everything, train_network
from ncl_sensing.validation import posterior_agreement


PROFILES = {
    "quick": {
        "repeats": 25,
        "shots": 300,
        "2x2": {"hidden_size": 64, "hidden_blocks": 1, "max_epochs": 800, "target_kl": 2e-5},
        "3x3": {"hidden_size": 128, "hidden_blocks": 1, "max_epochs": 800, "target_kl": 2e-3},
    },
    "paper": {
        "repeats": 500,
        "shots": 1000,
        "2x2": {"hidden_size": 128, "hidden_blocks": 2, "max_epochs": 5000, "target_kl": 1e-7},
        "3x3": {"hidden_size": 512, "hidden_blocks": 3, "max_epochs": 15000, "target_kl": 2e-5},
    },
}


def nr_log_posteriors(counts: np.ndarray, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    nr_by_pattern = excitation_counts(n_qubits)
    nr_counts = np.zeros((n_qubits + 1, counts.shape[1]), dtype=np.float64)
    for pattern in range(counts.shape[0]):
        nr_counts[nr_by_pattern[pattern]] += counts[pattern]
    targets = nr_counts + 1.0
    targets /= targets.sum(axis=1, keepdims=True)
    return np.log(targets), nr_by_pattern


def evaluate_system(
    group: dict[str, np.ndarray | float | int],
    name: str,
    profile_name: str,
    profile: dict,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    counts = np.asarray(group["counts_matrix"])
    eval_shots = np.asarray(group["eval_shots"])
    grid = np.asarray(group["grid_points"])
    n_qubits = int(group["n_row"] * group["n_col"])
    ground_truth = float(group["ground_truth"])
    cfi = float(group["gt_cfi"])

    calibration = grouped_calibration(
        all_bit_patterns(n_qubits), counts, pseudocount=0.0
    )
    network_config = NetworkConfig(
        input_size=n_qubits,
        output_size=grid.size,
        hidden_size=profile[name]["hidden_size"],
        hidden_blocks=profile[name]["hidden_blocks"],
        batch_norm=False,
    )
    training_config = TrainingConfig(
        learning_rate=1e-3,
        max_epochs=profile[name]["max_epochs"],
        patience=max(500, profile[name]["max_epochs"] // 5),
        target_kl=profile[name]["target_kl"],
        seed=seed,
        device="auto",
        log_every=500,
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
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / profile_name
        / f"fig4_{name}_network.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.checkpoint(
            {
                "figure": "Fig. 4",
                "system": name,
                "training": asdict(training_config),
                "result": {
                    "best_loss": result.best_loss,
                    "best_epoch": result.best_epoch,
                    "epochs_run": result.epochs_run,
                    "converged": result.converged,
                },
            }
        ),
        checkpoint_path,
    )

    network_logs = network_log_posteriors(model, n_qubits)
    calibration_logs = np.log(
        np.clip(calibration.targets.astype(np.float64), np.finfo(float).tiny, None)
    )
    agreement = posterior_agreement(
        calibration.targets, network_logs, calibration.weights
    )
    nr_logs, nr_by_pattern = nr_log_posteriors(counts, n_qubits)
    repeats = profile["repeats"]
    shots_per_run = profile["shots"]
    rng = np.random.default_rng(seed + 1000)
    mse_pattern = np.empty((repeats, shots_per_run), dtype=np.float64)
    mse_pattern_table = np.empty_like(mse_pattern)
    mse_nr = np.empty_like(mse_pattern)
    checkpoints = [value for value in (10, 100, 500, 1000) if value <= shots_per_run]
    posterior_sums = {value: np.zeros(grid.size) for value in checkpoints}

    for repeat in range(repeats):
        shots = rng.choice(eval_shots, size=shots_per_run, replace=False)
        pattern_trace = posterior_trace(network_logs, shots)
        table_trace = posterior_trace(calibration_logs, shots)
        nr_trace = posterior_trace(nr_logs, nr_by_pattern[shots])
        pattern_estimates = posterior_means(pattern_trace, grid)
        table_estimates = posterior_means(table_trace, grid)
        nr_estimates = posterior_means(nr_trace, grid)
        mse_pattern[repeat] = (pattern_estimates - ground_truth) ** 2
        mse_pattern_table[repeat] = (table_estimates - ground_truth) ** 2
        mse_nr[repeat] = (nr_estimates - ground_truth) ** 2
        for checkpoint in checkpoints:
            posterior_sums[checkpoint] += pattern_trace[checkpoint - 1]

    output = {
        f"{name}_grid": grid,
        f"{name}_mse_pattern": mse_pattern.mean(axis=0),
        f"{name}_mse_pattern_table_validation": mse_pattern_table.mean(axis=0),
        f"{name}_mse_nr": mse_nr.mean(axis=0),
        f"{name}_crb": 1.0 / (np.arange(1, shots_per_run + 1) * cfi),
        f"{name}_ground_truth": np.asarray(ground_truth),
    }
    for checkpoint, values in posterior_sums.items():
        output[f"{name}_posterior_m{checkpoint}"] = values / repeats

    diagnostics = {
        "network_config": asdict(network_config),
        "training_config": asdict(training_config),
        "training_result": {
            "best_loss": result.best_loss,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "converged": result.converged,
        },
        "network_vs_calibration_table": agreement.to_dict(),
        "mse_ratio_network_over_table": {
            str(checkpoint): float(
                mse_pattern[:, checkpoint - 1].mean()
                / mse_pattern_table[:, checkpoint - 1].mean()
            )
            for checkpoint in checkpoints
        },
        "repeats": repeats,
        "shots_per_run": shots_per_run,
        "checkpoint": checkpoint_path.relative_to(output_dir).as_posix(),
    }
    return output, diagnostics


def load_systems(path: Path) -> dict[str, dict[str, np.ndarray | float | int]]:
    """Load Fig. 4 calibration data from HDF5 or portable NPZ."""
    names = ("2x2", "3x3")
    dataset_keys = ("counts_matrix", "eval_shots", "grid_points")
    attribute_keys = ("n_row", "n_col", "ground_truth", "gt_cfi")
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return {
                name: {
                    **{key: data[f"{name}_{key}"] for key in dataset_keys},
                    **{
                        key: data[f"{name}_attr_{key}"].item()
                        for key in attribute_keys
                    },
                }
                for name in names
            }

    import h5py

    systems = {}
    with h5py.File(path, "r") as handle:
        for name in names:
            group = handle[name]
            systems[name] = {
                **{key: group[key][:] for key in dataset_keys},
                **{key: group.attrs[key] for key in attribute_keys},
            }
    return systems


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("2x2", "3x3"),
        default=["2x2", "3x3"],
        help="Systems to train. Completed systems are saved independently.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "processed" / "fig4_calibration.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "fig4")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]
    systems = load_systems(args.data)
    for name in args.systems:
        offset = ("2x2", "3x3").index(name)
        values, info = evaluate_system(
            systems[name],
            name,
            args.profile,
            profile,
            args.seed + offset,
            args.output_dir,
        )
        system_path = (
            args.output_dir / f"fig4_{name}_nn_results_{args.profile}.npz"
        )
        np.savez_compressed(system_path, **values)
        system_info_path = (
            args.output_dir / f"fig4_{name}_nn_results_{args.profile}.json"
        )
        system_info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Saved {system_path}")
        print(f"Saved {system_info_path}")

    arrays: dict[str, np.ndarray] = {
        "profile": np.asarray(args.profile),
        "seed": np.asarray(args.seed),
        "repeats": np.asarray(profile["repeats"]),
        "shots_per_run": np.asarray(profile["shots"]),
    }
    diagnostics = {}
    for name in ("2x2", "3x3"):
        system_path = args.output_dir / f"fig4_{name}_nn_results_{args.profile}.npz"
        system_info_path = (
            args.output_dir / f"fig4_{name}_nn_results_{args.profile}.json"
        )
        if not system_path.exists() or not system_info_path.exists():
            continue
        with np.load(system_path) as values:
            arrays.update({key: values[key] for key in values.files})
        diagnostics[name] = json.loads(system_info_path.read_text(encoding="utf-8"))

    if set(diagnostics) != {"2x2", "3x3"}:
        missing = sorted({"2x2", "3x3"} - set(diagnostics))
        print(f"Combined Fig. 4 result not written; missing systems: {', '.join(missing)}")
        return

    result_path = args.output_dir / f"fig4_nn_results_{args.profile}.npz"
    np.savez_compressed(result_path, **arrays)
    diagnostics_path = args.output_dir / f"fig4_nn_results_{args.profile}.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    print(f"Saved {diagnostics_path}")


if __name__ == "__main__":
    main()
