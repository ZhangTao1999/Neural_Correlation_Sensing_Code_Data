"""Paired, fresh neural-network vs Bayesian-table checks for manuscript Figs. 4/5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import copy
import os
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
DATA = PACKAGE / "data/processed"
OUTPUT_BASE = PACKAGE / "outputs/fig45"
OUTPUT = OUTPUT_BASE
REFINE = False
sys.path.insert(0, str(PACKAGE / "src"))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

from ncl_sensing.calibration import grouped_calibration, sample_counts
from ncl_sensing.inference import network_log_posteriors
from ncl_sensing.models import NetworkConfig, PosteriorNetwork
from ncl_sensing.patterns import all_bit_patterns, excitation_counts
from ncl_sensing.training import TrainingConfig, grouped_kl_loss, seed_everything, train_network
from ncl_sensing.validation import posterior_agreement


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def log_table(counts, alpha=0.0):
    values = np.asarray(counts, dtype=np.float64) + alpha
    assert np.all(values.sum(axis=1) > 0)
    with np.errstate(divide="ignore"):
        return np.log(values / values.sum(axis=1, keepdims=True))


def normalized(log_values):
    norm = logsumexp(log_values, axis=-1, keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError("Empirical zero counts have annihilated every candidate bin; no probability flooring is applied.")
    return np.exp(log_values - norm)


def full_trace(logs, shots):
    return normalized(np.cumsum(logs[shots], axis=0, dtype=np.float64))


def final_window(logs, shots, window=100):
    # This shortcut is used only for Fig. 5, whose probability tables are strictly positive.
    assert np.isfinite(logs).all()
    hist = np.bincount(shots[:-window], minlength=logs.shape[0])
    return normalized(hist @ logs + np.cumsum(logs[shots[-window:]], axis=0))


def paired_metrics(errors, reference):
    diff = errors - reference
    n = len(diff)
    sem = float(diff.std(ddof=1) / np.sqrt(n))
    return {
        "mse": float(errors.mean()),
        "reference_mse": float(reference.mean()),
        "ratio": float(errors.mean() / reference.mean()),
        "paired_difference": float(diff.mean()),
        "paired_difference_sem": sem,
        "paired_difference_normal_95_interval": [float(diff.mean() - 1.96 * sem), float(diff.mean() + 1.96 * sem)],
        "mse_sem": float(errors.std(ddof=1) / np.sqrt(n)),
    }


def test_equivalence():
    counts = torch.tensor([[2., 0., 1.], [0., 3., 2.]], dtype=torch.float64)
    logits = torch.tensor([[.2, -.4, .8], [-.1, .6, .3]], dtype=torch.float64, requires_grad=True)
    rows, labels = [], []
    for s in range(2):
        for j in range(3):
            rows.extend([s] * int(counts[s, j]))
            labels.extend([j] * int(counts[s, j]))
    snapshot_ce = F.cross_entropy(logits[rows], torch.tensor(labels))
    grouped_ce = -(counts * F.log_softmax(logits, dim=1)).sum() / counts.sum()
    kl = grouped_kl_loss(logits, counts / counts.sum(1, keepdim=True), counts.sum(1))
    grads = [torch.autograd.grad(x, logits, retain_graph=True)[0] for x in (snapshot_ce, grouped_ce, kl)]
    assert torch.allclose(snapshot_ce, grouped_ce, atol=1e-14, rtol=0)
    assert all(torch.allclose(grads[0], g, atol=1e-14, rtol=0) for g in grads[1:])
    rng = np.random.default_rng(20)
    logs = log_table(rng.random((16, 101)) + .01)
    shots = rng.integers(0, 16, 5000)
    assert np.allclose(full_trace(logs, shots)[-100:], final_window(logs, shots), atol=1e-10, rtol=1e-9)
    return {"snapshot_ce_equals_count_weighted_ce": True, "snapshot_ce_and_grouped_kl_gradients_equal": True, "window_shortcut_equals_full_updates": True}


def train(counts, n, alpha, seed, directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    save = directory / f"{label}_alpha{alpha:g}"
    cfg = NetworkConfig(input_size=n, output_size=101, hidden_size=512 if n == 9 else 128, hidden_blocks=3 if n == 9 else 2, batch_norm=False)
    tcfg = TrainingConfig(learning_rate=1e-3, max_epochs=15000 if n == 9 else 5000, patience=3000 if n == 9 else 1000, target_kl=2e-5 if n == 9 else 1e-7, seed=seed, device="cpu", log_every=500)
    cal = grouped_calibration(all_bit_patterns(n), counts, pseudocount=alpha)
    seed_everything(seed)
    model = PosteriorNetwork(cfg)
    print(f"TRAIN {label} alpha={alpha} seed={seed}", flush=True)
    start = time.perf_counter()
    if REFINE:
        initial = OUTPUT_BASE / directory.relative_to(OUTPUT) / f"{label}_alpha{alpha:g}.pt"
        model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=False)["state_dict"])
        fit, tcfg_dict = refine_model(model, counts, alpha)
        tcfg_dict["initial_checkpoint"] = Path(os.path.relpath(initial, PACKAGE)).as_posix()
    else:
        fit = train_network(model, cal.patterns, cal.targets, cal.weights, tcfg)
        tcfg_dict = asdict(tcfg)
    model.eval()
    if REFINE:
        with torch.no_grad():
            logs = F.log_softmax(model(torch.tensor(all_bit_patterns(n), dtype=torch.float32)).double(), dim=1).numpy()
    else:
        logs = network_log_posteriors(model, n).astype(np.float64)
    logs -= logsumexp(logs, axis=1, keepdims=True)
    target = np.exp(log_table(counts, alpha))
    info = {
        "network": asdict(cfg), "training": tcfg_dict, "pseudocount": alpha,
        "objective": "raw snapshot cross entropy (entropy-subtracted for monitoring)" if alpha == 0 else "cross entropy for add-one augmented counts; NOT raw snapshot CE",
        "best_loss": fit.best_loss, "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run,
        "converged": fit.converged, "seconds": time.perf_counter() - start,
        "agreement": posterior_agreement(target, logs, (counts + alpha).sum(1)).to_dict(),
    }
    torch.save(model.checkpoint(info), str(save) + ".pt")
    np.savez_compressed(str(save) + ".npz", counts=counts, logs=logs, loss_history=np.array(fit.history))
    write_json(Path(str(save) + ".json"), info)
    print(f"DONE {label} alpha={alpha}: {json.dumps(info)}", flush=True)
    return logs, info


def refine_model(model, counts, alpha):
    """Tighten calibration fit using only training loss, never sensing MSE."""
    n = model.config.input_size
    target_kl = 1e-7 if n == 9 else 1e-9
    x = torch.tensor(all_bit_patterns(n), dtype=torch.float32)
    augmented = np.asarray(counts, dtype=np.float64) + alpha
    y = torch.tensor(augmented / augmented.sum(1, keepdims=True), dtype=torch.float64)
    w = torch.tensor(augmented.sum(1) / augmented.sum(), dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=300, threshold=1e-5, min_lr=1e-7)
    best, best_epoch, state, history = float("inf"), 0, None, []
    model.train()
    for epoch in range(1, 12001):
        optimizer.zero_grad(set_to_none=True)
        logq = F.log_softmax(model(x).double(), dim=1)
        loss = (F.kl_div(logq, y, reduction="none").sum(1) * w).sum()
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best, best_epoch = value, epoch
            state = copy.deepcopy(model.state_dict())
        if best <= target_kl:
            break
        loss.backward()
        optimizer.step()
        scheduler.step(value)
        if epoch % 500 == 0:
            print(f"refine epoch={epoch} KL={value:.9g} best={best:.9g} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
    model.load_state_dict(state)
    fit = SimpleNamespace(best_loss=best, best_epoch=best_epoch, epochs_run=epoch, converged=best <= target_kl, history=history)
    config = {"optimizer": "Adam", "learning_rate": 1e-4, "max_epochs": 12000, "target_kl": target_kl, "loss_accumulation": "float64", "model_dtype": "float32", "scheduler": "ReduceLROnPlateau(factor=.5, patience=300, threshold=1e-5, min_lr=1e-7)", "stopping_uses": "calibration loss only"}
    return fit, config


def fig4(name, alphas):
    out = OUTPUT / f"fig4_{name}"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "fig4_calibration.npz") as d:
        counts = d[f"{name}_counts_matrix"]
        pool = d[f"{name}_eval_shots"]
        grid = d[f"{name}_grid_points"]
        truth = float(d[f"{name}_attr_ground_truth"])
        cfi = float(d[f"{name}_attr_gt_cfi"])
    n = 4 if name == "2x2" else 9
    seed = 17 if n == 4 else 18
    # Match the historical inset/main-panel sampling seeds, using the frozen independent sensing pool.
    rng = np.random.RandomState(1 if n == 4 else 4)
    shots = np.array([rng.choice(pool, size=1000, replace=False) for _ in range(500)], dtype=np.int16)
    arrays = {"counts": counts, "shots": shots, "grid": grid, "truth": truth, "crb": 1 / (np.arange(1, 1001) * cfi)}
    logs = {"table_raw": log_table(counts), "table_addone": log_table(counts, 1)}
    nr = excitation_counts(n)
    nr_counts = np.zeros((n + 1, len(grid)))
    np.add.at(nr_counts, nr, counts)
    logs["table_nr"] = log_table(nr_counts, 1)[nr]
    info = {"tests": test_equivalence(), "repeats": 500, "shots_per_trial": 1000, "seed": seed, "sensing_seed": 1 if n == 4 else 4, "calibration_shots_per_bin": int(counts.sum(0)[0]), "training": {}}
    for alpha in alphas:
        key = "nn_raw" if alpha == 0 else "nn_addone"
        logs[key], info["training"][key] = train(counts, n, alpha, seed, out, name)
    checkpoints = np.array([10, 100, 500, 1000])
    for key, value in logs.items():
        estimates = np.empty((500, 1000))
        post = np.empty((500, 4, 101))
        for r, seq in enumerate(shots):
            p = full_trace(value, seq)
            estimates[r] = p @ grid
            post[r] = p[checkpoints - 1]
        arrays[f"{key}_logs"] = value
        arrays[f"{key}_estimates"] = estimates
        arrays[f"{key}_errors"] = (estimates - truth) ** 2
        arrays[f"{key}_posteriors"] = post
    info["comparisons"] = {}
    for candidate, reference in [("nn_raw", "table_raw"), ("nn_raw", "table_addone"), ("nn_addone", "table_addone"), ("table_raw", "table_addone")]:
        if candidate not in logs:
            continue
        key = f"{candidate}_vs_{reference}"
        info["comparisons"][key] = {
            str(m): paired_metrics(arrays[f"{candidate}_errors"][:, m - 1], arrays[f"{reference}_errors"][:, m - 1]) for m in checkpoints
        }
        tv = .5 * np.abs(arrays[f"{candidate}_posteriors"] - arrays[f"{reference}_posteriors"]).sum(-1)
        arrays[key + "_posterior_tv"] = tv
        info["comparisons"][key]["mean_posterior_tv"] = tv.mean(0).tolist()
    np.savez_compressed(out / "comparison.npz", **arrays)
    write_json(out / "comparison.json", info)
    print(f"SAVED Fig4 {name} {out}", flush=True)


def fig5(alphas):
    out = OUTPUT / "fig5"
    out.mkdir(parents=True, exist_ok=True)
    source = DATA / "fig5_probabilities.h5"
    with h5py.File(source, "r") as f:
        grid = f.attrs["grid_points"]
        truth = float(f.attrs["gt_val"])
        gammas = f["gamma_values"][:]
        probabilities = {float(g): f[f"gamma_{g}/probabilities"][:] for g in gammas}
    rng = np.random.default_rng(10042)
    arrays = {"grid": grid, "truth": truth, "gammas": gammas[gammas > 0]}
    info = {"tests": test_equivalence(), "repeats": 500, "shots_per_trial": 5000, "final_window": 100, "calibration_shots_per_bin": 1000000, "sensing_seed": 10042, "training": {}, "comparisons": {}}
    trained = {}
    for i, gamma in enumerate(gammas):
        label = f"gamma_{gamma:g}"
        p = probabilities[float(gamma)]
        assert np.allclose(p.sum(1), 1, atol=1e-12)
        seed = 42 if i == 0 else 42 + i + 1
        counts = sample_counts(p, 1000000, seed)
        tables = {
            "table_raw": log_table(counts),
            "table_addone": log_table(counts, 1),
            "table_exact": log_table(p.T),
            "table_historical": log_table(p.T * 1000000, 1),
        }
        arrays[label + "_calibration_counts"] = counts
        for alpha in alphas:
            key = "nn_raw" if alpha == 0 else "nn_addone"
            tables[key], info["training"][label + "_" + key] = train(counts, 4, alpha, seed, out / "training", label)
        trained[float(gamma)] = tables
        for key, table in tables.items():
            arrays[label + "_" + key + "_logs"] = table
        if gamma == 0:
            continue
        gt_index = int(np.argmin(abs(grid - truth)))
        shots = rng.choice(16, size=(500, 5000), p=p[gt_index]).astype(np.uint8)
        arrays[label + "_shots"] = shots
        modes = {"matched": tables, "ideal": trained[0.]}
        for mode, mode_tables in modes.items():
            for key, table in mode_tables.items():
                window_errors = np.empty(500)
                final_estimates = np.empty(500)
                is_trace = np.isclose(gamma, .01)
                trace_errors = np.empty((500, 5000)) if is_trace else None
                for r, seq in enumerate(shots):
                    ppost = full_trace(table, seq) if is_trace else final_window(table, seq)
                    estimates = ppost @ grid
                    errors = (estimates - truth) ** 2
                    window_errors[r] = errors[-100:].mean()
                    final_estimates[r] = estimates[-1]
                    if is_trace:
                        trace_errors[r] = errors
                prefix = f"{label}_{mode}_{key}"
                arrays[prefix + "_window_errors"] = window_errors
                arrays[prefix + "_final_estimates"] = final_estimates
                if is_trace:
                    arrays[prefix + "_trace_errors"] = trace_errors
            for candidate, reference in [("nn_raw", "table_raw"), ("nn_raw", "table_historical"), ("nn_addone", "table_addone"), ("table_raw", "table_historical")]:
                if candidate not in mode_tables:
                    continue
                info["comparisons"][f"{label}_{mode}_{candidate}_vs_{reference}"] = paired_metrics(arrays[f"{label}_{mode}_{candidate}_window_errors"], arrays[f"{label}_{mode}_{reference}_window_errors"])
        # Save each completed decay rate, so partial progress can be independently checked.
        np.savez_compressed(out / "comparison.npz", **arrays)
        write_json(out / "comparison.json", info)
        print(f"SAVED Fig5 gamma={gamma:g}", flush=True)


def main():
    global OUTPUT, REFINE
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=("fig4-2x2", "fig4-3x3", "fig5", "test"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0., 1.])
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    REFINE = args.refine
    if REFINE:
        OUTPUT = OUTPUT_BASE / "refined"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.figure == "test":
        print(json.dumps(test_equivalence()))
    elif args.figure.startswith("fig4"):
        fig4(args.figure[5:], args.alphas)
    else:
        fig5(args.alphas)


if __name__ == "__main__":
    main()
