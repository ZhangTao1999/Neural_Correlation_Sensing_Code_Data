"""Load portable physical inputs used by the figure workflows."""
from pathlib import Path
import h5py
import numpy as np


def load_fig5_probabilities(path: Path):
    """Read the distributed HDF5 table or an independently regenerated NPZ."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as d:
            grid = d["grid_points"]
            truth = float(d["ground_truth"])
            gammas = d["gamma_values"]
            probabilities = d["probabilities"]
        if probabilities.shape[:2] != (len(gammas), len(grid)):
            raise ValueError("Fig. 5 probability table dimensions do not match its grids")
        tables = {float(g): probabilities[i] for i, g in enumerate(gammas)}
    else:
        with h5py.File(path, "r") as f:
            grid = np.asarray(f.attrs["grid_points"])
            truth = float(f.attrs["gt_val"])
            gammas = f["gamma_values"][:]
            tables = {float(g): f[f"gamma_{g}/probabilities"][:] for g in gammas}
    return grid, truth, gammas, tables
