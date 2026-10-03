"""Generate raw QFI and pattern-CFI scans for one array geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ncl_sensing.physics import RydbergArray, RydbergConfig


PROFILES = {
    "quick": {
        "spacing": np.linspace(0.4, 1.4, 6),
        "detuning": np.linspace(0.0, 6.0, 13),
        "times": np.asarray([10.0, 100.0]),
    },
    "paper": {
        "spacing": np.linspace(0.4, 1.4, 100),
        "detuning": np.linspace(0.0, 6.0, 121),
        "times": np.linspace(0.0, 100.0, 101),
    },
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, required=True)
    parser.add_argument("--n-cols", type=int, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--difference-step", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "fig2" / "raw"
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    spacing_values = profile["spacing"]
    detuning_values = profile["detuning"]
    times = profile["times"]
    shape = (detuning_values.size, spacing_values.size, times.size)
    qfi = np.empty(shape, dtype=np.float64)
    cfi = np.empty(shape, dtype=np.float64)

    for spacing_index, spacing in enumerate(spacing_values):
        simulator = RydbergArray(
            RydbergConfig(
                args.n_rows,
                args.n_cols,
                spacing=float(spacing),
                omega=args.omega,
            )
        )
        for detuning_index, detuning in enumerate(detuning_values):
            metrics = simulator.metrics_over_times(
                detuning=float(detuning),
                times=times,
                step=args.difference_step,
            )
            qfi[detuning_index, spacing_index] = metrics["qfi"]
            cfi[detuning_index, spacing_index] = metrics["cfi_pattern"]
        print(
            f"geometry={args.n_rows}x{args.n_cols} "
            f"spacing={spacing_index + 1}/{spacing_values.size}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"fig2_{args.n_rows}x{args.n_cols}_{args.profile}.npz"
    )
    np.savez_compressed(
        output,
        n_rows=np.asarray(args.n_rows),
        n_cols=np.asarray(args.n_cols),
        omega=np.asarray(args.omega),
        difference_step=np.asarray(args.difference_step),
        spacing_values=spacing_values,
        detuning_values=detuning_values,
        times=times,
        qfi=qfi,
        cfi_pattern=cfi,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
