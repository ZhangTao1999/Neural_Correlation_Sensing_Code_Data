"""Binary measurement-pattern utilities."""

from __future__ import annotations

import numpy as np


def all_bit_patterns(n_qubits: int, dtype=np.float32) -> np.ndarray:
    """Return computational-basis patterns ordered by their integer index."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    indices = np.arange(2**n_qubits, dtype=np.uint64)[:, None]
    shifts = np.arange(n_qubits - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((indices >> shifts) & 1).astype(dtype, copy=False)


def excitation_counts(n_qubits: int) -> np.ndarray:
    """Return the total number of excited sites for every pattern index."""
    return all_bit_patterns(n_qubits, dtype=np.int8).sum(axis=1)


def pattern_index(bits: np.ndarray) -> int:
    """Convert a most-significant-bit-first binary pattern to an integer."""
    bits = np.asarray(bits, dtype=np.int64)
    if bits.ndim != 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be a one-dimensional binary array")
    powers = 1 << np.arange(bits.size - 1, -1, -1)
    return int(bits @ powers)

