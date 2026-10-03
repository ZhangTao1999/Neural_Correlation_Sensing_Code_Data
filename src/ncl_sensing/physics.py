"""Small-system Rydberg-array dynamics used by the figure simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .patterns import all_bit_patterns


@dataclass(frozen=True)
class RydbergConfig:
    n_rows: int
    n_cols: int
    spacing: float
    omega: float = 1.0


class RydbergArray:
    """Dense exact simulator for the effective two-level Hamiltonian.

    The dimensionless convention is identical to the legacy scripts:
    ``C6 = hbar = 1`` and all frequencies are expressed in units of ``Omega``.
    The basis index is the integer represented by the site-occupation bit string,
    with sites ordered row by row.
    """

    def __init__(self, config: RydbergConfig):
        if config.n_rows < 1 or config.n_cols < 1:
            raise ValueError("array dimensions must be positive")
        if config.spacing <= 0:
            raise ValueError("spacing must be positive")
        self.config = config
        self.n_qubits = config.n_rows * config.n_cols
        self.dimension = 2**self.n_qubits
        self.patterns = all_bit_patterns(self.n_qubits, dtype=np.float64)
        self.excitation_numbers = self.patterns.sum(axis=1)
        self.interaction_energies = self._interaction_energies()
        self.drive = self._drive_operator()
        self.initial_state = np.zeros(self.dimension, dtype=np.complex128)
        self.initial_state[0] = 1.0

    def _coordinates(self) -> np.ndarray:
        return np.asarray(
            [
                (column * self.config.spacing, row * self.config.spacing)
                for row in range(self.config.n_rows)
                for column in range(self.config.n_cols)
            ],
            dtype=np.float64,
        )

    def _interaction_energies(self) -> np.ndarray:
        coordinates = self._coordinates()
        energies = np.zeros(self.dimension, dtype=np.float64)
        for left in range(self.n_qubits):
            for right in range(left + 1, self.n_qubits):
                distance = np.linalg.norm(coordinates[left] - coordinates[right])
                coupling = distance**-6
                energies += (
                    coupling
                    * self.patterns[:, left]
                    * self.patterns[:, right]
                )
        return energies

    def _drive_operator(self) -> np.ndarray:
        drive = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for state in range(self.dimension):
            for site in range(self.n_qubits):
                flipped = state ^ (1 << (self.n_qubits - site - 1))
                drive[state, flipped] = 0.5
        return drive

    def hamiltonian(self, detuning: float, omega: float | None = None) -> np.ndarray:
        """Return ``H/hbar`` for a uniform drive and detuning."""
        if omega is None:
            omega = self.config.omega
        diagonal = self.interaction_energies - detuning * self.excitation_numbers
        return np.diag(diagonal) + float(omega) * self.drive

    def statevector(
        self,
        detuning: float,
        time: float,
        omega: float | None = None,
    ) -> np.ndarray:
        """Evolve the all-ground-state product state for a fixed duration."""
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            self.hamiltonian(detuning, omega),
            overwrite_a=True,
            check_finite=False,
        )
        coefficients = eigenvectors.conj().T @ self.initial_state
        return eigenvectors @ (np.exp(-1j * eigenvalues * time) * coefficients)

    def metrics_over_times(
        self,
        detuning: float,
        times: np.ndarray,
        step: float = 1e-4,
        omega: float | None = None,
        include_qfi: bool = True,
        include_cfi: bool = True,
        probability_floor: float = 1e-12,
    ) -> dict[str, np.ndarray]:
        """Evaluate QFI and pattern CFI after three shared diagonalizations."""
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")

        states = []
        for shifted_detuning in (detuning, detuning + step, detuning - step):
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                self.hamiltonian(shifted_detuning, omega),
                overwrite_a=True,
                check_finite=False,
            )
            coefficients = eigenvectors.conj().T @ self.initial_state
            phases = np.exp(-1j * np.outer(eigenvalues, times))
            states.append(eigenvectors @ (coefficients[:, None] * phases))
        nominal, plus, minus = states
        results = {}

        if include_qfi:
            derivative = (plus - minus) / (2.0 * step)
            norm = np.sum(np.abs(derivative) ** 2, axis=0)
            overlap = np.sum(nominal.conj() * derivative, axis=0)
            results["qfi"] = 4.0 * (norm - np.abs(overlap) ** 2).real

        if include_cfi:
            nominal_probability = np.abs(nominal) ** 2
            derivative_probability = (
                np.abs(plus) ** 2 - np.abs(minus) ** 2
            ) / (2.0 * step)
            terms = np.zeros_like(nominal_probability)
            keep = nominal_probability > probability_floor
            terms[keep] = (
                derivative_probability[keep] ** 2 / nominal_probability[keep]
            )
            results["cfi_pattern"] = terms.sum(axis=0)
        return results

    def probabilities(
        self,
        detuning: float,
        time: float,
        omega: float | None = None,
    ) -> np.ndarray:
        state = self.statevector(detuning, time, omega)
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()

    def qfi(
        self,
        detuning: float,
        time: float,
        step: float = 1e-4,
        omega: float | None = None,
    ) -> float:
        """Pure-state QFI evaluated by the central difference used in Fig. 2."""
        plus = self.statevector(detuning + step, time, omega)
        minus = self.statevector(detuning - step, time, omega)
        state = self.statevector(detuning, time, omega)
        derivative = (plus - minus) / (2.0 * step)
        return float(
            4.0
            * (
                np.vdot(derivative, derivative).real
                - abs(np.vdot(state, derivative)) ** 2
            )
        )

    def cfi(
        self,
        detuning: float,
        time: float,
        step: float = 1e-4,
        omega: float | None = None,
        probability_floor: float = 1e-12,
    ) -> float:
        """Computational-basis CFI evaluated by central differences."""
        nominal = self.probabilities(detuning, time, omega)
        plus = self.probabilities(detuning + step, time, omega)
        minus = self.probabilities(detuning - step, time, omega)
        derivative = (plus - minus) / (2.0 * step)
        keep = nominal > probability_floor
        return float(np.sum(derivative[keep] ** 2 / nominal[keep]))


class LindbladRydbergArray(RydbergArray):
    """Exact master-equation simulator with independent Rydberg decay."""

    def __init__(self, config: RydbergConfig):
        super().__init__(config)
        self.decay_operators = self._decay_operators()

    def _decay_operators(self) -> tuple[np.ndarray, ...]:
        operators = []
        for site in range(self.n_qubits):
            bit = 1 << (self.n_qubits - site - 1)
            operator = np.zeros(
                (self.dimension, self.dimension), dtype=np.complex128
            )
            for state in range(self.dimension):
                if state & bit:
                    operator[state ^ bit, state] = 1.0
            operators.append(operator)
        return tuple(operators)

    def probabilities_with_decay(
        self,
        detuning: float,
        time: float,
        gamma: float,
        omega: float | None = None,
    ) -> np.ndarray:
        """Return computational-basis populations after Lindblad evolution."""
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if gamma == 0:
            return self.probabilities(detuning, time, omega)

        hamiltonian = self.hamiltonian(detuning, omega).astype(np.complex128)
        identity = np.eye(self.dimension, dtype=np.complex128)
        liouvillian = -1j * (
            np.kron(hamiltonian, identity)
            - np.kron(identity, hamiltonian.T)
        )
        for operator in self.decay_operators:
            number = operator.conj().T @ operator
            liouvillian += gamma * (
                np.kron(operator, operator.conj())
                - 0.5 * np.kron(identity, number.T)
                - 0.5 * np.kron(number, identity)
            )

        initial_density = np.zeros(
            (self.dimension, self.dimension), dtype=np.complex128
        )
        initial_density[0, 0] = 1.0
        evolved = scipy.linalg.expm(liouvillian * time) @ initial_density.ravel()
        density = evolved.reshape((self.dimension, self.dimension))
        probabilities = np.clip(np.real(np.diag(density)), 0.0, 1.0)
        return probabilities / probabilities.sum()
