from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]
from scipy.sparse import csr_array  # type: ignore[import-untyped]

from peropq.commutators import get_commutator_pauli_tensors
from peropq.pauli import PauliString
from peropq.variational_unitary import VariationalUnitary


class Optimizer:
    """Class performing the optimizer."""

    def __init__(self, variation_unitary: VariationalUnitary) -> None:
        """
        Initialize the optimizer.

        param: variational_unitary ansatz to be optimized.
        """
        self.variational_unitary = variation_unitary
        self.variational_unitary.set_theta_to_trotter()

        commutators: list[tuple[int, PauliString]] = []
        index_pairs: list[tuple[int, int]] = []
        i = 0
        for j_prime, h_j_prime in enumerate(
            self.variational_unitary.hamiltonian.pauli_string_list,
        ):
            for j in range(j_prime + 1, self.variational_unitary.n_terms):
                h_j = self.variational_unitary.hamiltonian.pauli_string_list[j]
                commutator = get_commutator_pauli_tensors(h_j, h_j_prime)
                if commutator:
                    index_pairs.append((j, j_prime))
                    commutators.append((i, commutator))
                    i += 1
        self.index_pairs: npt.NDArray = np.array(index_pairs)
        self.left_indices: npt.NDArray = np.unique(self.index_pairs[:, 0])
        self.right_indices: npt.NDArray = np.unique(self.index_pairs[:, 1])
        self.trace_tensor: npt.NDArray = 1j * np.zeros(
            (
                len(self.left_indices),
                len(self.right_indices),
                len(self.left_indices),
                len(self.right_indices),
            ),
        )
        for j_, j_commutator in commutators:
            for k_, k_commutator in commutators:
                if j_ < k_:
                    continue
                product_commutators: PauliString = j_commutator * k_commutator
                if product_commutators != 0:
                    trace: complex = product_commutators.normalized_trace()
                    if trace:
                        fac = 1 if j_ == k_ else 2.0
                        # Get the new indices
                        new_j: int = np.where(
                            self.left_indices == self.index_pairs[j_, 0],
                        )[0].item()
                        new_j_prime: int = np.where(
                            self.right_indices == self.index_pairs[j_, 1],
                        )[0].item()
                        new_k: int = np.where(
                            self.left_indices == self.index_pairs[k_, 0],
                        )[0].item()
                        new_k_prime: int = np.where(
                            self.right_indices == self.index_pairs[k_, 1],
                        )[0].item()
                        self.trace_tensor[new_j, new_j_prime, new_k, new_k_prime] = (
                            fac * product_commutators.normalized_trace()
                        )

    def c2_squared(self, theta: Sequence[float] = []) -> float:
        """
        Perturbative 2-norm.

        param: theta parameters of the variational unitary.
        returns: the perturbative approximation of the 2-norm difference between the exact and the variational representation.
        """
        if len(theta) != 0:
            theta_new = np.array(theta).reshape(
                (
                    self.variational_unitary.depth - 1,
                    self.variational_unitary.n_terms,
                ),
            )
            self.variational_unitary.update_theta(theta_new)
        chi_tensor = self.variational_unitary.chi_tensor(
            self.left_indices,
            self.right_indices,
        )
        s1, s2, s3, s4 = self.trace_tensor.shape
        chi_tensor = chi_tensor.reshape((s1 * s2,))
        trace_tensor: csr_array = csr_array(
            self.trace_tensor.reshape((s1 * s2, s3 * s4)),
        )
        return np.real(-chi_tensor.T @ trace_tensor @ chi_tensor)

    def get_minumum_c2_squared(
        self,
        initial_guess: Sequence[float] = [],
    ) -> tuple[scipy.optimize.OptimizeResult, float]:
        """
        Perform the minimization.

        param: initial_guess initial guess for the optimization. If not provided, use the parameters of the Trotterization instead
        returns: the result of the optimization
        returns: the perturbative 2-norm
        """
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = self.variational_unitary.get_initial_trotter_vector()
            x0 = self.variational_unitary.flatten_theta(x0)
        optimized_results = scipy.optimize.minimize(self.c2_squared, x0)
        return optimized_results, self.c2_squared(theta=optimized_results.x)
