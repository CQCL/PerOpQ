from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array  # type: ignore[import-untyped]

from peropq.commutators_bitstrings import get_commutator_pauli_tensors
from peropq.hamiltonian import Hamiltonian
from peropq.pauli_bitstring import PauliString


class VariationalUnitary:
    """class representing the variational unitary ansatz."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        number_of_layer: int,
        time: float,
    ) -> None:
        """
        Init function.

        :param hamiltonian: Hamiltonian of which one is interested in the dynamics.
        :param number_of_layer: number of layers for the optimization.
        :param time: final time to up to which one wants to perform the time evolution.
        """
        self.n_terms: int = hamiltonian.get_n_terms()
        self.pauli_string_list: Sequence[PauliString] = hamiltonian.pauli_string_list
        self.depth: int = number_of_layer
        self.theta: npt.NDArray = 1j * np.zeros((number_of_layer, self.n_terms))
        self.cjs: Sequence[complex] = hamiltonian.get_cjs()
        self.time: float = time
        self.test: npt.NDArray = np.zeros((number_of_layer, number_of_layer))
        self.trace_calculated = False
        for r in range(number_of_layer):
            for s in range(number_of_layer):
                self.test[r, s] = -1 if s > r else 1

    def update_theta(self, new_array: npt.NDArray) -> None:
        """
         Update theta ensuring that the condition Sum_i theta_i dt_i= is ensured.

        :param new_array: the new array containing the variational parameters. It's shape must be (R - 1, n_terms).
        """
        if new_array.shape != (self.depth - 1, self.n_terms):
            if self.depth == 1 and new_array.shape == (1, self.n_terms):
                pass
            else:
                error_message = "Wrong length provided."
                raise ValueError(error_message)
        for j in range(self.n_terms):
            for r in range(self.depth - 1):
                self.theta[r, j] = new_array[r, j]
            self.theta[self.depth - 1, j] = self.time * self.cjs[j]
            for r in range(self.depth - 1):
                self.theta[self.depth - 1, j] -= new_array[r, j]

    def get_initial_trotter_vector(self) -> npt.NDArray:
        """Get the variational parameters corresponding to the Trotterization. Useful to initialize the optimization."""
        theta_trotter: npt.NDArray
        if self.depth > 1:
            theta_trotter = np.zeros((self.depth - 1, self.n_terms), dtype=np.complex64)
            for j in range(self.n_terms):
                for r in range(self.depth - 1):
                    theta_trotter[r, j] = self.cjs[j] * self.time / self.depth
        else:
            theta_trotter = np.zeros((self.depth, self.n_terms), dtype=np.complex64)
            for j in range(self.n_terms):
                theta_trotter[0, j] = self.cjs[j] * self.time
        return theta_trotter

    def flatten_theta(self, theta: npt.NDArray) -> npt.NDArray:
        """Returns the variational parameters as flatten (R-1)*n_terms array. Useful to pass to a minimization function."""
        if self.depth > 1:
            return np.array(theta).reshape((self.depth - 1) * self.n_terms)
        return np.array(theta).reshape(self.n_terms)

    def unflatten_theta(self, flat_theta: npt.NDArray) -> npt.NDArray:
        """Returns the flattened variational parameters as an array with shape (depth,n_terms)."""
        if self.depth > 1:
            return np.array(flat_theta).reshape((self.depth - 1, self.n_terms))
        return np.array(flat_theta).reshape((1, self.n_terms))

    def set_theta_to_trotter(self) -> None:
        """Sets the variational parameters to the Trotter parameters."""
        theta_trotter: npt.NDArray = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)

    def chi_tensor(
        self,
        left_indices: npt.NDArray,
        right_indices: npt.NDArray,
    ) -> npt.NDArray:
        """
        Vectorized function to calculate all the chi coefficient at once.

        :param left_indices: indices of the left tensor which give non-zero contributions in the calculation of chi.
        :param right_indices: indices of the right tensor which give non-zero contributions in the calculation of chi.
        :returns: numpy array representing the tensor used in the norm calculation.
        """
        theta_left: npt.NDArray = self.theta[:, left_indices]
        theta_right: npt.NDArray = self.theta[:, right_indices]
        res: npt.NDArray = np.tensordot(theta_left, self.test, ([0], [0]))
        res = np.tensordot(res, theta_right, ([1], [0]))
        return 0.5 * res

    def calculate_traces(self) -> None:
        """Calculate the trace tensor."""
        commutators: list[tuple[int, PauliString]] = []
        index_pairs: list[tuple[int, int]] = []
        i = 0
        for j_prime, h_j_prime in enumerate(
            self.pauli_string_list,
        ):
            for j in range(j_prime + 1, self.n_terms):
                h_j = self.pauli_string_list[j]
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
        self.trace_calculated = True

    def c2_squared(self, theta: npt.ArrayLike = ()) -> float:
        """
        Perturbative 2-norm.

        :param theta: parameters of the variational unitary.
        :returns: the perturbative approximation of the 2-norm difference between the exact and the variational representation.
        """
        if not self.trace_calculated:
            self.calculate_traces()
        theta_new = self.unflatten_theta(np.array(theta))
        self.update_theta(theta_new)
        chi_tensor = self.chi_tensor(
            self.left_indices,
            self.right_indices,
        )
        s1, s2, s3, s4 = self.trace_tensor.shape
        chi_tensor = chi_tensor.reshape((s1 * s2,))
        trace_tensor: csr_array = csr_array(
            self.trace_tensor.reshape((s1 * s2, s3 * s4)),
        )
        return np.real(-chi_tensor.T @ trace_tensor @ chi_tensor)
