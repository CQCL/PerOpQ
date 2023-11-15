from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

from peropq.hamiltonian import Hamiltonian


class VariationalUnitary:
    """class representing the variational unitary ansataz."""

    def __init__(
        self, hamiltonian: Hamiltonian, number_of_layer: int, time: float
    ) -> None:
        """
        Init function.

        :param hamiltonian Hamiltonian of which one is interested in the dynamics.
        :param R number of steps for the optimization.
        :param t final time to up to which one wants to perform the time evolution.
        """
        self.hamiltonian: Hamiltonian = hamiltonian
        self.n_terms: int = hamiltonian.get_n_terms()
        self.number_of_layer: float = number_of_layer
        self.theta: np.array = np.zeros((number_of_layer, self.n_terms))
        self.cjs: Sequence[complex] = hamiltonian.get_cjs()
        self.time: float = time
        self.test: np.array = np.zeros((number_of_layer, number_of_layer))
        for r in range(number_of_layer):
            for s in range(number_of_layer):
                self.test[r, s] = -1 if s > r else 1

    def update_theta(self, new_array: np.array) -> None:
        """
         Update theta ensuring that the condition Sum_i theta_i dt_i= is ensured.

        :param new_array the new array containing the variational parameters. It's shape must be (R - 1, n_terms).
        """
        if new_array.shape != (self.number_of_layer - 1, self.n_terms):
            raise ValueError("Wrong length provided.")
        for j in range(self.n_terms):
            for r in range(self.number_of_layer - 1):
                self.theta[r, j] = new_array[r, j]
            self.theta[self.number_of_layer - 1, j] = self.time * self.cjs[j]
            for r in range(self.number_of_layer - 1):
                self.theta[self.number_of_layer - 1, j] -= new_array[r, j]

    def get_initial_trotter_vector(self) -> np.array:
        """Get the variational parameters corresponding to the Trotterization. Useful to initialize the optimization."""
        theta_trotter: np.array = np.zeros((self.number_of_layer - 1, self.n_terms))
        for j in range(self.n_terms):
            for r in range(self.number_of_layer - 1):
                theta_trotter[r, j] = self.cjs[j] * self.time / self.number_of_layer
        return theta_trotter

    def flatten_theta(self, theta: np.array) -> np.array:
        """Returns the variational parameters as flatten (R-1)*n_terms array. Useful to pass to a minimization function."""
        return np.array(theta).reshape((self.number_of_layer - 1) * self.n_terms)

    def set_theta_to_Trotter(self) -> None:
        """Sets the variational parameters to the Trotter parameters."""
        theta_trotter: np.array = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)

    def chi(self, j: int, m: int) -> float:
        """
        Returns chi for two indices.

        param: j index
        param: m index
        """
        cc1 = self.theta[:, j].transpose() @ self.test @ self.theta[:, m]
        return 0.5 * cc1

    def chi_tensor(self, left_indices: np.array, right_indices: np.array) -> np.array:
        """
        Vectorized function to calculate all the chi coefficient at once.

        param: left_indices indices of the left tensor which give non-zero contributions in the calculation of chi.
        param: right_indices indices of the right tensor which give non-zero contributions in the calculation of chi.
        """
        theta_L: np.array = self.theta[:, left_indices]
        theta_R: np.array = self.theta[:, right_indices]
        res: np.array = np.tensordot(theta_L, self.test, [[0], [0]])
        res: np.array = np.tensordot(res, theta_R, [[1], [0]])
        return 0.5 * res
