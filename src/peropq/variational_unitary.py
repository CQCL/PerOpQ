from typing import TYPE_CHECKING

import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


import numpy as np

from peropq.hamiltonian import Hamiltonian


class VariationalUnitary:
    """class representing the variational unitary ansataz."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        number_of_layer: int,
        time: float,
    ) -> None:
        """
        Init function.

        :param hamiltonian Hamiltonian of which one is interested in the dynamics.
        :param R number of steps for the optimization.
        :param t final time to up to which one wants to perform the time evolution.
        """
        self.hamiltonian: Hamiltonian = hamiltonian
        self.n_terms: int = hamiltonian.get_n_terms()
        self.number_of_layer: int = number_of_layer
        self.theta: npt.NDArray = np.zeros((number_of_layer, self.n_terms))
        self.cjs: Sequence[complex] = hamiltonian.get_cjs()
        self.time: float = time
        self.test: npt.NDArray = np.zeros((number_of_layer, number_of_layer))
        for r in range(number_of_layer):
            for s in range(number_of_layer):
                self.test[r, s] = -1 if s > r else 1

    def update_theta(self, new_array: npt.NDArray) -> None:
        """
         Update theta ensuring that the condition Sum_i theta_i dt_i= is ensured.

        :param new_array the new array containing the variational parameters. It's shape must be (R - 1, n_terms).
        """
        if new_array.shape != (self.number_of_layer - 1, self.n_terms):
            error_message = "Wrong length provided."
            raise ValueError(error_message)
        for j in range(self.n_terms):
            for r in range(self.number_of_layer - 1):
                self.theta[r, j] = new_array[r, j]
            self.theta[self.number_of_layer - 1, j] = self.time * self.cjs[j]
            for r in range(self.number_of_layer - 1):
                self.theta[self.number_of_layer - 1, j] -= new_array[r, j]

    def get_initial_trotter_vector(self) -> npt.NDArray:
        """Get the variational parameters corresponding to the Trotterization. Useful to initialize the optimization."""
        theta_trotter: npt.NDArray = np.zeros((self.number_of_layer - 1, self.n_terms))
        for j in range(self.n_terms):
            for r in range(self.number_of_layer - 1):
                theta_trotter[r, j] = self.cjs[j] * self.time / self.number_of_layer
        return theta_trotter

    def flatten_theta(self, theta: npt.NDArray) -> npt.NDArray:
        """Returns the variational parameters as flatten (R-1)*n_terms array. Useful to pass to a minimization function."""
        return np.array(theta).reshape((self.number_of_layer - 1) * self.n_terms)

    def set_theta_to_trotter(self) -> None:
        """Sets the variational parameters to the Trotter parameters."""
        theta_trotter: npt.NDArray = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)

    def chi(self, j: int, m: int) -> float:
        """
        Returns chi for two indices.

        param: j index
        param: m index
        """
        cc1 = self.theta[:, j].transpose() @ self.test @ self.theta[:, m]
        return 0.5 * cc1

    def chi_tensor(
        self,
        left_indices: npt.NDArray,
        right_indices: npt.NDArray,
    ) -> npt.NDArray:
        """
        Vectorized function to calculate all the chi coefficient at once.

        param: left_indices indices of the left tensor which give non-zero contributions in the calculation of chi.
        param: right_indices indices of the right tensor which give non-zero contributions in the calculation of chi.
        """
        theta_left: npt.NDArray = self.theta[:, left_indices]
        theta_right: npt.NDArray = self.theta[:, right_indices]
        res: npt.NDArray = np.tensordot(theta_left, self.test, ([0], [0]))
        res = np.tensordot(res, theta_right, ([1], [0]))
        return 0.5 * res
