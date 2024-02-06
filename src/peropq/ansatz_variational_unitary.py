import numpy as np
import numpy.typing as npt

from peropq.variational_unitary import VariationalUnitary
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary


class AnsatzVariationalUnitary(UnconstrainedVariationalUnitary):
    def update_theta(self, new_array: npt.NDArray) -> None:
        """
         Update theta

        :param new_array the new array containing the variational parameters. It's shape must be (R,  n_terms).
        """
        if new_array.shape != (self.depth, self.n_terms):
            if self.depth == 1 and new_array.shape == (1, self.n_terms):
                pass
            else:
                error_message = "Wrong length provided."
                raise ValueError(error_message)
        self.theta = new_array

    def flatten_theta(self, theta: npt.NDArray) -> npt.NDArray:
        """Returns the variational parameters as flatten depth*n_terms array. Useful to pass to a minimization function."""
        if self.depth > 1:
            return np.array(theta).reshape(self.depth * self.n_terms)
        else:
            return np.array(theta).reshape(self.n_terms)

    def get_initial_trotter_vector(self) -> npt.NDArray:
        """Get the variational parameters corresponding to the Trotterization. Useful to initialize the optimization."""
        if self.depth > 1:
            theta_trotter: npt.NDArray = np.zeros((self.depth, self.n_terms))
            for j in range(self.n_terms):
                for r in range(self.depth):
                    theta_trotter[r, j] = self.cjs[j] * self.time / self.depth
        else:
            theta_trotter: npt.NDArray = np.zeros((self.depth, self.n_terms))
            for j in range(self.n_terms):
                theta_trotter[0, j] = self.cjs[j] * self.time
        print("theta trotter initial ", theta_trotter.shape)
        return theta_trotter

    def set_theta_to_trotter(self) -> None:
        """Sets the variational parameters to the Trotter parameters."""
        theta_trotter: npt.NDArray = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)
