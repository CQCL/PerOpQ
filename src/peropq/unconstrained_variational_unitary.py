import numpy as np
import numpy.typing as npt

from peropq.variational_unitary import VariationalUnitary


class UnconstrainedVariationalUnitary(VariationalUnitary):
    """class representing the variational unitary ansataz, with no prior constraint on the variational parameters."""

    def update_theta(self, new_array: npt.NDArray) -> None:
        """
         Update theta.

        :param new_array: the new array containing the variational parameters. It's shape must be (R,  n_terms).
        """
        if new_array.shape != (self.depth, self.n_terms):
            if self.depth == 1 and new_array.shape == (1, self.n_terms):
                pass
            else:
                error_message = (
                    "Wrong length provided. Shape is"
                    + str(new_array.shape)
                    + " required is "
                    + str((1, self.n_terms))
                )
                raise ValueError(error_message)
        self.theta = new_array

    def flatten_theta(self, theta: npt.NDArray) -> npt.NDArray:
        """Returns an input theta as flatten depth*n_terms array. Useful to pass to a minimization function."""
        if self.depth > 1:
            return np.array(theta).reshape(self.depth * self.n_terms)
        return np.array(theta).reshape(self.n_terms)

    def unflatten_theta(self, flat_theta: npt.NDArray) -> npt.NDArray:
        """Returns the flattened variational parameters as an array with shape (depth,n_terms)."""
        return np.array(flat_theta).reshape((self.depth, self.n_terms))

    def get_flattened_theta(self) -> npt.NDArray:
        """Return the flatten current variational parameters."""
        return self.flatten_theta(self.theta)

    def get_initial_trotter_vector(self) -> npt.NDArray:
        """Get the variational parameters corresponding to the Trotterization. Useful to initialize the optimization."""
        theta_trotter: npt.NDArray
        if self.depth > 1:
            theta_trotter = np.zeros((self.depth, self.n_terms))
            for j in range(self.n_terms):
                for r in range(self.depth):
                    theta_trotter[r, j] = np.real(self.cjs[j]) * self.time / self.depth
        else:
            theta_trotter = np.zeros((self.depth, self.n_terms))
            for j in range(self.n_terms):
                theta_trotter[0, j] = np.real(self.cjs[j]) * self.time
        return theta_trotter

    def set_theta_to_trotter(self) -> None:
        """Sets the variational parameters to the Trotter parameters."""
        theta_trotter: npt.NDArray = self.get_initial_trotter_vector()
        self.update_theta(theta_trotter)
