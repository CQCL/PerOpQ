from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]

from peropq.variational_unitary import VariationalUnitary


class Optimizer:
    """Class performing the optimizer."""

    def get_minumum_c2_squared(
        self,
        variational_unitary: VariationalUnitary,
        initial_guess: Sequence[float] = [],
    ) -> tuple[scipy.optimize.OptimizeResult, float]:
        """
        Perform the minimization.

        param: variational_unitary ansatz used for optimization
        param: initial_guess initial guess for the optimization. If not provided, use the parameters of the Trotterization instead
        returns: the result of the optimization
        returns: the perturbative 2-norm
        """
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = variational_unitary.get_initial_trotter_vector()
            x0 = variational_unitary.flatten_theta(x0)
        if not variational_unitary.trace_calculated:
            variational_unitary.calculate_traces()
        optimized_results = scipy.optimize.minimize(variational_unitary.c2_squared, x0)
        return optimized_results, variational_unitary.c2_squared(
            theta=optimized_results.x,
        )
