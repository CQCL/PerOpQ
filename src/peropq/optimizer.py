from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]

from peropq.bch_optimized import VariationalNorm
from peropq.exact_norm import ExactUnitary
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary
from peropq.variational_unitary import VariationalUnitary

EMPTY_ARRAY = np.array([])


class Optimizer:
    """Class performing the optimizer."""

    def optimize(
        self,
        variational_unitary: VariationalUnitary,
        initial_guess: Sequence[float] = [],
    ) -> tuple[scipy.optimize.OptimizeResult, float]:
        """
        Perform the minimization.

        :param variational_unitary: ansatz used for optimization
        :param initial_guess: initial guess for the optimization. If not provided, use the parameters of the Trotterization instead
        :returns: the result of the optimization
        :returns: the perturbative 2-norm
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

    # ruff: noqa: PLR0913
    def optimize_arbitrary(
        self,
        variational_unitary: UnconstrainedVariationalUnitary,
        order: int,
        initial_guess: Sequence[float] = [],
        tol: float = 0,
        *,
        unconstrained: bool = False,
        cache: bool = False,
        init_variational_norm: VariationalNorm | None = None,
    ) -> (
        scipy.optimize.OptimizeResult
        | tuple[scipy.optimize.OptimizeResult, VariationalNorm]
    ):
        """
        Perform the minimization.

        :param variational_unitary: ansatz used for optimization
        :param initial_guess initial: guess for the optimization. If not provided, use the parameters of the Trotterization instead
        :returns: the result of the optimization
        :returns: the perturbative 2-norm
        """
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = variational_unitary.get_initial_trotter_vector()
            x0 = variational_unitary.flatten_theta(x0)
        if init_variational_norm is None:
            variational_norm = VariationalNorm(
                variational_unitary,
                order=order,
                unconstrained=unconstrained,
            )

            variational_norm.get_commutators()
            variational_norm.get_traces()
        else:
            variational_norm = init_variational_norm

        if tol == 0:
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm,
                x0,
            )
        else:
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm,
                x0,
                tol=tol,
            )
        # Set the theta of the variational_norm to the optimized result
        variational_unitary.theta = variational_unitary.unflatten_theta(
            optimized_results.x,
        )
        if cache:
            return optimized_results, variational_norm
        return optimized_results

    def optimize_exact(
        self,
        exact_unitary: ExactUnitary,
        initial_guess: npt.NDArray = EMPTY_ARRAY,
        tol: float = 0,
    ) -> scipy.optimize.OptimizeResult:
        """
        Optimize an instance of ExactUnitary.

        :param exact_unitary: to be optimized
        :param initial_guess: first guess to start the optimization
        :param tol: tolerance passed to the optimization function
        :returns: the optimization result form scipy.optimize.minimize
        """
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = exact_unitary.get_initial_trotter_vector()
            x0 = exact_unitary.flatten_theta(x0)

        return scipy.optimize.minimize(exact_unitary.get_exact_norm, x0, tol=tol)
