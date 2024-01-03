from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]

from peropq.bch import VariationalNorm
from peropq.hamiltonian import Hamiltonian
from peropq.variational_unitary import VariationalUnitary


class Optimizer:
    """Class performing the optimizer."""

    def optimize(
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

    def optimize_arbitrary(
        self,
        variational_unitary: VariationalUnitary,
        order: float,
        initial_guess: Sequence[float] = [],
        tol: float = 0,
        unconstrained=False,
    ) -> scipy.optimize.OptimizeResult:
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
        variational_norm = VariationalNorm(
            variational_unitary, order=order, unconstrained=unconstrained
        )
        variational_norm.get_commutators()
        print("terms order 0 ")
        for aterm in variational_norm.terms[0]:
            aterm.pretty_print()

        print("terms order 1 ")
        for aterm in variational_norm.terms[1]:
            aterm.pretty_print()
        variational_norm.get_traces()
        if tol == 0:
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm, x0
            )
        else:
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm, x0, tol=tol
            )
        return optimized_results

    def optimize_steps(
        self,
        hamiltonian: Hamiltonian,
        final_time: float,
        order: float,
        dt_optimize: float,
    ) -> tuple[scipy.optimize.OptimizeResult, VariationalUnitary]:
        """
        Try to optimize with increasing time steps instead of using Trotter
        """
        variational_unitary = VariationalUnitary(hamiltonian, 3, dt_optimize)
        x0 = variational_unitary.get_initial_trotter_vector()
        x0 = variational_unitary.flatten_theta(x0)
        variational_norm = VariationalNorm(variational_unitary, order=2)
        variational_norm.get_commutators()
        variational_norm.get_traces()
        optimized_results = scipy.optimize.minimize(variational_norm.calculate_norm, x0)
        evolve_to_time = 2 * dt_optimize
        while evolve_to_time < final_time:
            print("evolve_to_time ", evolve_to_time)
            variational_unitary = VariationalUnitary(hamiltonian, 3, evolve_to_time)
            x0 = variational_unitary.get_initial_trotter_vector()
            x0 = variational_unitary.flatten_theta(x0)
            variational_norm = VariationalNorm(variational_unitary, order=2)
            variational_norm.get_commutators()
            variational_norm.get_traces()
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm,
                optimized_results.x,
            )
            evolve_to_time += dt_optimize
        return optimized_results, variational_unitary
