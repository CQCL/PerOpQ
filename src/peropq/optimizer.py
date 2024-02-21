from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import-untyped]

from peropq.ansatz_bch import AnsatzVariationalNorm
from peropq.bch import VariationalNorm
from peropq.hamiltonian import Hamiltonian
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary
from peropq.variational_unitary import VariationalUnitary
from peropq.exact_norm import ExactUnitary


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
            variational_unitary,
            order=order,
            unconstrained=unconstrained,
        )
        variational_norm.get_commutators()
        variational_norm.get_traces()
        variational_norm.get_analytical_gradient()
        if tol == 0:
            variational_norm.get_analytical_gradient()
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm,
                x0,
                jac=variational_norm.get_numerical_gradient,
            )
        else:
            optimized_results = scipy.optimize.minimize(
                variational_norm.calculate_norm,
                x0,
                tol=tol,
                jac=variational_norm.get_numerical_gradient,
            )
        return optimized_results

    def optimize_exact(
        self,
        exact_unitary: ExactUnitary,
        initial_guess: Sequence[float] = [],
        tol: float = 0,
    ):
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = variational_unitary.get_initial_trotter_vector()
            x0 = variational_unitary.flatten_theta(x0)
        
        return scipy.optimize.minimize(exact_unitary.get_exact_norm, x0)
        

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

    def optimize_ansatz(
        self,
        variational_unitary: UnconstrainedVariationalUnitary,
        order: float,
        initial_guess: Sequence[float],
        hamiltonian: Hamiltonian,
        tol: float = 0,
    ) -> scipy.optimize.OptimizeResult:
        if len(initial_guess) != 0:
            x0: npt.NDArray = np.array(initial_guess)
        else:
            x0 = variational_unitary.get_initial_trotter_vector()
            x0 = variational_unitary.flatten_theta(x0)
        variational_norm = AnsatzVariationalNorm(
            variational_unitary,
            order=order,
            hamiltonian=hamiltonian,
        )
        variational_norm.get_commutators()
        variational_norm.get_traces()
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
        return optimized_results
