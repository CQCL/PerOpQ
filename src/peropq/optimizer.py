from collections.abc import Sequence

import numpy as np
import scipy

from peropq.commutators import get_commutator_pauli_tensors
from peropq.hamiltonian import Hamiltonian
from peropq.uvar import Uvar


class Optimizer:
    def __init__(self, hamiltonian: Hamiltonian, R: int = 3, t: float = 1.0):
        self.hamiltonian = hamiltonian
        self.R = R
        self.t = t
        self.n_terms = hamiltonian.get_n_terms()
        self.cjs = hamiltonian.get_cjs()
        self.uvar = Uvar(n_terms=self.n_terms, R=self.R, cjs=self.cjs, t=self.t)
        self.uvar.set_theta_to_Trotter()
        self.cache = {}

        self.index_pairs = []
        commutators = []

        i = 0
        for j_prime, H_j_prime in enumerate(self.hamiltonian.pauli_string_list):
            for j in range(j_prime + 1, self.n_terms):
                H_j = self.hamiltonian.pauli_string_list[j]
                commutator = get_commutator_pauli_tensors(H_j, H_j_prime)
                if commutator:
                    self.index_pairs.append((j, j_prime))
                    commutators.append((i, commutator))
                    i += 1

        self.traces = []
        for j_, j_commutator in commutators:
            for k_, k_commutator in commutators:
                if j_ < k_:
                    continue
                product_commutators = j_commutator * k_commutator
                if product_commutators != 0:
                    trace = product_commutators.normalized_trace()
                    if trace:
                        fac = 1 if j_ == k_ else 2.0
                        self.traces.append(
                            (j_, k_, fac * product_commutators.normalized_trace()),
                        )

    def C2_squared(self, theta: Sequence[float] = []):
        if len(theta) != 0:
            theta_new = np.array(theta).reshape((self.uvar.R - 1, self.uvar.n_terms))
            self.uvar.update_theta(theta_new)
        c2_squared_result = 0
        chi = [self.uvar.chi(j, m) for j, m in self.index_pairs]
        for i, j, trace in self.traces:
            c2_squared_result -= chi[i] * chi[j] * trace
        return c2_squared_result

    def get_minumum_c2_squared(self, initial_guess: Sequence[float] = None):
        if initial_guess != None:
            x0 = initial_guess
        else:
            x0 = self.uvar.get_initial_trotter_vector()
            x0 = self.uvar.flatten_theta(x0)
        optimized_results = scipy.optimize.minimize(self.C2_squared, x0)
        return optimized_results, self.C2_squared(theta=optimized_results.x)
