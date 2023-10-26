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

    def C2_squared(self, theta: Sequence[float] = []):
        if len(theta) != 0:
            theta_new = np.array(theta).reshape((self.uvar.R - 1, self.uvar.n_terms))
            self.uvar.update_theta(theta_new)
        if self.cache == {}:
            print("starting cache")
            commutator_list = []
            indices_list = []
            for j, H_j in enumerate(self.hamiltonian.pauli_string_list):
                for j_prime, H_j_prime in enumerate(self.hamiltonian.pauli_string_list):
                    if j > j_prime:
                        commutator_list.append(
                            get_commutator_pauli_tensors(H_j, H_j_prime),
                        )
                        indices_list.append((j, j_prime))
            self.cache["commutator_list"] = commutator_list
            self.cache["trace_list"] = []
            self.cache["indices_list"] = []
            for j, j_commutator in enumerate(self.cache["commutator_list"]):
                for k, k_commutator in enumerate(self.cache["commutator_list"]):
                    product_commutators = j_commutator * k_commutator
                    if product_commutators != 0 and product_commutators.coefficient != 0:
                        if product_commutators.normalized_trace() != 0:
                            self.cache["trace_list"].append(product_commutators.normalized_trace())
                            self.cache["indices_list"].append((indices_list[j][0],indices_list[j][1],indices_list[k][0],indices_list[k][1]))
            print("end cache")
        c2_squared_result = 0
        for i,trace in enumerate(self.cache["trace_list"]): 
            c2_squared_result -= (
                self.uvar.chi(
                    self.cache["indices_list"][i][0],
                    self.cache["indices_list"][i][1],
                )
                * self.uvar.chi(
                    self.cache["indices_list"][i][2],
                    self.cache["indices_list"][i][3],
                )
                * trace
            )
        return c2_squared_result

    def get_minumum_c2_squared(self, initial_guess: Sequence[float] = None):
        if initial_guess != None:
            x0 = initial_guess
        else:
            x0 = self.uvar.get_initial_trotter_vector()
            x0 = self.uvar.flatten_theta(x0)
        optimized_results = scipy.optimize.minimize(self.C2_squared, x0)
        return optimized_results, self.C2_squared(theta=optimized_results.x)
