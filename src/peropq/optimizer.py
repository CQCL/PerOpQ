from collections.abc import Sequence

import numpy as np
import scipy
from scipy.sparse import csr_array

from peropq.commutators import get_commutator_pauli_tensors
from peropq.hamiltonian import Hamiltonian
from peropq.variational_unitary import VariationalUnitary


class Optimizer:
    def __init__(self, variation_unitary: VariationalUnitary):
        self.hamiltonian = variation_unitary.hamiltonian
        self.R = variation_unitary.R
        self.t = variation_unitary.t
        self.n_terms = self.hamiltonian.get_n_terms()
        self.cjs = self.hamiltonian.get_cjs()
        self.variational_unitary = variation_unitary
        self.variational_unitary.set_theta_to_Trotter()
        self.cache = {}

        commutators = []
        index_pairs = []
        i = 0
        for j_prime, H_j_prime in enumerate(self.hamiltonian.pauli_string_list):
            for j in range(j_prime + 1, self.n_terms):
                H_j = self.hamiltonian.pauli_string_list[j]
                commutator = get_commutator_pauli_tensors(H_j, H_j_prime)
                if commutator:
                    index_pairs.append((j, j_prime))
                    commutators.append((i, commutator))
                    i += 1
        self.index_pairs = np.array(index_pairs)
        self.left_indices= np.unique(self.index_pairs[:,0])
        self.right_indices= np.unique(self.index_pairs[:,1])
        chi_tensor = self.variational_unitary.chi_tensor(self.left_indices,self.right_indices)
        self.trace_tensor= 1j*np.zeros((len(self.left_indices),len(self.right_indices),len(self.left_indices),len(self.right_indices)))
        self.traces = []
        self.chi_left = []
        self.chi_right = []
        for j_, j_commutator in commutators:
            for k_, k_commutator in commutators:
                if j_ < k_:
                    continue
                product_commutators = j_commutator * k_commutator
                if product_commutators != 0:
                    trace = product_commutators.normalized_trace()
                    if trace:
                        fac = 1 if j_ == k_ else 2.0
                        #Get the new indices
                        new_j=np.where(self.left_indices==self.index_pairs[j_,0])[0].item()
                        new_j_prime=np.where(self.right_indices==self.index_pairs[j_,1])[0].item()
                        new_k=np.where(self.left_indices==self.index_pairs[k_,0])[0].item()
                        new_k_prime=np.where(self.right_indices==self.index_pairs[k_,1])[0].item()
                        self.trace_tensor[new_j,new_j_prime,new_k,new_k_prime]=fac * product_commutators.normalized_trace()

    def C2_squared(self, theta: Sequence[float] = []):
        if len(theta) != 0:
            theta_new = np.array(theta).reshape((self.variational_unitary.R - 1, self.variational_unitary.n_terms))
            self.variational_unitary.update_theta(theta_new)
        chi_tensor = self.variational_unitary.chi_tensor(self.left_indices,self.right_indices)
        s1,s2,s3,s4 = self.trace_tensor.shape
        chi_tensor =chi_tensor.reshape((s1*s2,))
        trace_tensor = csr_array(self.trace_tensor.reshape((s1*s2,s3*s4)))
        return -chi_tensor.T@trace_tensor@chi_tensor

    def get_minumum_c2_squared(self, initial_guess: Sequence[float] = None):
        if initial_guess != None:
            x0 = initial_guess
        else:
            x0 = self.variational_unitary.get_initial_trotter_vector()
            x0 = self.variational_unitary.flatten_theta(x0)
        optimized_results = scipy.optimize.minimize(self.C2_squared, x0)
        return optimized_results, self.C2_squared(theta=optimized_results.x)
