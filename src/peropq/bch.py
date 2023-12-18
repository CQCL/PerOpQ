from dataclasses import dataclass

import numpy as np

from peropq import commutators
from peropq.pauli import PauliString
from peropq.variational_unitary import VariationalUnitary

"""
BCH formula given by:
Z = X + Y + 1/2 [X,Y] + 1/12 [X,[X,Y]] - 1/12 [Y,[X,Y]] + ...
"""


@dataclass()
class NormTerm:
    """
    :param pauli_string
    :param order
    :param coefficient
    :param theta_indices representing the indices of the theta to be multiplied in front
    """

    pauli_string: PauliString
    order: int
    coefficient: float
    theta_indices: list[int]


def commutator(aterm: NormTerm, other: NormTerm) -> NormTerm:
    commutator_string = commutators.get_commutator_pauli_tensors(
        aterm.pauli_string, other.pauli_string
    )
    order: int = aterm.order + other.order
    coefficient: float = aterm.coefficient * other.coefficient
    theta_indices: list[tuple[int, int]] = aterm.theta_indices + other.theta_indices
    new_term = NormTerm(commutator_string, order, coefficient, theta_indices)
    return new_term


class VariationalNorm:
    def __init__(self, variational_unitary: VariationalUnitary, order: int):
        self.variational_unitary = variational_unitary
        self.term_norm = []
        self.order = order
        self.terms: list[NormTerm] = []
        self.coefficients: list[complex] = []

    def compute_commutator_sum(self, term_list1: NormTerm, term_list2: NormTerm):
        result_list: list[NormTerm] = []
        for term1 in term_list1:
            for term2 in term_list2:
                com_term: NormTerm = commutator(term1, term2)
                if com_term.pauli_string:
                    result_list.append(com_term)
        return result_list

    def add_term(self, new_term: NormTerm):  # TODO: take care of coefficient
        # First order:
        self.terms.append(new_term)
        # Second order:
        commutator_list: list[NormTerm] = self.compute_commutator_sum(
            [new_term], self.terms
        )
        for norm_term in commutator_list:
            if norm_term.order <= self.order:
                self.terms.append(norm_term)

    def get_commutators(self):
        layer = 0
        first_norm_term = NormTerm(
            pauli_string=self.variational_unitary.pauli_string_list[0],
            order=1,
            coefficient=1,
            theta_indices=[(0, 0)],
        )
        self.terms.append(first_norm_term)
        # First add all the terms for the first layer
        for i_term in range(1, self.variational_unitary.n_terms):
            new_term = NormTerm(
                pauli_string=self.variational_unitary.pauli_string_list[i_term],
                order=1,
                coefficient=1,
                theta_indices=[(0, i_term)],
            )
            self.add_term(new_term)
        # Loop over the higher layers to add all the terms and include them in the calculation
        layer += 1
        while layer < self.variational_unitary.depth:
            for i_term in range(self.variational_unitary.n_terms):
                new_term = NormTerm(
                    pauli_string=self.variational_unitary.pauli_string_list[i_term],
                    order=1,
                    coefficient=1,
                    theta_indices=[(layer, i_term)],
                )
                self.add_term(new_term)
            layer += 1

    def get_traces(self):
        self.indices: list[tuple] = []
        self.trace_list: list[float] = []
        for i_term, a_term in enumerate(self.terms):
            for j_term, another_term in enumerate(self.terms):
                product_commutators: PauliString = (
                    a_term.pauli_string * another_term.pauli_string
                )
                trace: complex = product_commutators.normalized_trace()
                if trace:
                    self.indices.append((i_term, j_term))
                    self.trace_list.append(trace)
        self.calculated_trace = True

    def calculate_norm(self, theta):
        if np.array(theta).shape[0] > self.variational_unitary.n_terms:
            theta_new = np.array(theta).reshape(
                (
                    self.variational_unitary.depth - 1,
                    self.variational_unitary.n_terms,
                ),
            )
            self.variational_unitary.update_theta(theta_new)
        if np.array(theta).shape[0] == self.variational_unitary.n_terms:
            theta_new = np.array(theta).reshape(
                (
                    1,
                    self.variational_unitary.n_terms,
                ),
            )
            self.variational_unitary.update_theta(theta_new)
        s_norm: float = 0
        for i_trace, trace in enumerate(self.trace_list):
            theta_coeff: float = 1.0
            left_term = self.terms[self.indices[i_trace][0]]
            right_term = self.terms[self.indices[i_trace][1]]
            if left_term.order > 1 and right_term.order > 1:
                for i_theta in left_term.theta_indices:
                    theta_coeff *= self.variational_unitary.theta[i_theta]
                    if left_term.order == 2:
                        theta_coeff *= 0.5
                    else:
                        print("order ", left_term.order, " is not implemented")
                        import sys

                        sys.exit()
                for i_theta in right_term.theta_indices:
                    theta_coeff *= self.variational_unitary.theta[i_theta]
                    if right_term.order == 2:
                        theta_coeff *= 0.5
                    else:
                        print("order ", right_term.order, " is not implemented")
                        import sys

                        sys.exit()
                s_norm += theta_coeff * trace
        return -1.0 * s_norm
