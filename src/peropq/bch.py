from dataclasses import dataclass

import numpy as np

from peropq import commutators
from peropq.pauli import PauliString
from peropq.variational_unitary import VariationalUnitary
import copy

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
    coefficient: complex
    theta_indices: list[int]

    def pretty_print(self):
        pauli_print_string = ""
        pauli_print_string += str(self.pauli_string.coefficient)
        pauli_print_string += "*"
        for akey in self.pauli_string.qubit_pauli_map.keys():
            pauli_print_string += str(self.pauli_string.qubit_pauli_map[akey])
            pauli_print_string += str(akey) + " "
        print(self.coefficient, "*", pauli_print_string, self.theta_indices)


def commutator(aterm: NormTerm, other: NormTerm) -> NormTerm:
    commutator_string = commutators.get_commutator_pauli_tensors(
        aterm.pauli_string,
        other.pauli_string,
    )
    order: int = aterm.order + other.order
    coefficient: float = aterm.coefficient * other.coefficient
    theta_indices: list[tuple[int, int]] = aterm.theta_indices + other.theta_indices
    new_term = NormTerm(commutator_string, order, coefficient, theta_indices)
    return new_term


class VariationalNorm:
    def __init__(
        self,
        variational_unitary: VariationalUnitary,
        order: int,
        unconstrained=False,
    ):
        self.variational_unitary = variational_unitary
        self.term_norm = []
        self.order = order
        self.terms: dict[list[NormTerm]] = {}  # indices:(order,term_index)
        for order_index in range(order):
            self.terms[order_index] = []
        self.unconstrained = unconstrained

    def compute_commutator_sum(
        self,
        term_list1: list[NormTerm],
        term_list2: list[NormTerm],
    ) -> list[NormTerm]:
        result_list: list[NormTerm] = []
        for term1 in term_list1:
            for term2 in term_list2:
                com_term: NormTerm = commutator(term1, term2)
                if com_term.pauli_string:
                    result_list.append(com_term)
        return result_list

    def add_term(self, new_term: NormTerm):  # TODO: take care of coefficient
        # Fourth order
        if self.order>=4:        
            #2nd order with terms of order 3
            x_y_4 = self.compute_commutator_sum([new_term], self.terms[2])
            for aterm in x_y_4:
                aterm.coefficient = +0.5 * aterm.coefficient
            self.terms[3] += x_y_4
            # 3rd order with terms of order 2
            x_x_y_4 = self.compute_commutator_sum(
                [new_term],
                self.compute_commutator_sum([new_term], self.terms[1]),
            )
            y_y_x_4 = []
            y_y_x_4 += self.compute_commutator_sum(
                self.terms[1],
                self.compute_commutator_sum(self.terms[1], [new_term]),
            )
            for i_norm_term, norm_term in enumerate(x_x_y_4):
                norm_term.coefficient = -(1.0 / 12.0) * norm_term.coefficient
            for i_norm_term, norm_term in enumerate(y_y_x_4):
                norm_term.coefficient = -(1.0 / 12.0) * norm_term.coefficient
            self.terms[3] += x_x_y_4
            self.terms[3] += y_y_x_4
            #4th order with terms of order 1
            y_x_x_y_4 = self.compute_commutator_sum([new_term],self.compute_commutator_sum(self.terms[0],self.compute_commutator_sum(self.terms[0],[new_term])))
            for i_norm_term,norm_term in enumerate(y_x_x_y_4):
                norm_term.coefficient = (-1.0/24.0)*norm_term.coefficient
            self.terms[3] +=y_x_x_y_4
        # Third order
        if self.order >= 3:
            ######################
            # Commutators with terms of order 2:
            x_y_3 = self.compute_commutator_sum([new_term], self.terms[1])
            for aterm in x_y_3:
                aterm.coefficient = +0.5 * aterm.coefficient
            self.terms[2] += x_y_3
            x_x_y_3 = self.compute_commutator_sum(
                [new_term],
                self.compute_commutator_sum([new_term], self.terms[0]),
            )
            y_y_x_3 = []
            y_y_x_3 += self.compute_commutator_sum(
                self.terms[0],
                self.compute_commutator_sum(self.terms[0], [new_term]),
            )
            for i_norm_term, norm_term in enumerate(x_x_y_3):
                norm_term.coefficient = +(1.0 / 12.0) * norm_term.coefficient
            for i_norm_term, norm_term in enumerate(y_y_x_3):
                norm_term.coefficient = +(1.0 / 12.0) * norm_term.coefficient
            self.terms[2] += x_x_y_3
            self.terms[2] += y_y_x_3
        # Second order:
        if self.order > 1:
            x_y_2: list[NormTerm] = self.compute_commutator_sum(
                [new_term],
                self.terms[0],
            )
            # Do some sanity check
            for norm_term in x_y_2:
                norm_term.coefficient = -0.5
                if norm_term.order != 2:
                    message = "second order contained terms of higher order"
                    raise RuntimeError(message)
            self.terms[1] += x_y_2
        # First order:
        self.terms[0].append(new_term)

    def get_commutators(self):
        layer = 0
        first_norm_term = NormTerm(
            pauli_string=self.variational_unitary.pauli_string_list[0],
            order=1,
            coefficient=-1j,
            theta_indices=[(0, 0)],
        )
        self.terms[0].append(first_norm_term)
        # First add all the terms for the first layer
        for i_term in range(1, self.variational_unitary.n_terms):
            new_term = NormTerm(
                pauli_string=self.variational_unitary.pauli_string_list[i_term],
                order=1,
                coefficient=-1j,
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
                    coefficient=-1j,
                    theta_indices=[(layer, i_term)],
                )
                self.add_term(new_term)
            layer += 1
        # Add the Trotter terms with a minus for the unconstrained unitary
        # Add this only in the trace!!
        if self.unconstrained:
            for i_term in range(self.variational_unitary.n_terms):
                new_term = NormTerm(
                    pauli_string=self.variational_unitary.pauli_string_list[i_term],
                    order=1,
                    coefficient=1j
                    * self.variational_unitary.cjs[i_term]
                    * self.variational_unitary.time,
                    theta_indices=[(None, None)],
                )
                self.terms[0].append(new_term)
        # for aterm in self.terms[2]:
        #     aterm.pretty_print()

    def get_traces(self):
        def string_string_dagger(string1,string2):
            string_dagger:PauliString = copy.deepcopy(string2)
            string_dagger.coefficient = np.conjugate(string_dagger.coefficient)
            return string1*string_dagger
        self.indices: list[tuple] = []
        self.trace_list: list[float] = []
        self.all_the_terms: list[NormTerm] = []
        for order_index in range(self.order):
            for a_term in self.terms[order_index]:
                self.all_the_terms.append(a_term)
                # a_term.pretty_print()
        for i_term, a_term in enumerate(self.all_the_terms):
            for j_term, another_term in enumerate(self.all_the_terms):
                product_commutators: PauliString = (
                    # a_term.pauli_string * another_term.pauli_string
                    string_string_dagger(a_term.pauli_string,another_term.pauli_string)
                )
                trace: complex = product_commutators.normalized_trace()
                if trace:
                    # print("a_term ",a_term)
                    # print("another_term ",another_term)
                    # print("trace ",trace)
                    self.indices.append((i_term, j_term))
                    self.trace_list.append(trace)
        self.calculated_trace = True

    def calculate_norm(self, theta):
        if np.array(theta).shape[0] > self.variational_unitary.n_terms:
            # TODO: write a function unflatten theta
            try:
                theta_new = np.array(theta).reshape(
                    (
                        self.variational_unitary.depth - 1,
                        self.variational_unitary.n_terms,
                    ),
                )
            except:
                theta_new = np.array(theta).reshape(
                    (
                        self.variational_unitary.depth,
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
        if self.unconstrained:
            min_order = 0
        else:
            min_order = 1
        for i_trace, trace in enumerate(self.trace_list):
            theta_coeff: float = 1.0
            left_term = self.all_the_terms[self.indices[i_trace][0]]
            right_term = self.all_the_terms[self.indices[i_trace][1]]
            if left_term.order > min_order and right_term.order > min_order:
                for i_theta in left_term.theta_indices:
                    if None not in i_theta:
                        theta_coeff *= self.variational_unitary.theta[i_theta]
                for i_theta in right_term.theta_indices:
                    if None not in i_theta:
                        theta_coeff *= self.variational_unitary.theta[i_theta]
                # print("left right contribution")
                # print(left_term)
                # print(right_term)
                # print(left_term.coefficient * right_term.coefficient * trace)
                # breakpoint()
                s_norm += (
                    # theta_coeff * left_term.coefficient * right_term.coefficient * trace
                    theta_coeff * left_term.coefficient * np.conjugate(right_term.coefficient) * trace
                )
        return +s_norm

    def get_analytical_gradient(self):
        def get_i_derivative(theta_indices, theta_index):
            position_list = []
            for i, index_tuple in enumerate(theta_indices):
                if theta_index == index_tuple:
                    position_list.append(i)
            indices_list_list = []
            if len(position_list) > 0:
                for pos in position_list:
                    indices_list = []
                    for i, index_tuple in enumerate(theta_indices):
                        if i != pos:
                            indices_list.append(index_tuple)
                    indices_list_list.append(indices_list)
            return indices_list_list

        if self.unconstrained:
            min_order = 0
        else:
            min_order = 1
        s_norm: float = 0
        self.gradient_theta_indices: list[list[list[int]]] = []
        self.gradient_vector_constants: list[list[float]] = []
        #########################################
        # grad_{r,n} = sum_k sum_i [theta_{k,i,1}... theta_{k,i,n}] trace_k
        # where i stands for the different derivatives of the index theta_{r,n}
        # and k stands for the different terms of the norm
        # self.gradient_vector_constants[counter,k]=trace_k
        ########################################
        counter = 0
        for depth_index in range(self.variational_unitary.depth):
            for term_index in range(self.variational_unitary.n_terms):
                gradient_indices_list = []
                trace_k_list = []
                for i_trace, trace in enumerate(self.trace_list):
                    left_term = self.all_the_terms[self.indices[i_trace][0]]
                    right_term = self.all_the_terms[self.indices[i_trace][1]]
                    if (
                        (left_term.order > min_order)
                        and (right_term.order > min_order)
                    ):
                        #TODO: just try to remove the None indices in this list and that should fix it
                        indices_list_list = get_i_derivative(
                            left_term.theta_indices + right_term.theta_indices,
                            (depth_index, term_index),
                        )
                        gradient_indices_list.append(indices_list_list)
                        trace_k_list.append(
                            trace * left_term.coefficient * right_term.coefficient
                        )
                counter += 1
                self.gradient_theta_indices.append(gradient_indices_list)
                self.gradient_vector_constants.append(trace_k_list)

    def get_numerical_gradient(self, theta_flat):
        theta = theta_flat.reshape(
            (self.variational_unitary.depth, self.variational_unitary.n_terms)
        )
        self.variational_unitary.theta = theta
        gradient = np.zeros(
            self.variational_unitary.depth * self.variational_unitary.n_terms
        )
        for i, indices_list_list in enumerate(
            self.gradient_theta_indices
        ):  # coordinate
            for k, indices_list in enumerate(indices_list_list):  # trace
                for l, derivative_indices in enumerate(
                    indices_list
                ):  # sum of terms of the derivative
                    product_theta = self.gradient_vector_constants[i][k]
                    for index_tuple in derivative_indices:
                        if index_tuple!=(None,None):
                            product_theta *= self.variational_unitary.theta[index_tuple]
                    gradient[i] += product_theta
        return -gradient