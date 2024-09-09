from dataclasses import dataclass

import numpy as np
from numpy import typing as npt
from numba import jit,njit

from peropq import commutators
# from peropq.pauli import PauliString
from peropq.pauli_bitstring import PauliString,pauli_mul_dagger
# from peropq.variational_unitary import VariationalUnitary
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary as VariationalUnitary
import copy
import rich

# The BCH formula is given by:
# Z = X + Y + 1/2 [X,Y] + 1/12 [X,[X,Y]] - 1/12 [Y,[X,Y]] + ...

@njit(cache=True)
def _get_non_zero_trace_indices(bitstrings:npt.NDArray)-> npt.NDArray:
    """
    Get all the the term indices which have non zero trace, i.e Tr(term[i]*term[j])!=0.
    Works directly at the bitstring level.
    bitstring: array containing bitstrings one which to compute the trace over.
    """
    indices_list:list = []
    for i in range(len(bitstrings)):
        for j in range(len(bitstrings)):
            if np.array_equal(bitstrings[i],bitstrings[j]):
                indices_list.append((i,j))
    return np.array(indices_list)

@jit(nopython=True)
def _loop_over_trace(trace_list:npt.NDArray[float],indices:npt.NDArray[int],theta:npt.NDArray[int],min_order:int,all_the_orders:npt.NDArray[int],all_the_indices:npt.NDArray[int],begin_list:npt.NDArray[int],end_list:npt.NDArray[int],all_the_coefficients:npt.NDArray[float])-> float:
    """
    Just in timed compiled function which computes the norm once all the traces are calculated.
    trace_list: array containing all the non zero traces
    indices: array containing the indices corresponding to every trace
    theta: variational parameters
    min_order: minimum order to be taken into the calculation (i.e. 0 for uncontrained and 1 for contrained)
    all_the_orders: array containaing the order of every term included in the terms
    all_the_indices: indices of the theta parameters for all the terms in the trace
    begin_list: array containing the first index to take into account in the trace for the variational parameters
    end_list: array containing the last index to take into account in the trace for the variational parameters
    all_the_coefficients: coefficient of the NormTerms which need to be taken into account in the trace
    """
    s_norm:float = 0.0
    for i_trace, trace in enumerate(trace_list):
        theta_coeff: float = 1.0
        i_l= indices[i_trace][0]
        i_r= indices[i_trace][1]
        left_term_order= all_the_orders[i_l]
        right_term_order= all_the_orders[i_r]
        left_term_theta_indices = all_the_indices[begin_list[i_l]:end_list[i_l]]
        right_term_theta_indices = all_the_indices[begin_list[i_r]:end_list[i_r]]
        right_term_coefficient = all_the_coefficients[i_r]
        left_term_coefficient = all_the_coefficients[i_l]
        if left_term_order >= min_order and right_term_order >= min_order:
            if left_term_order>0:
                for i_theta in left_term_theta_indices:
                    if 0.5 not in i_theta:
                        i1 = i_theta[0]
                        i2 = i_theta[1]
                        theta_coeff *= theta[i1,i2]
            if right_term_order>0:
                for i_theta in right_term_theta_indices:
                    if 0.5 not in i_theta:
                        i1 = i_theta[0]
                        i2 = i_theta[1]
                        theta_coeff *= theta[i1,i2]
            s_norm += (
                theta_coeff * left_term_coefficient * np.conjugate(right_term_coefficient) * trace
            )
    return s_norm

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
    theta_indices: list[tuple[int,int]]

    def _pretty_print(self):
        """ Useful function for testing"""
        pauli_print_string = ""
        pauli_print_string += str(self.pauli_string.coefficient)
        pauli_print_string += "*"
        for akey in self.pauli_string.qubit_pauli_map.keys():
            pauli_print_string += str(self.pauli_string.qubit_pauli_map[akey])
            pauli_print_string += str(akey) + " "
        rich.print(self.coefficient, "*", pauli_print_string, self.theta_indices)

def commutator(aterm: NormTerm, other: NormTerm) -> NormTerm:
    """Take the commutator between two NormTerm instances """
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
        """
        Class representing the perturbative norm with some variational parameters. It is used in the optimization process.
        variational_unitary: unitary that needs to be optimized
        order: order at which the optimization needs to be performed
        unconstrained: Whether the zero and first order are constrained to cancel each other 
        """
        self.variational_unitary = variational_unitary
        self.term_norm = []
        self.order = order
        self.terms: dict[list[NormTerm]] = {}  # indices:(order,term_index)
        for order_index in range(order):
            self.terms[order_index] = []
        self.unconstrained = unconstrained
        self.calculated_trace=False

    def compute_commutator_sum(
        self,
        term_list1: list[NormTerm],
        term_list2: list[NormTerm],
    ) -> list[NormTerm]:
        """
        Compute the commutator of two sums of Pauli strings: 
        [term_list1[0]+term_list1[1]+term_list1[2]+...,term_list2[0]+term_list2[1]+term_list2[2]+...]
        """
        result_list: list[NormTerm] = []
        for term1 in term_list1:
            for term2 in term_list2:
                com_term: NormTerm = commutator(term1, term2)
                if com_term.pauli_string:
                    result_list.append(com_term)
        return result_list

    def add_term(self, new_term: NormTerm):
        """
        Take into account a new term in the variational norm. This is currently implemented up to order 4.
        """  
        # Fourth order
        if self.order>=4:        
            #2nd order with terms of order 3
            x_y_4 = self.compute_commutator_sum(self.terms[2],[new_term])
            for aterm in x_y_4:
                aterm.coefficient = 0.5 * aterm.coefficient
            self.terms[3] += x_y_4
            # 3rd order with terms of order 2
            y_y_x_4 = self.compute_commutator_sum(
                [new_term],
                self.compute_commutator_sum([new_term], self.terms[1]),
            )
            for i_norm_term, norm_term in enumerate(y_y_x_4):
                norm_term.coefficient = (1.0 / 12.0) * norm_term.coefficient
            self.terms[3] += y_y_x_4
            # 3rd order with terms of order 1 and 2
            # order 1
            x_x_y_4_1 =self.compute_commutator_sum(
                            self.terms[0],
                            self.compute_commutator_sum(self.terms[1], [new_term]),
                        )
            for i_norm_term, norm_term in enumerate(x_x_y_4_1):
                norm_term.coefficient = (1.0 / 12.0) * norm_term.coefficient
            self.terms[3]+=x_x_y_4_1
            # order 2
            x_x_y_4_2 =self.compute_commutator_sum(
                            self.terms[1],
                            self.compute_commutator_sum(self.terms[0], [new_term]),
                        )
            for i_norm_term, norm_term in enumerate(x_x_y_4_2):
                norm_term.coefficient = (1.0 / 12.0) * norm_term.coefficient
            self.terms[3]+=x_x_y_4_2
            # 4th order with terms of order 1
            y_x_x_y_4 = self.compute_commutator_sum([new_term],
                                                    self.compute_commutator_sum(self.terms[0],
                                                                                self.compute_commutator_sum(self.terms[0],[new_term])))
            for i_norm_term,norm_term in enumerate(y_x_x_y_4):
                norm_term.coefficient = -(1.0/24.0)*norm_term.coefficient
            self.terms[3] +=y_x_x_y_4
        # Third order
        if self.order >= 3:
            # Commutators with terms of order 2:
            x_y_3 = self.compute_commutator_sum(self.terms[1],[new_term])
            for aterm in x_y_3:
                aterm.coefficient = 0.5 * aterm.coefficient
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
                norm_term.coefficient = (1.0 / 12.0) * norm_term.coefficient
            for i_norm_term, norm_term in enumerate(y_y_x_3):
                norm_term.coefficient = (1.0 / 12.0) * norm_term.coefficient
            self.terms[2] += x_x_y_3
            self.terms[2] += y_y_x_3
        # Second order:
        if self.order > 1:
            x_y_2: list[NormTerm] = self.compute_commutator_sum(
                self.terms[0],
                [new_term],
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
        """
        Get the commutators required to calculate the perturbative variational norm
        """
        layer:int = 0
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
                    order=0,
                    coefficient=1j
                    * self.variational_unitary.cjs[i_term]
                    * self.variational_unitary.time,
                    theta_indices=[(self.variational_unitary.depth+1,self.variational_unitary.n_terms+1)],
                )
                self.terms[0].append(new_term)
        rich.print("Calculated the commutators")

    def get_traces(self):
        """ Calculate all the traces of commutator needed for the perturbative variational norm. """ 
        self.indices: list[tuple] = []
        self.trace_list: list[float] = []
        self.all_the_terms: list[NormTerm] = []
        self.all_the_order: list[int] = []
        self.all_the_theta_indices: list = []
        all_the_indices = []
        self.begin_list = []
        self.end_list = []
        self.all_the_coefficients: list[float] = []
        all_the_bitstring_list:list[npt.NDarray] = []
        for order_index in range(self.order):
            for a_term in self.terms[order_index]:
                self.all_the_terms.append(a_term)
                all_the_bitstring_list.append(a_term.pauli_string.bit_string)
                self.all_the_order.append(a_term.order)
                self.begin_list.append(len(all_the_indices))
                self.all_the_theta_indices.append(a_term.theta_indices)
                for index_pair in a_term.theta_indices:
                    all_the_indices.append(index_pair)
                self.end_list.append(len(all_the_indices))
                self.all_the_coefficients.append(a_term.coefficient)
        self.all_the_indices = np.array(all_the_indices)
        # First calculate the traces which are not zero
        self.indices = _get_non_zero_trace_indices(np.array(all_the_bitstring_list))
        # Get the trace_list
        for index_tuple in self.indices:
            # Get the trace of the product -> coefficient1 * conj(coefficient2)
            self.trace_list.append(self.all_the_terms[index_tuple[0]].pauli_string.coefficient*np.conj(self.all_the_terms[index_tuple[1]].pauli_string.coefficient))
        rich.print("Calculated traces")
        self.calculated_trace = True

    def calculate_norm(self, theta:npt.NDArray[float])->float:
        """ Calculate the norm for a given variational parameter theta. """
        if np.array(theta).shape[0] > self.variational_unitary.n_terms:
            # TODO: Get rid of the try expect
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
            min_order:int = 0
        else:
            min_order:int = 1
        s_norm = _loop_over_trace(np.array(self.trace_list),np.array(self.indices),self.variational_unitary.theta,min_order,np.array(self.all_the_order),self.all_the_indices,np.array(self.begin_list),np.array(self.end_list),np.array(self.all_the_coefficients))
        return np.real(s_norm)

