import numba
from numba import jit
import numpy as np

@jit(nopython=True)
def loop_over_trace(trace_list,all_the_terms,indices,variational_unitary,min_order):
    s_norm = 0.0
    for i_trace, trace in enumerate(trace_list):
        theta_coeff: float = 1.0
        left_term = all_the_terms[indices[i_trace][0]]
        right_term = all_the_terms[indices[i_trace][1]]
        if left_term.order > min_order and right_term.order > min_order:
            for i_theta in left_term.theta_indices:
                if None not in i_theta:
                    theta_coeff *= variational_unitary.theta[i_theta]
            for i_theta in right_term.theta_indices:
                if None not in i_theta:
                    theta_coeff *= variational_unitary.theta[i_theta]
            s_norm += (
                theta_coeff * left_term.coefficient * np.conjugate(right_term.coefficient) * trace
            )
    return s_norm

