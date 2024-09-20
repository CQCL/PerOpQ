from typing import Literal

from peropq.pauli_bitstring import PauliString


def get_commutator_pauli_tensors(
    left_pauli_string: PauliString,
    right_pauli_string: PauliString,
) -> PauliString | Literal[0]:
    """Calculate the commutator of any two pauli tensors.

    :param left_pauli_string: left side of commutator
    :param right_pauli_string: right side of commutator.

    The general formula for A = c(A_1 x ... A_n), B = d(B_1 x ... x B_N)
    is [A, B] = 1 - (-1)^k cd(A_1 B_1 x ... x A_N B_N)
    where x is the tensor product, and k is the number of
    anti-commuting Pauli pairs.

    :returns: None if the strings commute, otherwise the commutator as a PauliString
    """
    if left_pauli_string.commutes_with(right_pauli_string):
        return 0
    return 2 * left_pauli_string * right_pauli_string
