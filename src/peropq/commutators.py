from typing import Literal

from peropq.pauli import Pauli, PauliTensor


def paulis_commute(left_op: Pauli, right_op: Pauli) -> bool:
    """Return whether two Pauli gates commute.

    :param left_op The left side Pauli of the commutator
    :param right_op The right side of Pauli the commutator.

    :return: True if Paulis commute, False otherwise
    """
    if left_op == right_op:
        return True
    if left_op == Pauli.I:
        return True
    if right_op == Pauli.I:
        return True
    return False


def get_commutator_pauli_tensors(
    left_tens: PauliTensor,
    right_tens: PauliTensor,
) -> PauliTensor | Literal[0]:
    """Calculate the commutator of any two pauli tensors.

    :param left_tens left side of commutator
    :param right_tens right side of commutator.

    The general formula for A = c(A_1 x ... A_n), B = d(B_1 x ... x B_N)
    is [A, B] = 1 - (-1)^k cd(A_1 B_1 x ... x A_N B_N)
    where x is the tensor product, and k is the number of
    anti-commuting Pauli pairs.

    :returns: None if the tensors commute, otherwise the commutator as a Pauli tensor
    """
    new_tensor = PauliTensor()
    number_anti_commute = 0
    leftover_right_qubits = set(right_tens.qubit_pauli_map.keys())
    for qubit, left_pauli in left_tens.qubit_pauli_map.items():
        leftover_right_qubits.discard(qubit)
        right_pauli = right_tens.get_pauli(qubit)
        coeff, new_pauli = left_pauli * right_pauli
        if not paulis_commute(left_pauli, right_pauli):
            number_anti_commute += 1
        new_tensor.update({qubit: new_pauli}, coeff)
    # If the number of anti-commuting Paulis is even,
    # then the strings in total commute -> return None
    # this also covers the case that all Paulis commute
    if number_anti_commute % 2 == 0:
        return 0
    # Add factor two from 1 - (-1)^k = 2
    new_tensor.update({}, 2)
    # Deal with qubits in right_tens that aren't in left_tens
    # These will always commute, so don't affect the calculations above
    for qubit in leftover_right_qubits:
        pauli = right_tens.get_pauli(qubit)
        if pauli != Pauli.I:
            new_tensor.update({qubit: pauli})
    return new_tensor
