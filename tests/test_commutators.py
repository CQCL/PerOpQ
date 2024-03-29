import pytest
from peropq.commutators import get_commutator_pauli_tensors
from peropq.pauli import Pauli, PauliString


@pytest.mark.parametrize(
    ("tenslist1", "tenslist2", "coeff", "tenslistresult"),
    [
        ([Pauli.X, Pauli.X, Pauli.Z], [Pauli.X, Pauli.X, Pauli.Z], 0, []),
        (
            [Pauli.X, Pauli.X, Pauli.X],
            [Pauli.Y, Pauli.Y, Pauli.Y],
            2 * pow(1j, 3),
            [Pauli.Z, Pauli.Z, Pauli.Z],
        ),
        (
            [Pauli.X, Pauli.Z, Pauli.Z],
            [Pauli.X, Pauli.X, Pauli.Z],
            2 * 1j,
            [Pauli.I, Pauli.Y, Pauli.I],
        ),
        ([Pauli.X, Pauli.Z], [Pauli.X, Pauli.Y], 2 * (-1j), [Pauli.I, Pauli.X]),
        (
            [Pauli.X, Pauli.Y, Pauli.X],
            [Pauli.Y, Pauli.X, Pauli.Y],
            2 * (-1j) * pow(1j, 2),
            [Pauli.Z, Pauli.Z, Pauli.Z],
        ),
    ],
)
def test_tensor_commutators(
    tenslist1: list[Pauli],
    tenslist2: list[Pauli],
    coeff: complex,
    tenslistresult: list[Pauli],
) -> None:
    tens1 = PauliString.from_pauli_sequence(tenslist1)
    tens2 = PauliString.from_pauli_sequence(tenslist2)
    tens_result = (
        0 if coeff == 0 else PauliString.from_pauli_sequence(tenslistresult, coeff)
    )
    assert tens_result == get_commutator_pauli_tensors(tens1, tens2)
