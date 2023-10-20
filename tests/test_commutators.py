import pytest
from peropq.commutators import get_commutator_pauli_tensors, paulis_commute
from peropq.pauli import Pauli, PauliTensor


def test_pauls_commute() -> None:
    assert paulis_commute(Pauli.I, Pauli.I)
    assert paulis_commute(Pauli.X, Pauli.X)
    assert paulis_commute(Pauli.Y, Pauli.Y)
    assert paulis_commute(Pauli.Z, Pauli.Z)
    assert paulis_commute(Pauli.I, Pauli.X)
    assert paulis_commute(Pauli.X, Pauli.I)
    assert paulis_commute(Pauli.I, Pauli.Y)
    assert paulis_commute(Pauli.Y, Pauli.I)
    assert paulis_commute(Pauli.I, Pauli.Z)
    assert paulis_commute(Pauli.Z, Pauli.I)
    assert not paulis_commute(Pauli.X, Pauli.Y)
    assert not paulis_commute(Pauli.Y, Pauli.X)
    assert not paulis_commute(Pauli.Y, Pauli.Z)
    assert not paulis_commute(Pauli.Z, Pauli.Y)
    assert not paulis_commute(Pauli.Z, Pauli.X)
    assert not paulis_commute(Pauli.X, Pauli.Z)


@pytest.mark.parametrize(
    ("tenslist1", "tenslist2", "coeff", "tenslistresult"),
    [
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
        ([Pauli.X, Pauli.X, Pauli.Z], [Pauli.X, Pauli.X, Pauli.Z], 0, None),
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
    tens1 = PauliTensor.from_pauli_sequence(tenslist1)
    tens2 = PauliTensor.from_pauli_sequence(tenslist2)
    tens_result = (
        None
        if not tenslistresult
        else PauliTensor.from_pauli_sequence(tenslistresult, coeff)
    )
    assert tens_result == get_commutator_pauli_tensors(tens1, tens2)
