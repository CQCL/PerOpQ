import pytest
from peropq.pauli import Pauli, PauliString


def test_pauli() -> None:
    """Simple test that pauli values are correct strings."""
    assert Pauli.I.value == "I"
    assert Pauli.X.value == "X"
    assert Pauli.Y.value == "Y"
    assert Pauli.Z.value == "Z"


def test_pauli_mult() -> None:
    assert Pauli.I * Pauli.I == (1, Pauli.I)
    assert Pauli.X * Pauli.X == (1, Pauli.I)
    assert Pauli.Y * Pauli.Y == (1, Pauli.I)
    assert Pauli.Z * Pauli.Z == (1, Pauli.I)
    assert Pauli.I * Pauli.X == (1, Pauli.X)
    assert Pauli.X * Pauli.I == (1, Pauli.X)
    assert Pauli.I * Pauli.Y == (1, Pauli.Y)
    assert Pauli.Y * Pauli.I == (1, Pauli.Y)
    assert Pauli.I * Pauli.Z == (1, Pauli.Z)
    assert Pauli.Z * Pauli.I == (1, Pauli.Z)
    assert Pauli.X * Pauli.Y == (complex(0, 1), Pauli.Z)
    assert Pauli.Y * Pauli.X == (complex(0, -1), Pauli.Z)
    assert Pauli.Y * Pauli.Z == (complex(0, 1), Pauli.X)
    assert Pauli.Z * Pauli.Y == (complex(0, -1), Pauli.X)
    assert Pauli.Z * Pauli.X == (complex(0, 1), Pauli.Y)
    assert Pauli.X * Pauli.Z == (complex(0, -1), Pauli.Y)


def test_paulis_commute() -> None:
    assert Pauli.I.commutes_with(Pauli.I)
    assert Pauli.X.commutes_with(Pauli.X)
    assert Pauli.Y.commutes_with(Pauli.Y)
    assert Pauli.Z.commutes_with(Pauli.Z)
    assert Pauli.I.commutes_with(Pauli.X)
    assert Pauli.X.commutes_with(Pauli.I)
    assert Pauli.I.commutes_with(Pauli.Y)
    assert Pauli.Y.commutes_with(Pauli.I)
    assert Pauli.I.commutes_with(Pauli.Z)
    assert Pauli.Z.commutes_with(Pauli.I)
    assert not Pauli.X.commutes_with(Pauli.Y)
    assert not Pauli.Y.commutes_with(Pauli.X)
    assert not Pauli.Y.commutes_with(Pauli.Z)
    assert not Pauli.Z.commutes_with(Pauli.Y)
    assert not Pauli.Z.commutes_with(Pauli.X)
    assert not Pauli.X.commutes_with(Pauli.Z)


def test_pauli_string_equality() -> None:
    string1 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 2j)
    string2 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 2.03j)
    string3 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 2.00000000000003j)
    string4 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 2.03)

    assert string1 != 5
    assert string1 != string2
    assert string2 != string3
    assert string1 == string3
    assert string2 != string4

    string5 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Z], 2j)
    string6 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y, Pauli.Z], 2j)
    string7 = PauliString(string1.qubit_pauli_map.copy(), string1.coefficient)
    string7.qubit_pauli_map[2] = Pauli.I

    assert string1 != string5
    assert string1 != string6
    assert string1 == string7


def test_get_pauli() -> None:
    string1 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 2j)
    with pytest.raises(ValueError):
        string1.get_pauli(-4)
    assert string1.get_pauli(0) == Pauli.X
    assert string1.get_pauli(1) == Pauli.Y
    assert string1.get_pauli(2) == Pauli.I
    assert string1.get_pauli(200) == Pauli.I


def test_pauli_string_init() -> None:
    xy_string = PauliString.from_pauli_sequence(paulis=[Pauli.X, Pauli.Y])
    xy_string_prime = PauliString(
        qubit_pauli_map={0: Pauli.X, 1: Pauli.Y},
        coefficient=1,
    )
    assert xy_string == xy_string_prime
    ixy_string = PauliString.from_pauli_sequence(
        paulis=[Pauli.X, Pauli.Y],
        start_qubit=1,
    )
    ixy_string_prime = PauliString(
        qubit_pauli_map={0: Pauli.I, 1: Pauli.X, 2: Pauli.Y},
        coefficient=1,
    )
    assert ixy_string == ixy_string_prime


def test_pauli_string_prune() -> None:
    paulis = [
        Pauli.X,
        Pauli.X,
        Pauli.I,
        Pauli.X,
        Pauli.X,
        Pauli.Y,
        Pauli.X,
        Pauli.X,
        Pauli.I,
        Pauli.X,
        Pauli.X,
        Pauli.I,
        Pauli.X,
    ]
    expected_qubit_pauli_map = {
        q: pauli for q, pauli in enumerate(paulis) if pauli != Pauli.I
    }
    pauli_string = PauliString.from_pauli_sequence(paulis)
    assert pauli_string.qubit_pauli_map == expected_qubit_pauli_map
    pauli_string.update({0: Pauli.I, 2: Pauli.I})
    pauli_string.prune()
    assert 0 not in pauli_string.qubit_pauli_map
    assert 2 not in pauli_string.qubit_pauli_map


def test_mult_with_scalar() -> None:
    pauli_string = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y])
    assert (pauli_string * 2j).coefficient == 2j
    assert (2j * pauli_string).coefficient == 2j
    assert (pauli_string * 2).coefficient == 2
    assert (2 * pauli_string).coefficient == 2


@pytest.mark.parametrize(
    ("pauli_list1", "pauli_list2", "coeff_result", "pauli_list_result"),
    [
        ([Pauli.X, Pauli.Y], [Pauli.Z, Pauli.Z], 1, [Pauli.Y, Pauli.X]),
        ([Pauli.X, Pauli.Y], [Pauli.X, Pauli.Y], 1, [Pauli.I, Pauli.I]),
        ([Pauli.X, Pauli.X], [Pauli.Y, Pauli.Y], -1, [Pauli.Z, Pauli.Z]),
        ([Pauli.X, Pauli.I], [Pauli.Y, Pauli.Y], 1j, [Pauli.Z, Pauli.Y]),
    ],
)
def test_pauli_string_mult(
    pauli_list1: list[Pauli],
    pauli_list2: list[Pauli],
    coeff_result: complex,
    pauli_list_result: list[Pauli],
) -> None:
    paulis_string1 = PauliString.from_pauli_sequence(pauli_list1)
    paulis_string2 = PauliString.from_pauli_sequence(pauli_list2)
    paulis_string_result = PauliString.from_pauli_sequence(
        pauli_list_result,
        coeff_result,
    )
    assert paulis_string1 * paulis_string2 == paulis_string_result


@pytest.mark.parametrize(
    ("pauli_list1", "pauli_list2", "commutes"),
    [
        ([Pauli.X, Pauli.Y], [Pauli.Z, Pauli.Z], True),
        ([Pauli.X, Pauli.Y], [Pauli.X, Pauli.Y], True),
        ([Pauli.X, Pauli.X], [Pauli.Y, Pauli.Y], True),
        ([Pauli.X, Pauli.X, Pauli.X], [Pauli.Y, Pauli.Y, Pauli.Y], False),
        ([Pauli.X, Pauli.I], [Pauli.Y, Pauli.Y], False),
    ],
)
def test_pauli_string_commutes(
    pauli_list1: list[Pauli],
    pauli_list2: list[Pauli],
    commutes: bool,
) -> None:
    paulis_string1 = PauliString.from_pauli_sequence(pauli_list1)
    paulis_string2 = PauliString.from_pauli_sequence(pauli_list2)
    assert paulis_string1.commutes_with(paulis_string2) == commutes


@pytest.mark.parametrize(
    ("pauli_coeff", "pauli_list", "trace"),
    [
        (1, [Pauli.X, Pauli.Y], 0),
        (2, [Pauli.X], 0),
        (2, [Pauli.Y], 0),
        (2, [Pauli.Z], 0),
        (3.3, [], 3.3),
        (3 + 4j, [Pauli.I], 3 + 4j),
    ],
)
def test_pauli_string_normalized_trace(
    pauli_coeff: complex,
    pauli_list: list[Pauli],
    trace: complex,
) -> None:
    paulis_string = PauliString.from_pauli_sequence(pauli_list, pauli_coeff)
    assert paulis_string.normalized_trace() == trace
