from peropq.pauli import Pauli, PauliTensor


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


def test_pauli_init() -> None:
    xy_string = PauliTensor.from_pauli_sequence(paulis=[Pauli.X, Pauli.Y])
    xy_string_prime = PauliTensor(
        qubit_pauli_map={0: Pauli.X, 1: Pauli.Y},
        coefficient=1,
    )
    assert xy_string == xy_string_prime
    ixy_string = PauliTensor.from_pauli_sequence(
        paulis=[Pauli.X, Pauli.Y],
        start_qubit=1,
    )
    ixy_string_prime = PauliTensor(
        qubit_pauli_map={0: Pauli.I, 1: Pauli.X, 2: Pauli.Y},
        coefficient=1,
    )
    for i in range(3):
        assert ixy_string.get_pauli(i) == ixy_string_prime.get_pauli(i)
