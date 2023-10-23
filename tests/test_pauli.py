from peropq.pauli import Pauli


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
