from peropq.pauli import Pauli


def test_pauli() -> None:
    """Simple test that pauli values are correct strings."""
    assert Pauli.I.value == "I"
    assert Pauli.X.value == "X"
    assert Pauli.Y.value == "Y"
    assert Pauli.Z.value == "Z"
