import pytest
from peropq.commutators_bitstrings import get_commutator_pauli_tensors
from peropq.pauli_bitstring import pauli_from_string


@pytest.mark.parametrize(
    ("tenslist1", "tenslist2", "coeff", "tenslistresult"),
    [
        ("XXZ", "XXZ", 0, "III"),
        (
            "XXX",
            "YYY",
            2 * pow(1j, 3),
            "ZZZ",
        ),
        (
            "XZZ",
            "XXZ",
            2 * 1j,
            "IYI",
        ),
        ("XZ", "XY", 2 * (-1j), "IX"),
        (
            "XYX",
            "YXY",
            2 * (-1j) * pow(1j, 2),
            "ZZZ",
        ),
    ],
)
def test_tensor_commutators(
    tenslist1: str,
    tenslist2: str,
    coeff: complex,
    tenslistresult: str,
) -> None:
    tens1 = pauli_from_string(string=tenslist1, length=len(tenslist1))
    tens2 = pauli_from_string(string=tenslist2, length=len(tenslist2))
    tens_result = (
        0
        if coeff == 0
        else pauli_from_string(
            string=tenslistresult,
            length=len(tenslistresult),
            coefficient=coeff,
        )
    )
    assert tens_result == get_commutator_pauli_tensors(tens1, tens2)
