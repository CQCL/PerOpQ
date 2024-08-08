from __future__ import annotations
import numpy as np
from numpy import typing as npt
import numba
from cmath import isclose
# npt.NDArray


class PauliString:
    def __init__(self, bit_string: npt.NDArray, coefficient: complex = 1.0) -> None:
        assert len(bit_string)//2 != 0
        self.bit_string = bit_string
        self.coefficient = coefficient

    def __mul__(self, other: PauliString | complex) -> PauliString:
        """Multiply PauliString with a complex number or PauliString."""
        if isinstance(other, PauliString):
            return _pauli_string_mult(self, other)
        return PauliString(bit_string=self.bit_string, coefficient=self.coefficient * other)

    def __eq__(self,other:object)->bool:
        """Equality for PauliStrings.

        Two PauliStrings are equal if their coefficients are equal and
        if their Paulis are equal on all qubits. The underlying qubit_pauli_maps
        may differ (in the number of Identities).
        """
        if not isinstance(other, PauliString):
            return False
        if not isclose(self.coefficient, other.coefficient):
            return False
        return np.array_equal(self.bit_string,other.bit_string) 

def pauli_from_string(string: str, length: int, start_qubit: int = 0, coefficient: complex = 1.0)->PauliString:
    assert start_qubit+len(string) <= length
    bit_string: npt.NDArray = np.zeros(2*length,dtype=int)
    for i, character in enumerate(string):
        k = start_qubit+i
        if character == 'X':
            bit_string[2*k] = 1
        elif character == 'Y':
            bit_string[2*k] = 1
            bit_string[2*k+1] = 1
        elif character == 'Z':
            bit_string[2*k+1] = 1
        elif character == 'I':
            pass
        else:
            message = "The string providing contained invalid character. The recognized characters are 'X','Y','Z' and 'I'"
            raise(ValueError(message))
    return PauliString(bit_string=bit_string,coefficient=coefficient)


def _get_i_power(bit_string1: npt.NDArray, bit_string2: npt.NDArray) -> int:
    N = len(bit_string1)
    power = 0
    for k in range(int(N/2)):
        power += bit_string1[2*k+1] * bit_string2[2*k] - bit_string1[2*k] * bit_string2[2*k+1] + 2*(((bit_string1[2*k]+bit_string2[2*k])//2) * (
            bit_string1[2*k+1]+bit_string2[2*k+1]) + (bit_string1[2*k]+bit_string2[2*k]) * ((bit_string1[2*k+1]+bit_string2[2*k+1])//2))
    return power % 4


def _pauli_string_mult(
    pauli_string1: PauliString,
    pauli_string2: PauliString,
) -> PauliString:
    N = len(pauli_string1.bit_string)
    new_bit_string = np.bitwise_xor(pauli_string1.bit_string, pauli_string2.bit_string)
    new_i_power = _get_i_power(
        pauli_string1.bit_string, pauli_string2.bit_string)
    return PauliString(bit_string=new_bit_string, coefficient=pauli_string1.coefficient*pauli_string2.coefficient*(1j)**new_i_power)
