from __future__ import annotations
import numpy as np
from numpy import typing as npt
from numba import njit
from cmath import isclose
# npt.NDArray

class PauliString:
    def __init__(self, bit_string: npt.NDArray, coefficient: complex = 1.0,non_identity_indices:npt.NDArray=np.array([])) -> None:
        assert len(bit_string)//2 != 0
        self.bit_string = bit_string
        self.coefficient = coefficient
        # if len(non_identity_indices)==0:
        #     self.non_identity_indices = _get_non_identity_indices(self.bit_string)

    def __mul__(self, other: PauliString | complex) -> PauliString:
        """Multiply PauliString with a complex number or PauliString."""
        if isinstance(other, PauliString):
            return _pauli_string_mult(self, other)
        return PauliString(bit_string=self.bit_string, coefficient=self.coefficient * other)

    
    def __rmul__(self, other: complex) -> PauliString:
        """Multiply PauliString with a complex number or PauliString."""
        return PauliString(self.bit_string, self.coefficient * other)

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

    def commutes_with(self, other: PauliString) -> bool:
        """Return whether this PauliString commutes with given PauliString.

        :param other The right side of the Commutator

        :return: True if PauliStrings commute, False otherwise
        """
        return bit_string_commutation(bit_string1= self.bit_string, bit_string2=other.bit_string)

def pauli_from_string(string: str, length: int, start_qubit: int = 0, coefficient: complex = 1.0)->PauliString:
    assert start_qubit+len(string) <= length
    bit_string: npt.NDArray = np.zeros(2*length,dtype=int)
    non_identity_indices:list = []
    for i, character in enumerate(string):
        k = start_qubit+i
        if character == 'X':
            bit_string[2*k] = 1
            non_identity_indices.append(k)
        elif character == 'Y':
            bit_string[2*k] = 1
            bit_string[2*k+1] = 1
            non_identity_indices.append(k)
        elif character == 'Z':
            bit_string[2*k+1] = 1
            non_identity_indices.append(k)
        elif character == 'I':
            pass
        else:
            message = "The string providing contained invalid character. The recognized characters are 'X','Y','Z' and 'I'"
            raise(ValueError(message))
    return PauliString(bit_string=bit_string,coefficient=coefficient,non_identity_indices=np.array(non_identity_indices))

@njit(cache=True)
def _get_non_identity_indices(bit_string:npt.NDArray)->npt.NDArray:
    number_of_qubits:int = int(len(bit_string)/2)
    non_identity_indices:list = []
    for k in range(number_of_qubits):
        if bit_string[2*k]!=0 and bit_string[2*k+1]!=0:
            non_identity_indices.append(k)
    return np.array(non_identity_indices)

    
@njit(cache=True)
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

@njit(cache=True)
def bit_string_commutation(bit_string1:npt.NDArray, bit_string2: npt.NDArray) -> bool:
    """
    Return whether two bitstrings representing pauli strings commute
    """
    number_anti_commute:int = 0
    assert len(bit_string1)==len(bit_string2)
    for k in range(int(len(bit_string1)/2)):
        # Check identity
        left_is_identity:bool =  (bit_string1[2*k]==0 and bit_string1[2*k+1]==0) 
        right_is_identity:bool = (bit_string2[2*k]==0 and bit_string2[2*k+1]==0)
        if not (left_is_identity or right_is_identity):
            # Check if they are equal
            if not ((bit_string1[2*k]==bit_string2[2*k]) and (bit_string1[2*k+1]==bit_string2[2*k+1])):
                number_anti_commute+=1
    if number_anti_commute%2 == 0:
        return True
    return False
