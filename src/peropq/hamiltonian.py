from collections.abc import Sequence
from dataclasses import dataclass

from peropq import pauli



@dataclass
class Hamiltonian:
    # def __init__(self,commuting_pauli_string_list):
    pauli_string_list: Sequence[pauli.PauliString]

    def get_n_terms(self) -> int:
        return len(self.pauli_string_list)

    def get_cjs(self) -> Sequence[complex]:
        cjs:Sequence[complex] = []
        for pauli_string in self.pauli_string_list:
            cjs.append(pauli_string.coefficient)
        return cjs
