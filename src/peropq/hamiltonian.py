from collections.abc import Sequence
from dataclasses import dataclass

from peropq import pauli



@dataclass
class Hamiltonian:
    # def __init__(self,commuting_pauli_string_list):
    pauli_string_list: Sequence[pauli.PauliString]
    def __post_init__(self):
        cjs:Sequence[complex] = []
        for pauli_string in self.pauli_string_list:
            cjs.append(pauli_string.coefficient)
            pauli_string.coefficient=1.0
        self.cjs=cjs
    
    def get_n_terms(self) -> int:
        return len(self.pauli_string_list)

    def get_cjs(self) -> Sequence[complex]:
        return self.cjs
