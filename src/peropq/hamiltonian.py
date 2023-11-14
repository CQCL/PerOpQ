from collections.abc import Sequence
from dataclasses import dataclass

from peropq import pauli


@dataclass
class Hamiltonian:
    """Class representing the Hamilonian."""

    pauli_string_list: Sequence[pauli.PauliString]

    def __post_init__(self) -> None:
        """Initialize the Pauli strings with coeff 1.0 and store the coefficients in self.cjs."""
        cjs: Sequence[complex] = []
        for pauli_string in self.pauli_string_list:
            cjs.append(pauli_string.coefficient)
            pauli_string.coefficient = 1.0
        self.cjs = cjs

    def get_n_terms(self) -> int:
        """Method returning the number of Pauli Strings."""
        return len(self.pauli_string_list)

    def get_cjs(self) -> Sequence[complex]:
        """Method returning the coefficients."""
        return self.cjs
