from collections.abc import Sequence

from peropq import pauli_bitstring


class Hamiltonian:
    """Class representing the Hamilonian."""

    def __init__(
        self,
        pauli_string_list: Sequence[pauli_bitstring.PauliString],
    ) -> None:
        """
        Initialization.

        :param pauli_string_list: contains all the terms of the hamiltonian.
        """
        self.pauli_string_list = pauli_string_list
        # Initialize the Pauli strings with coeff 1.0 and store the coefficients in self.cjs.
        cjs: list[complex] = []
        for pauli_string in self.pauli_string_list:
            cjs.append(pauli_string.coefficient)
            pauli_string.coefficient = 1.0
        self.cjs = cjs

    def get_n_terms(self) -> int:
        """
        Method returning the number of Pauli Strings.

        :returns: number of terms (i.e. Pauli strings) of the Hamiltonian.
        """
        return len(self.pauli_string_list)

    def get_cjs(self) -> list[complex]:
        """
        Method returning the coefficients.

        :returns: coefficient in front of each term of the Hamiltonian.
        """
        return self.cjs
