from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class Pauli(Enum):
    """Enum class for Pauli gates."""

    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __mul__(self, other: "Pauli") -> tuple[complex, "Pauli"]:
        """Multiply two Paulis."""
        match (self, other):
            case (Pauli.X, Pauli.Y) | (Pauli.Y, Pauli.X):
                coeff = 1j if self == Pauli.X else -1j
                return coeff, Pauli.Z
            case (Pauli.Y, Pauli.Z) | (Pauli.Z, Pauli.Y):
                coeff = 1j if self == Pauli.Y else -1j
                return coeff, Pauli.X
            case (Pauli.Z, Pauli.X) | (Pauli.X, Pauli.Z):
                coeff = 1j if self == Pauli.Z else -1j
                return coeff, Pauli.Y
            case (Pauli.I, _):
                return 1, other
            case (_, Pauli.I):
                return 1, self
        return 1, Pauli.I


@dataclass()
class PauliString:
    """Class representing a Pauli string multiplied by a complex coefficient."""

    qubit_pauli_map: dict[int, Pauli] = field(default_factory=dict)
    coefficient: complex = 1

    @classmethod
    def from_pauli_sequence(
        cls,
        paulis: Sequence[Pauli],
        coeff: complex = 1,
        start_qubit: int = 0,
    ) -> "PauliString":
        """Create from a sequence of Paulis.

        The Paulis will be placed sequentially on qubits starting from qubit index <start_qubit>

        :param paulis a list/tuple of Paulis
        :param coeff coefficient of the tensor
        :param start_qubit which qubit to place first Pauli on
        """
        return cls({(i + start_qubit): pauli for i, pauli in enumerate(paulis)}, coeff)

    def get_pauli(self, qubit: int) -> Pauli:
        """Return the pauli at a given qubit.

        :param qubit the qubit index to retrieve the Pauli from

        :returns: the pauli at qubit index <qubit> otherwise, the identity Pauli.I
        """
        if qubit < 0:
            message = "qubit index cannot be negative"
            raise ValueError(message)
        return self.qubit_pauli_map.get(qubit, Pauli.I)

    def update(
        self,
        qubit_pauli_map: dict[int, Pauli],
        coeff: complex | None = None,
    ) -> "PauliString":
        """Update the PauliTensor."""
        for qubit, pauli in qubit_pauli_map.items():
            self.qubit_pauli_map[qubit] = pauli
        if coeff:
            self.coefficient = self.coefficient * coeff
        return self
