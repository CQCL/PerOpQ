from __future__ import annotations

from cmath import isclose
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Pauli(Enum):
    """Enum class for Pauli gates."""

    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __mul__(self, other: Pauli) -> tuple[complex, Pauli]:
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

    def commutes_with(self, other: Pauli) -> bool:
        """Return whether this Pauli commutes with given Pauli.

        :param other The right side of the Commutator

        :return: True if PauliStrings commute, False otherwise
        """
        if self == other:
            return True
        if self == Pauli.I:
            return True
        if other == Pauli.I:
            return True
        return False


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
    ) -> PauliString:
        """Create from a sequence of Paulis.

        The Paulis will be placed sequentially on qubits starting from qubit index <start_qubit>

        :param paulis a list/tuple of Paulis
        :param coeff coefficient of the tensor
        :param start_qubit which qubit to place first Pauli on
        """
        return cls({(i + start_qubit): pauli for i, pauli in enumerate(paulis)}, coeff)

    def __post_init__(self) -> None:
        """Special dataclass method that runs directly after initialization."""
        self.prune()

    def prune(self) -> PauliString:
        """Remove any Identities from the string."""
        qubits = list(self.qubit_pauli_map.keys())
        for qubit in qubits:
            if self.get_pauli(qubit) == Pauli.I:
                self.qubit_pauli_map.pop(qubit)
        return self

    def __eq__(self, other: object) -> bool:
        """Equality for PauliStrings.

        Two PauliStrings are equal if their coefficients are equal and
        if their Paulis are equal on all qubits. The underlying qubit_pauli_maps
        may differ (in the number of Identities).
        """
        if not isinstance(other, PauliString):
            return False
        if not isclose(self.coefficient, other.coefficient):
            return False
        leftover_qubits = set(other.qubit_pauli_map.keys())
        for qubit, self_pauli in self.qubit_pauli_map.items():
            if self_pauli != other.get_pauli(qubit):
                return False
            leftover_qubits.discard(qubit)
        for qubit in leftover_qubits:
            if self.get_pauli(qubit) != other.get_pauli(qubit):
                return False
        return True

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
        update_qubit_pauli_map: dict[int, Pauli],
        coeff: complex | None = None,
    ) -> PauliString:
        """Update the PauliString."""
        for qubit, pauli in update_qubit_pauli_map.items():
            if pauli == Pauli.I:
                self.qubit_pauli_map.pop(qubit, Pauli.I)
            else:
                self.qubit_pauli_map[qubit] = pauli
        if coeff:
            self.coefficient = self.coefficient * coeff
        return self

    def __mul__(self, other: PauliString | complex) -> PauliString:
        """Multiply PauliString with a complex number or PauliString."""
        if isinstance(other, PauliString):
            return _pauli_string_mult(self, other)
        return PauliString(self.qubit_pauli_map, self.coefficient * other)

    def __rmul__(self, other: PauliString | complex) -> PauliString:
        """Multiply PauliString with a complex number or PauliString."""
        if isinstance(other, PauliString):
            return _pauli_string_mult(self, other)
        return PauliString(self.qubit_pauli_map, self.coefficient * other)

    def commutes_with(self, other: PauliString) -> bool:
        """Return whether this PauliString commutes with given PauliString.

        :param other The right side of the Commutator

        :return: True if PauliStrings commute, False otherwise
        """
        number_anti_commute = 0
        for qubit, self_pauli in self.qubit_pauli_map.items():
            other_pauli = other.get_pauli(qubit)
            if not self_pauli.commutes_with(other_pauli):
                number_anti_commute += 1
        if number_anti_commute % 2 == 0:
            return True
        return False

    def normalized_trace(self) -> complex:
        """Returns `2^(-n)*Tr(self)`.

        The trace is just the coefficient times the product of the traces of the Paulis
        over all n qubits (including any identities on qubits not explicitly contained in the qubit_pauli_map). Since
        `Tr(Pauli.I) == 2`, and `Tr(Pauli.X) == Tr(Pauli.Y) == Tr(Pauli.Z) == 0`, the result will either be 0 (if any
        of the Paulis is not the Identity) or `self.coefficient` (if all paulis are the identity).
        """
        self.prune()
        if len(self.qubit_pauli_map) == 0:
            return self.coefficient
        return 0


def _pauli_string_mult(
    pauli_string1: PauliString,
    pauli_string2: PauliString,
) -> PauliString:
    new_pauli_string = PauliString(
        {},
        pauli_string1.coefficient * pauli_string2.coefficient,
    )
    leftover_qubits = set(pauli_string2.qubit_pauli_map.keys())
    for qubit, left_pauli in pauli_string1.qubit_pauli_map.items():
        leftover_qubits.discard(qubit)
        right_pauli = pauli_string2.get_pauli(qubit)
        coeff, new_pauli = left_pauli * right_pauli
        new_pauli_string.update({qubit: new_pauli}, coeff)
    for qubit in leftover_qubits:
        right_pauli = pauli_string2.get_pauli(qubit)
        if right_pauli != Pauli.I:
            # left pauli must be I
            new_pauli_string.update({qubit: right_pauli})
    return new_pauli_string
