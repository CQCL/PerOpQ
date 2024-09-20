import scipy
import scipy.sparse as sp  # type: ignore[import-untyped]
from numpy import typing as npt

from peropq.hamiltonian import Hamiltonian
from peropq.pauli_bitstring import PauliString
from peropq.variational_unitary import VariationalUnitary


class ExactDiagonalization:
    """Class implementing exact diagonalization for benchmarks on small system sizes."""

    def __init__(self, number_of_qubits: int) -> None:
        """
        Initializing the ed module.

        :param qubit_number: number of qubits.
        """
        self.number_of_qubits = number_of_qubits

    def pauli_to_sparse(self, pauli: str) -> sp.spmatrix:
        """
        Converts a Pauli to a 2x2 sparse matrix.

        :param pauli: to be converted.
        """
        match pauli:
            case "I":
                return sp.csc_matrix([[1.0, 0], [0, 1.0]], dtype=complex)
            case "X":
                return sp.csc_matrix([[0, 1.0], [1.0, 0]], dtype=complex)
            case "Y":
                return sp.csc_matrix([[0, -1j], [1j, 0]], dtype=complex)
            case "Z":
                return sp.csc_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    def get_sparse(self, pauli_string: PauliString) -> sp.spmatrix:
        """
        Transforms PauliString into sparse matrix.

        :param pauli_string: to be transformed into a sparse matrix.
        """
        sparse_string: sp.spmatrix = pauli_string.coefficient * self.pauli_to_sparse(
            pauli_string.get_pauli(0),
        )
        for qubit in range(1, self.number_of_qubits):
            sparse_string = sp.kron(
                sparse_string,
                self.pauli_to_sparse(pauli_string.get_pauli(qubit)),
            )
        return sp.csc_matrix(sparse_string)

    def get_hamiltonian_matrix(self, hamiltonian: Hamiltonian) -> sp.spmatrix:
        """Param hamiltonian: to be converted to sparse."""
        hamiltonian_matrix = self.get_sparse(
            hamiltonian.cjs[0] * hamiltonian.pauli_string_list[0],
        )
        for i_string in range(1, len(hamiltonian.pauli_string_list)):
            hamiltonian_matrix += self.get_sparse(
                hamiltonian.cjs[i_string] * hamiltonian.pauli_string_list[i_string],
            )
        return hamiltonian_matrix

    def get_continuous_time_evolution(
        self,
        hamiltonian: Hamiltonian,
        time: float,
    ) -> sp.spmatrix:
        """
        Get the continuous time evolution of an Hamiltonian.

        :param hamiltonian: governing the dynamics
        :param time: at which we want to time evolve.
        """
        hamiltonian_matrix = self.get_hamiltonian_matrix(hamiltonian)
        return sp.linalg.expm(-1j * time * hamiltonian_matrix)

    def get_variational_evolution(
        self,
        variational_unitary: VariationalUnitary,
    ) -> sp.spmatrix:
        """
        Get the time evolution unitary from a variational unitary.

        :param variational_unitary: to be evolved.
        """
        u_sparse = sp.eye(2**self.number_of_qubits)
        for layer in range(variational_unitary.depth):
            for i_term, term in enumerate(variational_unitary.pauli_string_list):
                u_sparse = u_sparse @ sp.linalg.expm(
                    -1j
                    * variational_unitary.theta[layer, i_term]
                    * self.get_sparse(term),
                )
        return u_sparse

    def apply_continuous_to_state(
        self,
        hamiltonian: Hamiltonian,
        time: float,
        state: npt.NDArray,
    ) -> npt.NDArray:
        """
        Apply the continuous time evolution.

        :param hamiltonian: governing the dynamics
        :param time: at which we want to time evolve.
        :param state: to be evolved.
        """
        hamiltonian_matrix = self.get_hamiltonian_matrix(hamiltonian)
        return scipy.sparse.linalg.expm_multiply(
            -1j * time * hamiltonian_matrix,
            state,
        )

    def apply_variational_to_state(
        self,
        variational_unitary: VariationalUnitary,
        state: npt.NDArray,
    ) -> npt.NDArray:
        """
        Apply the variational unitary to a state.

        :param variational_unitary: to be applied
        :param state: on which the unitary is applied
        """
        unitary = self.get_variational_evolution(variational_unitary)
        return unitary @ state

    def get_error(
        self,
        variational_unitary: VariationalUnitary,
        hamiltonian: Hamiltonian,
    ) -> float:
        """
        Returns the error made for by the variational unitary (compared to the continuous time evolution).

        :param variational_unitary  to be compared with the continuous time evolution.
        :param hamiltonian: used to caclulated the continuous time evolution.
        :returns: exact error norm
        """
        sparse_variational_unitary = self.get_variational_evolution(variational_unitary)
        sparse_continuous_unitary = self.get_continuous_time_evolution(
            hamiltonian,
            time=variational_unitary.time,
        )
        return sp.linalg.norm(
            sparse_continuous_unitary - sparse_variational_unitary,
            ord="fro",
        )
