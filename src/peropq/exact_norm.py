import scipy
import scipy.sparse as sp  # type: ignore[import-untyped]
from numpy import typing as npt
import numpy as np

from peropq.hamiltonian import Hamiltonian
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary
from peropq.ed_module import ExactDiagonalization

class ExactUnitary(UnconstrainedVariationalUnitary):
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        number_of_layer: int,
        time: float,
        number_of_qubits: int,
        ):
        """
        Class representing a variational unitary which is optimized through an exact evaluation of the norm by means of exact diagonalization.
        hamiltonian: target Hamiltonian
        number_of_layers: number of layers to be optimized over
        time: final time
        number_of_qubits: number of qubits 
        """
        super().__init__(
        hamiltonian,
        number_of_layer,
        time,
        )
        # Get exact unitary
        self.exact_diagonalization = ExactDiagonalization(number_of_qubits=number_of_qubits)
        self.exact_unitary = self.exact_diagonalization.get_continuous_time_evolution(hamiltonian=hamiltonian,time=time)
    

    def get_exact_norm(self,theta:npt.NDArray):
        """
        Get the exact norm:
        theta: variational parameter for which the norm must be evaluated
        """
        # Reshape theta
        theta_updated = np.array(theta).reshape((self.depth,self.n_terms))
        self.update_theta(theta_updated)
        # Get the variational unitary
        variational_evolution = self.exact_diagonalization.get_variational_evolution(variational_unitary=self)
        # Get the frobenius norm squared
        return sp.linalg.norm(
            self.exact_unitary - variational_evolution,
            ord="fro",
        )


