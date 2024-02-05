from peropq.ed_module import ExactDiagonalization# type: ignore[import-untyped]  
from peropq.hamiltonian import Hamiltonian# type: ignore[import-untyped]  
from peropq.pauli import Pauli, PauliString# type: ignore[import-untyped]  
from peropq.variational_unitary import VariationalUnitary# type: ignore[import-untyped]  

import numpy as np


def test_ed_module() ->None:
    z_list: list[PauliString] = []
    x_list: list[PauliString] = []
    y_list: list[PauliString] = []
    n = 4
    for i in range(n):
        zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
        z_list.append(zi)
        xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
        x_list.append(xi)
        yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
        y_list.append(yi)
    # Ising model
    term_list = []
    for i in range(n - 1):
        term_list.append(z_list[i] * z_list[i + 1])
    for i in range(n):
        term_list.append(x_list[i])
    h_ising = Hamiltonian(pauli_string_list=term_list)
    variational_unitary = VariationalUnitary(h_ising, number_of_layer=3, time=1.0)
    variational_unitary.set_theta_to_trotter()
    ed = ExactDiagonalization(number_of_qubits=4)
    assert np.isclose(ed.get_error(variational_unitary,h_ising),1.2703740757929238)

    
