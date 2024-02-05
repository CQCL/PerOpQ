import numpy as np
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString

"""
def test_optimizer() -> None:
    z_list = []
    x_list = []
    N=50
    for i in range(N):
        zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
        z_list.append(zi)
        xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
        x_list.append(xi)
    term_list = []
    for i in range(N - 1):
        term_list.append(z_list[i] * z_list[i + 1])
    for i in range(N):
        term_list.append(x_list[i])
    h_ising = Hamiltonian(pauli_string_list=term_list)
    opt = Optimizer(hamiltonian=h_ising, R=3, t=1.0)
    print("starting optimization")
    res = opt.get_minumum_c2_squared()
    print("res ",res)
"""
