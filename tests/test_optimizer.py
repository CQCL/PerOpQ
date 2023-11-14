import numpy as np
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary


def test_optimizer() -> None:
    z_list = []
    x_list = []
    y_list = []
    N = 20
    for i in range(N):
        zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
        z_list.append(zi)
        xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
        x_list.append(xi)
        yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
        y_list.append(yi)
    # Ising model
    term_list = []
    for i in range(N - 1):
        term_list.append(z_list[i] * z_list[i + 1])
    for i in range(N):
        term_list.append(x_list[i])
    h_ising = Hamiltonian(pauli_string_list=term_list)
    variational_unitary = VariationalUnitary(h_ising, R=3, t=1.0)
    opt = Optimizer(variation_unitary=variational_unitary)
    # assert np.isclose(opt.C2_squared(), 0.25*2.66666666666666)
    res = opt.get_minumum_c2_squared()
    assert res[0].fun < 1e-10
    # XY+YZ+X+Z
    n_terms = 2 * (N - 1) + 2 * N
    term_list = []
    for i in range(N - 1):
        term_list.append(x_list[i] * y_list[i + 1])
    for i in range(N - 1):
        term_list.append(y_list[i] * z_list[i + 1])
    for i in range(N):
        term_list.append(x_list[i])
    for i in range(N):
        term_list.append(-1.0*z_list[i])
    h_off_diag = Hamiltonian(pauli_string_list=term_list)
    variational_unitary = VariationalUnitary(h_off_diag, R=3, t=1.0)
    opt = Optimizer(variation_unitary=variational_unitary)
    # assert np.isclose(opt.C2_squared(),4.444444444444444)
    res = opt.get_minumum_c2_squared()
    assert res[0].fun < 1e-10
test_optimizer()
