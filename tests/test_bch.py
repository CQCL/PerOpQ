from peropq.bch import VariationalNorm
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary


def test_optimizer() -> None:
    z_list: list[PauliString] = []
    x_list: list[PauliString] = []
    y_list: list[PauliString] = []
    n = 30
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
    opt = Optimizer()
    variational_unitary.set_theta_to_trotter()
    c2_old = variational_unitary.c2_squared(
        variational_unitary.get_initial_trotter_vector(),
    )
    # Try to get c2_squared via the bch formula
    variational_unitary.set_theta_to_trotter()
    variational_norm = VariationalNorm(variational_unitary, order=2)
    variational_norm.get_commutators()
    variational_norm.get_traces()
    norm_var = variational_norm.calculate_norm(
        variational_unitary.get_initial_trotter_vector()
    )
    print("norm_var ", norm_var)
    print("c2_old ", c2_old)
    variational_unitary.set_theta_to_trotter()
    res = opt.optimize(variational_unitary)
    variational_unitary.set_theta_to_trotter()
    res2 = opt.optimize_arbitrary(variational_unitary, order=2)
    print(res)
    print(res2)
    import sys

    sys.exit()
    # assert res[0].fun < 1e-10
    #######################################
    #
    # Other example with off diagonal terms
    #
    #######################################
    # # XY+YZ+X+Z
    term_list = []
    for i in range(n - 1):
        term_list.append(x_list[i] * y_list[i + 1])
    for i in range(n - 1):
        term_list.append(y_list[i] * z_list[i + 1])
    for i in range(n):
        term_list.append(x_list[i])
    for i in range(n):
        term_list.append(-1.0 * z_list[i])
    h_off_diag = Hamiltonian(pauli_string_list=term_list)
    variational_unitary = VariationalUnitary(h_off_diag, number_of_layer=3, time=0.8)
    opt = Optimizer()
    variational_unitary.set_theta_to_trotter()
    c2_old = variational_unitary.c2_squared(
        variational_unitary.get_initial_trotter_vector(),
    )
    # Try to get c2_squared via the bch formula
    variational_unitary.set_theta_to_trotter()
    variational_norm = VariationalNorm(variational_unitary, order=2)
    variational_norm.get_commutators()
    variational_norm.get_traces()
    norm_var = variational_norm.calculate_norm(
        variational_unitary.get_initial_trotter_vector()
    )
    print("norm_var ", norm_var)
    print("c2_old ", c2_old)
    import sys

    sys.exit()
    # res = opt.optimize(variational_unitary)
    # assert res[0].fun < 1e-10


test_optimizer()
