import numpy as np
from peropq.bch import VariationalNorm
from peropq.ed_module import ExactDiagonalization as ED
from peropq.hamiltonian import Hamiltonian
from peropq.pauli_bitstring import PauliString,pauli_from_string
from peropq.unconstrained_variational_unitary import (
    UnconstrainedVariationalUnitary as VU,
)

########################################
# This file tests that the variaional norm calculation is correct by comparing it to an analytical formula
########################################

def expectation_value(state, obs_list):
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj() @ obs @ state)
    return expectation_value_list


z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
bc_modifier = 1
nx = 2
ny = 2
n = nx * ny
for i in range(n):
    zi = pauli_from_string(string = 'Z', length=4,start_qubit=i)
    xi = pauli_from_string(string = 'X', length=4,start_qubit=i)
    yi = pauli_from_string(string = 'Y', length=4,start_qubit=i)
    z_list.append(zi)
    x_list.append(xi)
    y_list.append(yi)
term_list = []
for i in range(n):
    term_list.append(0.3 * z_list[i])
for i in range(n):
    term_list.append(0.3 * x_list[i])
V = -1
# vertical bonds
for col in range(nx):
    v_list = []
    for site in range(col, n + 1 - nx + col, nx):
        v_list.append(site)
    print("vertical bonds")
    for isite in range(len(v_list) - bc_modifier):
        term_list.append(
            z_list[v_list[isite]] * z_list[v_list[(isite + 1) % len(v_list)]],
        )
        print(v_list[isite], v_list[(isite + 1) % len(v_list)])

# horizontal bonds
start_sites = []
for site in range(0, n + 1 - nx, nx):
    start_sites.append(site)
print("horizontal bonds")
for site in start_sites:
    for col in range(nx - bc_modifier):
        term_list.append(z_list[site + col] * z_list[site + (col + 1) % (nx)])
        print((site + col), site + (col + 1) % (nx))

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
time_list = [0.4]
ed = ED(number_of_qubits=n)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Get list of single Z string
z_list_sparse = []
for site in range(n - 1):
    z_list_sparse.append(ed.get_sparse(x_list[site] * x_list[site + 1]))
z_t_continous = []
z_t_trotter = []
z_t_variational = []
z_t_variational_c = []
energy_trotter = []
energy_variational = []
energy_variational_c = []
# Get the observable for the continuous time evolution
state_init = np.array([1.0 + 0.0j] + [0.0] * (2**n - 1))
energy = state_init.T @ h_ising_matrix @ state_init
nlayer = 1
for time in time_list:
    variational_unitary = VU(h_ising, number_of_layer=nlayer, time=time)
    variational_unitary.set_theta_to_trotter()
    c2test = variational_unitary.c2_squared_test(variational_unitary.theta)
    c2 = variational_unitary.c2_squared(variational_unitary.theta)
    print("second order")
    print(c2test)
    print(c2)
    print("third order")
    c3test = variational_unitary.c3_squared_test(variational_unitary.theta)
    c2test = variational_unitary.c2_squared_test(variational_unitary.theta)
    variational_norm = VariationalNorm(
        variational_unitary=variational_unitary, order=3, unconstrained=True
    )
    variational_norm.get_commutators()
    variational_norm.get_traces()
    c3 = variational_norm.calculate_norm(variational_unitary.theta)
    print("c3 ", c3)
    print("c3_test ", c3test)
    # trotter_unitary = copy.deepcopy(variational_unitary)
    variational_norm = VariationalNorm(
        variational_unitary=variational_unitary, order=2, unconstrained=True
    )
    variational_norm.get_commutators()
    variational_norm.get_traces()
    n2 = variational_norm.calculate_norm(variational_unitary.theta)
    variational_norm = VariationalNorm(
        variational_unitary=variational_unitary, order=3, unconstrained=True
    )
    variational_norm.get_commutators()
    variational_norm.get_traces()
    n3 = variational_norm.calculate_norm(variational_unitary.theta)
    trotter_error = ed.get_error(variational_unitary, hamiltonian=h_ising)
    print("n2 ", n2)
    print("n3 ", n3)
    print("trotter ", trotter_error)
