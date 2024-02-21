import copy

import matplotlib.pyplot as plt
import numpy as np
from peropq.bch import VariationalNorm
from peropq.ed_module import ExactDiagonalization as ED
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.unconstrained_variational_unitary import (
    UnconstrainedVariationalUnitary as VU,
)
from peropq.exact_norm import ExactUnitary



def expectation_value(state, obs_list):
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj() @ obs @ state)
    return expectation_value_list

#############
# Options 
# Random?
random_hamiltonian= True
# Tilted or transverse?
hamiltonian_type = 'transverse'

z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
bc_modifier = 1
nx = 2
ny = 2
n = nx * ny
np.random.seed(0)
for i in range(n):
    zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
    z_list.append(zi)
    xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
    x_list.append(xi)
    yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
    y_list.append(yi)
term_list = []
if hamiltonian_type == 'tilted':
    for i in range(n):
        if not random_hamiltonian:
            term_list.append(z_list[i])
        else:
            term_list.append(np.random.rand() * z_list[i])
elif hamiltonian_type == 'transverse':
    pass
else:
    raise ValueError('Hamiltonian type not implemented!')
for i in range(n):
    if not random_hamiltonian:
        term_list.append(x_list[i])
    else:
        term_list.append(np.random.rand() * x_list[i])
V = -1
# vertical bonds
for col in range(nx):
    v_list = []
    for site in range(col, n + 1 - nx + col, nx):
        v_list.append(site)
    print("vertical bonds")
    for isite in range(len(v_list) - bc_modifier):
        if random_hamiltonian:
            term_list.append(
                np.random.rand()*z_list[v_list[isite]] * z_list[v_list[(isite + 1) % len(v_list)]],
            )
        else:
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
        if random_hamiltonian:
            term_list.append(np.random.rand()*z_list[site + col] * z_list[site + (col + 1) % (nx)])
        else:
            term_list.append(z_list[site + col] * z_list[site + (col + 1) % (nx)])
        print((site + col), site + (col + 1) % (nx))

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
time_list = [0.1]
ed = ED(number_of_qubits=n)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Get list of single Z string
z_list_sparse = []
for site in range(n - 1):
    z_list_sparse.append(ed.get_sparse(z_list[site]))
z_t_continous = []
z_t_trotter = []
z_t_variational_3 = []
z_t_variational_c = []
z_t_variational_exact = []
energy_trotter = []
energy_variational_3 = []
energy_variational_c = []
energy_variational_exact = []
# Get the observable for the continuous time evolution
state_init = np.array([0.0 + 0.0j] + [0.0] * (2**n - 1))
random_integer = np.random.randint(low=0,high=2**n)
state_init[random_integer] = 1.0
energy = state_init.T @ h_ising_matrix @ state_init
nlayer = 3
variational_3_unitary = VU(h_ising, number_of_layer=nlayer, time=time_list[0])
variational_3_unitary.set_theta_to_trotter()
for itime,time in enumerate(time_list):
    variational_3_unitary.time=time
    trotter_unitary = copy.deepcopy(variational_3_unitary)
    opt = Optimizer()
    trotter_unitary.set_theta_to_trotter()
    theta_flat = variational_3_unitary.flatten_theta(variational_3_unitary.theta)
    res = opt.optimize_arbitrary(
        variational_unitary=variational_3_unitary,
        order=3,
        unconstrained=True,
        initial_guess=theta_flat,
    )
    print("order 3 res ", res)

variational_unitary_c = VU(h_ising, number_of_layer=nlayer, time=time_list[0])
variational_unitary_c.set_theta_to_trotter()
trotter_unitary = copy.deepcopy(variational_unitary_c)
for time in time_list:
    variational_unitary_c.time = time
    opt = Optimizer()
    theta_flat = variational_unitary_c.flatten_theta(variational_unitary_c.theta)
    res = opt.optimize_arbitrary(
        variational_unitary=variational_unitary_c,
        order=2,
        unconstrained=True,
    )  
    print("order 2 res ", res)

########
# Do the optimization with ExactUnitary

exact_unitary = ExactUnitary(h_ising, number_of_layer=nlayer, time=time_list[0],number_of_qubits=n)
exact_unitary.set_theta_to_trotter()
for time in time_list:
    exact_unitary.time = time
    opt = Optimizer()
    theta_flat = exact_unitary.flatten_theta(exact_unitary.theta)
    res = opt.optimize_exact(
        exact_unitary=exact_unitary,
        initial_guess=theta_flat,
    )  
    print("res exact unitary, ",res)
state_variational_3 = copy.deepcopy(state_init)
state_variational_c = copy.deepcopy(state_init)
state_variational_exact = copy.deepcopy(state_init)
state_trotter = copy.deepcopy(state_init)
state_continuous = copy.deepcopy(state_init)
# Check that the norm is better for the variational unitary
trotter_error = ed.get_error(trotter_unitary, hamiltonian=h_ising)
variational_3_error = ed.get_error(
    variational_unitary=variational_3_unitary,
    hamiltonian=h_ising,
)

variational_c_error = ed.get_error(
    variational_unitary=variational_unitary_c,
    hamiltonian=h_ising,
)
exact_unitary_error =ed.get_error(variational_unitary=exact_unitary,hamiltonian=h_ising)
print("trotter error", trotter_error)
print("variational error order 3", variational_3_error)
print("variational error order 2", variational_c_error)
print("exact unitary error ",exact_unitary_error)

for istep in range(10):
    state_trotter = ed.apply_variational_to_state(trotter_unitary, state_trotter)
    state_continuous = ed.apply_continuous_to_state(
        hamiltonian=h_ising,
        time=time,
        state=state_continuous,
    )
    state_variational_3 = ed.apply_variational_to_state(
        variational_3_unitary,
        state_variational_3,
    )
    state_variational_c = ed.apply_variational_to_state(
        variational_unitary_c,
        state_variational_c,
    )
    state_variational_exact = ed.apply_variational_to_state(
        exact_unitary,
        state_variational_exact,
    )

    sz_expectation_value_trotter = expectation_value(state_trotter, z_list_sparse)
    sz_expectation_value_continuous = expectation_value(state_continuous, z_list_sparse)
    sz_expectation_value_variational_3 = expectation_value(
        state_variational_3,
        z_list_sparse,
    )
    sz_expectation_value_variational_exact = expectation_value(
        state_variational_exact,
        z_list_sparse,
    )
    sz_expectation_value_variational_c = expectation_value(
        state_variational_c,
        z_list_sparse,
    )

    z_t_continous.append(sz_expectation_value_continuous)
    z_t_trotter.append(sz_expectation_value_trotter)
    z_t_variational_3.append(sz_expectation_value_variational_3)
    z_t_variational_c.append(sz_expectation_value_variational_c)
    z_t_variational_exact.append(sz_expectation_value_variational_exact)

    energy_trotter.append(state_trotter.T.conj() @ h_ising_matrix @ state_trotter)
    energy_variational_3.append(
        state_variational_3.T.conj() @ h_ising_matrix @ state_variational_3,
    )
    energy_variational_c.append(
        state_variational_c.T.conj() @ h_ising_matrix @ state_variational_c,
    )

    energy_variational_exact.append(
        state_variational_exact.T.conj() @ h_ising_matrix @ state_variational_exact,
    )
z_array_continuous = np.array(z_t_continous)
z_array_trotter = np.array(z_t_trotter)
z_array_variational_3 = np.array(z_t_variational_3)
z_array_variational_c = np.array(z_t_variational_c)
z_array_variational_exact = np.array(z_t_variational_exact)
energy_trotter = np.array(energy_trotter)
energy_variational_3 = np.array(energy_variational_3)
energy_variational_c = np.array(energy_variational_c)
energy_variational_exact = np.array(energy_variational_exact)

plt.figure()
plt.plot(z_array_continuous[:, int(n / 2)], label="continuous")
plt.plot(z_array_trotter[:, int(n / 2)], label="trotter")
plt.plot(z_array_variational_3[:, int(n / 2)], label="variational order 3")
plt.plot(z_array_variational_c[:, int(n / 2)], label="variational order 2")
plt.plot(z_array_variational_exact[:,int(n/2)], label="variational exact")
plt.xlabel("time step")
plt.ylabel(r"$\langle Z \rangle$")
plt.legend(loc="best")
# Plot the error on the energy
plt.figure()
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_trotter[:, int(n / 2)]),
    linestyle="--",
    marker="o",
    label="trotter",
)
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_variational_3[:, int(n / 2)]),
    label="variational order 3",
)
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_variational_exact[:, int(n / 2)]),
    label="variational exact",
)
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_variational_c[:, int(n / 2)]),
    label="variational order 2",
)
plt.xlabel("time step")
plt.ylabel("error")
plt.legend(loc="best")
plt.savefig('err_Z_Ising3x3_third_order.pdf')
plt.figure()
plt.plot(np.abs(np.array(energy_trotter) - energy), label="trotter")
plt.plot(np.abs(np.array(energy_variational_3) - energy), label="variational order 3")
plt.plot(
    np.abs(np.array(energy_variational_c) - energy),
    label="variational order 2",
)
plt.plot(
    np.abs(np.array(energy_variational_exact) - energy),
    label="variational exact",
)
plt.xlabel("time step")
plt.ylabel("energy error")
plt.legend(loc="best")
plt.savefig('err_E_titled_3x3_third_order.pdf')
plt.show()
