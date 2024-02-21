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
        term_list.append(z_list[site + col] * z_list[site + (col + 1) % (nx)])
        print((site + col), site + (col + 1) % (nx))

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
time_list = [0.3]
ed = ED(number_of_qubits=n)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Get list of single Z string
z_list_sparse = []
for site in range(n - 1):
    z_list_sparse.append(ed.get_sparse(z_list[site]))
z_t_continous = []
z_t_trotter = []
z_t_variational = []
z_t_variational_c = []
energy_trotter = []
energy_variational = []
energy_variational_c = []
# Get the observable for the continuous time evolution
state_init = np.array([0.0 + 0.0j] + [0.0] * (2**n - 1))
random_integer = np.random.randint(low=0,high=2**n)
state_init[random_integer] = 1.0
energy = state_init.T @ h_ising_matrix @ state_init
# print("unconstrained ")
# print("-------")
nlayer = 1
variational_unitary = VU(h_ising, number_of_layer=nlayer, time=time_list[0])
variational_unitary.set_theta_to_trotter()
trotter_unitary = copy.deepcopy(variational_unitary)

########
# Do the optimization with ExactUnitary

state_trotter = copy.deepcopy(state_init)
state_continuous = copy.deepcopy(state_init)
# Check that the norm is better for the variational unitary
trotter_error = ed.get_error(trotter_unitary, hamiltonian=h_ising)

print("trotter error", trotter_error)

for istep in range(10):
    state_trotter = ed.apply_variational_to_state(trotter_unitary, state_trotter)
    state_continuous = ed.apply_continuous_to_state(
        hamiltonian=h_ising,
        time=time_list[0],
        state=state_continuous,
    )

    sz_expectation_value_trotter = expectation_value(state_trotter, z_list_sparse)
    sz_expectation_value_continuous = expectation_value(state_continuous, z_list_sparse)

    z_t_continous.append(sz_expectation_value_continuous)
    z_t_trotter.append(sz_expectation_value_trotter)
    energy_trotter.append(state_trotter.T.conj() @ h_ising_matrix @ state_trotter)

z_array_continuous = np.array(z_t_continous)
z_array_trotter = np.array(z_t_trotter)
energy_trotter = np.array(energy_trotter)

plt.figure()
plt.plot(z_array_continuous[:, int(n / 2)], label="continuous")
plt.plot(z_array_trotter[:, int(n / 2)], label="trotter")
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
plt.xlabel("time step")
plt.ylabel("error")
plt.legend(loc="best")
plt.savefig('err_Z_Ising3x3_third_order.pdf')
plt.figure()
plt.plot(np.abs(np.array(energy_trotter) - energy), label="trotter")
plt.xlabel("time step")
plt.ylabel("energy error")
plt.legend(loc="best")
plt.savefig('err_E_titled_3x3_third_order.pdf')
plt.show()
