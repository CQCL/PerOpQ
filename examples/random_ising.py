import numpy as np
from numpy import typing as npt
import scipy
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli_bitstring import PauliString, pauli_from_string
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary
from peropq.ed_module import ExactDiagonalization as ED
import matplotlib.pyplot as plt
import pickle


def expectation_value(state, obs_list):
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj()@obs@state)
    return expectation_value_list


def get_observale_different_times(state_init: npt.NDArray, variational_unitary: UnconstrainedVariationalUnitary, hamiltonian_matrix: scipy.sparse.spmatrix, obs_list: list, n_steps: int) -> tuple[list, list]:
    """
    return the energy of hamiltonian and the expectation values of obs_list when applying variational_unitary to state_init n_steps time.
    """
    expectation_value_list = []
    energy_list = []
    evolved_state = state_init.copy()
    for step in range(n_steps):
        evolved_state = ed.apply_variational_to_state(
            variational_unitary, evolved_state)
        obs_expectations = expectation_value(
            evolved_state, obs_list)
        expectation_value_list.append(obs_expectations)
        energy_list.append(evolved_state.T.conj() @
                           hamiltonian_matrix@evolved_state)
    return energy_list, expectation_value_list


def get_continuous_observables(state_init: npt.NDArray, hamiltonian: Hamiltonian, obs_list: list, time: float, n_steps: int):
    expectation_value_list = []
    energy_list = []
    state_continuous = state_init.copy()
    hamiltonian_matrix = ed.get_hamiltonian_matrix(hamiltonian=hamiltonian)
    for n in range(n_steps):
        state_continuous = ed.apply_continuous_to_state(
            hamiltonian=h_ising, time=time, state=state_continuous)
        expectation_value_list.append(expectation_value(
            state_continuous, obs_list))
        energy_list.append((state_continuous.T.conj() @
                           hamiltonian_matrix@state_continuous))
    return energy_list, expectation_value_list


# Choose the mode
# norm_mode = True
norm_mode = False
if norm_mode:
    observable_mode = False
else:
    observable_mode = True

z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
bc_modifier = 1
nx = 3
ny = 3
length = nx*ny
n = nx*ny
for i in range(n):
    zi = pauli_from_string(string='Z', length=length, start_qubit=i)
    xi = pauli_from_string(string='X', length=length, start_qubit=i)
    yi = pauli_from_string(string='Y', length=length, start_qubit=i)
    z_list.append(zi)
    x_list.append(xi)
    y_list.append(yi)
term_list = []
for i in range(n):
    term_list.append(1.0 * z_list[i])
    term_list.append(1.0 * x_list[i])
V = -1
# vertical bonds
for col in range(0, nx):
    v_list = []
    for site in range(col, n + 1 - nx + col, nx):
        v_list.append(site)
    print("vertical bonds")
    for isite in range(len(v_list) - bc_modifier):
        term_list.append(
            z_list[v_list[isite]] * z_list[v_list[(isite + 1) % len(v_list)]]
        )
        print(v_list[isite], v_list[(isite + 1) % len(v_list)])

# horizontal bonds
start_sites = []
for site in range(0, n + 1 - ny, ny):
    start_sites.append(site)
print("horizontal bonds")
for site in start_sites:
    for col in range(0, ny - bc_modifier):
        term_list.append(z_list[site + col] * z_list[site + (col + 1) % (ny)])
        print((site + col), site + (col + 1) % (ny))

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
time = 0.3
nlayer = 3
if nx < 4:
    ed = ED(number_of_qubits=n)
try:
    h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)
except:
    pass
if norm_mode:
    for order in [2, 3, 4]:
        trotter_error_list = []
        variational_error_list = []
        print("order ", order)
        variational_unitary = UnconstrainedVariationalUnitary(
            h_ising, number_of_layer=nlayer, time=time)
        variational_unitary.set_theta_to_trotter()
        try:
            trotter_error = ed.get_error(
                variational_unitary=variational_unitary, hamiltonian=h_ising)
        except:
            pass
        opt = Optimizer()
        res = opt.optimize_arbitrary(
            variational_unitary=variational_unitary,
            order=order,
            unconstrained=True,
            tol=5e-4
        )
        try:
            variational_error = ed.get_error(
                variational_unitary=variational_unitary,
                hamiltonian=h_ising,
            )
            variational_error_list.append(variational_error)
            print("trotter_error ", trotter_error)
            print("variational_error", variational_error_list[-1])
        except Exception as e:
            print(e)

if observable_mode:
    # Number of steps for the time evolution
    n_steps = 10
    # Get list of single Z string
    z_list_sparse = []
    for site in range(n):
        z_list_sparse.append(ed.get_sparse(z_list[site]))
    # z_t_continous = []
    z_t_trotter = []
    z_t_variational = []
    energy_trotter = []
    energy_variational = []

    # Get the observable for the continuous time evolution
    state_init = np.array([1.0+0.0j]+[0.0]*(2**n-1))
    energy = state_init.T@h_ising_matrix@state_init

    # Exact
    energy_continuous, z_t_continuous = get_continuous_observables(
        state_init=state_init, hamiltonian=h_ising, time=time, obs_list=z_list_sparse, n_steps=n_steps)

    # Trotter
    variational_unitary = UnconstrainedVariationalUnitary(
        h_ising, number_of_layer=nlayer, time=time)
    variational_unitary.set_theta_to_trotter()
    # Do the time evolution and get expectation values
    energy_trotter, z_t_trotter = get_observale_different_times(
        state_init=state_init, variational_unitary=variational_unitary, hamiltonian_matrix=h_ising_matrix, obs_list=z_list_sparse, n_steps=n_steps)
    # Get the norm error
    trotter_error = ed.get_error(
        variational_unitary=variational_unitary, hamiltonian=h_ising)

    # Variational
    energy_order: dict = {}
    z_t_order: dict = {}
    norm_error_order: dict = {}
    for order in [2, 3, 4]:
        variational_unitary = UnconstrainedVariationalUnitary(
            h_ising, number_of_layer=nlayer, time=time)
        opt = Optimizer()
        res = opt.optimize_arbitrary(
            variational_unitary=variational_unitary,
            order=order,
            unconstrained=True,
            tol=5e-4
        )
        energy, z_t = get_observale_different_times(
            state_init=state_init, variational_unitary=variational_unitary, hamiltonian_matrix=h_ising_matrix, obs_list=z_list_sparse, n_steps=n_steps)
        z_t_order[order] = z_t
        energy_order[order] = energy
        norm_error = ed.get_error(
            variational_unitary=variational_unitary, hamiltonian=h_ising)
        norm_error_order[order] = norm_error

# Make the plot
plt.figure()
plt.title('Magnetisation')
plt.plot(np.abs(np.sum(np.array(z_t_trotter), axis=1) -
         np.sum(np.array(z_t_continuous), axis=1)), label='trotter'+" err="+str(trotter_error))
for order in [2, 3, 4]:
    plt.plot(np.abs(np.sum(np.array(z_t_order[order]), axis=1)-np.sum(
        np.array(z_t_continuous), axis=1)), label='order'+str(order)+" err="+str(norm_error_order[order]))
plt.legend(loc='best')
plt.show()

plt.figure()
plt.title('Energy')
plt.plot(np.abs(np.array(energy_continuous) -
         np.array(energy_trotter)), label='trotter')
for order in [2, 3, 4]:
    plt.plot(np.abs(np.array(
        energy_order[order])-np.array(energy_trotter)), label="order"+str(order))
plt.legend(loc='best')
plt.show()
