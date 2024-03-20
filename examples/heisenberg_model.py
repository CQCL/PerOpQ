import copy
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rich
import scipy.sparse as sp
from peropq.exact_diagonalization import ExactDiagonalization
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary

Bondaryconditions = Literal["periodic", "open"]


def expectation_value(state: sp.spmatrix, obs_list: list[sp.spmatrix]) -> list[float]:
    """
    return the expecation value of observables.

    param: state on which one want to evaluate the expectation value.
    param: obs_list list of observables of which one wants the expectation value.
    """
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj() @ obs @ state)
    return expectation_value_list


def get_pauli_string_lists(
    n: float,
) -> tuple[list[PauliString], list[PauliString], list[PauliString]]:
    """
    return the list of sparse Pauli matrices.

    param: n system size.
    """
    z_list: list[PauliString] = []
    x_list: list[PauliString] = []
    y_list: list[PauliString] = []
    for i in range(n):
        zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
        z_list.append(zi)
        xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
        x_list.append(xi)
        yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
        y_list.append(yi)
    return x_list, y_list, z_list


def get_square_heisenberg_model(
    n_x: float,
    bondary_condition: Bondaryconditions = "open",
) -> Hamiltonian:
    r"""
    Get the square Heisenberg Hamiltonian.
    """
    n = n_x * n_x
    x_list, y_list, z_list = get_pauli_string_lists(n)
    bc_modifier = 0 if bondary_condition == "periodic" else 1
    term_list = []
    # vertical bonds
    for col in range(n_x):
        v_list = []
        for site in range(col, n + 1 - n_x + col, n_x):
            v_list.append(site)
        for isite in range(len(v_list) - bc_modifier):
            term_list.append(z_list[v_list[isite]]* z_list[v_list[(isite + 1) % len(v_list)]])
            term_list.append(x_list[v_list[isite]]* x_list[v_list[(isite + 1) % len(v_list)]])
            term_list.append(y_list[v_list[isite]]* y_list[v_list[(isite + 1) % len(v_list)]])

    # horizontal bonds
    start_sites = []
    for site in range(0, n + 1 - n_x, n_x):
        start_sites.append(site)
    for site in start_sites:
        for col in range(n_x - bc_modifier):
            term_list.append(
                z_list[site + col] * z_list[site + (col + 1) % (n_x)],
            )
            term_list.append(
                x_list[site + col] * x_list[site + (col + 1) % (n_x)],
            )
            term_list.append(
                y_list[site + col] * y_list[site + (col + 1) % (n_x)],
            )
    return Hamiltonian(pauli_string_list=term_list)


# Ising model
final_time = 0.1
n_x = 3
n = n_x * n_x
h_ising = get_square_heisenberg_model(n_x=n_x)
ed = ExactDiagonalization(number_of_qubits=n_x * n_x)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Get list of single Z string
x_list, y_list, z_list = get_pauli_string_lists(n)
z_list_sparse = []
for site in range(n - 1):
    z_list_sparse.append(ed.get_sparse(z_list[site]*z_list[site+1]))
z_t_continous = []
z_t_trotter = []
z_t_variational = []
energy_trotter = []
energy_variational = []
# Get the observable for the continuous time evolution
np.random.seed(1)
state_init = np.array([0.0 + 0.0j] + [0.0] * (2**n - 1))
random_integer = np.random.randint(low=0,high=2**n)
state_init[random_integer]=1.0
energy = state_init.T @ h_ising_matrix @ state_init
nlayer = 3

# Start the optimization
variational_unitary = VariationalUnitary(
    h_ising,
    number_of_layer=nlayer,
    time=final_time,
)
variational_unitary.set_theta_to_trotter()
trotter_unitary = copy.deepcopy(variational_unitary)
opt = Optimizer()
trotter_unitary.set_theta_to_trotter()
res = opt.optimize(
    variational_unitary=variational_unitary,
)
state_variational = copy.deepcopy(state_init)
state_trotter = copy.deepcopy(state_init)
state_continuous = copy.deepcopy(state_init)
# Check that the norm is better for the variational unitary
trotter_error = ed.get_error(trotter_unitary, hamiltonian=h_ising)
variational_error = ed.get_error(
    variational_unitary=variational_unitary,
    hamiltonian=h_ising,
)
rich.print("trotter error", trotter_error)
rich.print("variational error order 2", variational_error)
for _ in range(7):
    state_trotter = ed.apply_variational_to_state(trotter_unitary, state_trotter)
    state_continuous = ed.apply_continuous_to_state(
        hamiltonian=h_ising,
        time=final_time,
        state=state_continuous,
    )
    state_variational = ed.apply_variational_to_state(
        variational_unitary,
        state_variational,
    )

    sz_expectation_value_trotter = expectation_value(state_trotter, z_list_sparse)
    sz_expectation_value_continuous = expectation_value(state_continuous, z_list_sparse)
    sz_expectation_value_variational = expectation_value(
        state_variational,
        z_list_sparse,
    )

    z_t_continous.append(sz_expectation_value_continuous)
    z_t_trotter.append(sz_expectation_value_trotter)
    z_t_variational.append(sz_expectation_value_variational)

    energy_trotter.append(state_trotter.T.conj() @ h_ising_matrix @ state_trotter)
    energy_variational.append(
        state_variational.T.conj() @ h_ising_matrix @ state_variational,
    )
    print("energy ",state_continuous.T.conj()@h_ising_matrix@state_continuous)

z_array_continuous = np.array(z_t_continous)
z_array_trotter = np.array(z_t_trotter)
z_array_variational = np.array(z_t_variational)
energy_trotter = np.array(energy_trotter)
energy_variational = np.array(energy_variational)
# Plot the error on the energy
plt.figure()
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_trotter[:, int(n / 2)]),
    linestyle="--",
    marker="o",
    label="err trotter",
)
plt.plot(
    np.abs(z_array_continuous[:, int(n / 2)] - z_array_variational[:, int(n / 2)]),
    label="variational",
)
plt.xlabel("time step")
plt.ylabel("error " + r"$Z_{i} Z_{i+1}$")
plt.legend(loc="best")
plt.savefig("correlation_error.pdf")
plt.figure()
plt.plot(np.abs(np.array(energy_trotter) - energy), label="trotter")
plt.plot(np.abs(np.array(energy_variational) - energy), label="variational")
plt.xlabel("time step")
plt.ylabel("energy error")
plt.legend(loc="best")
plt.savefig("energy_error.pdf")
plt.show()