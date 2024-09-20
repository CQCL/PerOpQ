import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import typing as npt
from peropq.ed_module import ExactDiagonalization
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli_bitstring import PauliString, pauli_from_string
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary

r"""
This scipt demonstrates how to use PerOpQ to optimize the time-evolution circuit for the two dimensional Ising model:
`$\sum_{<i,j>} Z_i Z_j +\sum_i X_i$`.
The optimization is performed at orders 2,3 and 4.
Then, the results of the optimized circuits are compared to the Trotterize time evolution and to the continuous time evolution.
"""


def expectation_value(
    state: npt.NDArray,
    obs_list: list[scipy.sparse.spmatrix],
) -> list:
    """
    Get the expectation value.

    :param state: wave-function on which the observables need to be measured
    :param obs_list: list of observables
    :return: list of expectation value
    """
    expectation_value_list: list[complex] = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj() @ obs @ state)
    return expectation_value_list


def get_observale_different_times(
    state_init: npt.NDArray,
    variational_unitary: UnconstrainedVariationalUnitary,
    hamiltonian_matrix: scipy.sparse.spmatrix,
    obs_list: list[scipy.sparse.spmatrix],
    n_steps: int,
) -> tuple[list, list]:
    """Return the energy of hamiltonian and the expectation values of obs_list when applying variational_unitary to state_init n_steps time."""
    expectation_value_list = []
    energy_list = []
    evolved_state = state_init.copy()
    for _ in range(n_steps):
        evolved_state = ed.apply_variational_to_state(
            variational_unitary,
            evolved_state,
        )
        obs_expectations = expectation_value(evolved_state, obs_list)
        expectation_value_list.append(obs_expectations)
        energy_list.append(evolved_state.T.conj() @ hamiltonian_matrix @ evolved_state)
    return energy_list, expectation_value_list


def get_continuous_observables(
    state_init: npt.NDArray,
    hamiltonian: Hamiltonian,
    obs_list: list,
    time: float,
    n_steps: int,
) -> tuple[list, list]:
    """Get the expectation value for observable under exact, continuous time evolution. Useful to compare with the results obtained from variational unitaries."""
    expectation_value_list = []
    energy_list = []
    state_continuous = state_init.copy()
    hamiltonian_matrix = ed.get_hamiltonian_matrix(hamiltonian=hamiltonian)
    for _ in range(n_steps):
        state_continuous = ed.apply_continuous_to_state(
            hamiltonian=h_ising,
            time=time,
            state=state_continuous,
        )
        expectation_value_list.append(expectation_value(state_continuous, obs_list))
        energy_list.append(
            state_continuous.T.conj() @ hamiltonian_matrix @ state_continuous,
        )
    return energy_list, expectation_value_list


# Construct the Hamiltonian
# Construct the Pauli strings
z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
# The following line ensures open boundary conditions:
bc_modifier = 1  # Change to 0 for periodic.
# Geometry nx x ny square lattice
nx = 3
ny = 2
length = nx * ny
# Construct the needed Pauli strings
for i in range(length):
    zi = pauli_from_string(string="Z", length=length, start_qubit=i)
    xi = pauli_from_string(string="X", length=length, start_qubit=i)
    yi = pauli_from_string(string="Y", length=length, start_qubit=i)
    z_list.append(zi)
    x_list.append(xi)
    y_list.append(yi)

# Define the Hamiltonian by storing all the terms in term_list
term_list = []
for i in range(length):
    term_list.append(1.0 * z_list[i])
    term_list.append(1.0 * x_list[i])
V = -1
# Vertical bonds
for col in range(nx):
    v_list = []
    for site in range(col, length + 1 - nx + col, nx):
        v_list.append(site)
    for isite in range(len(v_list) - bc_modifier):
        term_list.append(
            z_list[v_list[isite]] * z_list[v_list[(isite + 1) % len(v_list)]],
        )
# Horizontal bonds
start_sites = []
for site in range(0, length + 1 - ny, ny):
    start_sites.append(site)
for site in start_sites:
    for col in range(ny - bc_modifier):
        term_list.append(z_list[site + col] * z_list[site + (col + 1) % (ny)])

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
# Define the exact diagonalization class
ed = ExactDiagonalization(number_of_qubits=length)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Parameters for the time evolution
# Time for the variational unitary
time = 0.3
# Number of layers to optimize over
nlayer = 3
# Number of steps for the exact time evolution for the benchmark
n_steps = 10
# Get list of single Z string as a sparse matrix
z_list_sparse = []
for site in range(length):
    z_list_sparse.append(ed.get_sparse(z_list[site]))
z_t_trotter = []
z_t_variational = []
energy_trotter = []
energy_variational = []
# Get the observable for the continuous time evolution
state_init = np.array([1.0 + 0.0j] + [0.0] * (2**length - 1))
energy = state_init.T @ h_ising_matrix @ state_init

# Exact
energy_continuous, z_t_continuous = get_continuous_observables(
    state_init=state_init,
    hamiltonian=h_ising,
    time=time,
    obs_list=z_list_sparse,
    n_steps=n_steps,
)

# Trotter circuit
variational_unitary = UnconstrainedVariationalUnitary(
    h_ising,
    number_of_layer=nlayer,
    time=time,
)
# Set the parameter to Trotter (no optimization here).
variational_unitary.set_theta_to_trotter()
# Do the time evolution and get expectation values
energy_trotter, z_t_trotter = get_observale_different_times(
    state_init=state_init,
    variational_unitary=variational_unitary,
    hamiltonian_matrix=h_ising_matrix,
    obs_list=z_list_sparse,
    n_steps=n_steps,
)
# Get the norm error
trotter_error = ed.get_error(
    variational_unitary=variational_unitary,
    hamiltonian=h_ising,
)

# Optimization of the variational circuit
energy_order: dict = {}
z_t_order: dict = {}
norm_error_order: dict = {}
order_list = [2, 3, 4]
for order in order_list:
    # Define the ansatz
    variational_unitary = UnconstrainedVariationalUnitary(
        h_ising,
        number_of_layer=nlayer,
        time=time,
    )
    # Define the optimizer
    opt = Optimizer()
    # Run the optimization
    res = opt.optimize_arbitrary(
        variational_unitary=variational_unitary,
        order=order,
        unconstrained=True,
        tol=5e-4,
    )
    # Obtain observables to compare with Trotter and continuous time evolution
    energy, z_t = get_observale_different_times(
        state_init=state_init,
        variational_unitary=variational_unitary,
        hamiltonian_matrix=h_ising_matrix,
        obs_list=z_list_sparse,
        n_steps=n_steps,
    )
    z_t_order[order] = z_t
    energy_order[order] = energy
    norm_error = ed.get_error(
        variational_unitary=variational_unitary,
        hamiltonian=h_ising,
    )
    norm_error_order[order] = norm_error

# Make the plot
plt.figure()
plt.title("Magnetisation")
plt.plot(
    np.abs(
        np.sum(np.array(z_t_trotter), axis=1)
        - np.sum(np.array(z_t_continuous), axis=1),
    ),
    label="trotter" + " err=" + str(trotter_error),
)
for order in order_list:
    plt.plot(
        np.abs(
            np.sum(np.array(z_t_order[order]), axis=1)
            - np.sum(np.array(z_t_continuous), axis=1),
        ),
        label="order" + str(order) + " err=" + str(norm_error_order[order]),
    )
plt.legend(loc="best")
plt.xlabel("number of steps")
plt.ylabel("error on the magnetisation")
plt.show()
plt.savefig("magnetisation.pdf")

plt.figure()
plt.title("Energy")
plt.plot(
    np.abs(np.array(energy_continuous) - np.array(energy_trotter)),
    label="trotter",
)
for order in order_list:
    plt.plot(
        np.abs(np.array(energy_order[order]) - np.array(energy_trotter)),
        label="order" + str(order),
    )
plt.legend(loc="best")
plt.xlabel("number of steps")
plt.ylabel("error on the energy")
plt.show()
plt.savefig("energy.pdf")
