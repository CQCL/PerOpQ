import numpy as np
import scipy
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary
from peropq.ed_module import ExactDiagonalization as ED
import matplotlib.pyplot as plt
import pickle
import copy


def expectation_value(state,obs_list):
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj()@obs@state)
    return expectation_value_list
        
# Choose the mode
norm_mode=False
if norm_mode:
    observable_mode=False
else:
    observable_mode=True

z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
bc_modifier = 1
n = 9
nx = 3
ny = 3
for i in range(n):
    zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
    z_list.append(zi)
    xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
    x_list.append(xi)
    yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
    y_list.append(yi)
term_list = []
for i in range(n):
    term_list.append(1.0 * z_list[i])
    term_list.append(1.0 * x_list[i])
V = -1
# vertical bonds
for col in range(0, nx):
    v_list = []
    for site in range(col,n + 1 - nx + col, nx):
        v_list.append(site)
    print("vertical bonds")
    for isite in range(len(v_list) - bc_modifier):
        term_list.append(
            z_list[v_list[isite]] * z_list[v_list[(isite + 1) % len(v_list)]]
        )
        print(v_list[isite], v_list[(isite + 1) % len(v_list)])

# horizontal bonds
start_sites = []
for site in range(0, n + 1 - nx, nx):
    start_sites.append(site)
print("horizontal bonds")
for site in start_sites:
    for col in range(0, nx - bc_modifier):
        term_list.append(z_list[site + col] * z_list[site + (col + 1) % (nx)])
        print((site + col), site + (col + 1) % (nx))

# Ising model
h_ising = Hamiltonian(pauli_string_list=term_list)
time_list=[0.3]
ed =  ED(number_of_qubits=n)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)
if norm_mode:
    trotter_error_list = []
    variational_error_list = []
    for time in time_list:
        variational_unitary = VariationalUnitary(h_ising, number_of_layer=3, time=time)
        variational_unitary.set_theta_to_trotter()
        trotter_error = ed.get_error(variational_unitary=variational_unitary,hamiltonian=h_ising)
        opt = Optimizer()
        variational_unitary.c2_squared(
            variational_unitary.get_initial_trotter_vector(),
        )
        res = opt.optimize(variational_unitary)
        variational_error = ed.get_error(variational_unitary=variational_unitary,hamiltonian=h_ising)
        trotter_error_list.append(trotter_error)
        variational_error_list.append(variational_error)
        print("trotter_error ",trotter_error)
        print("variational_error",variational_error)

    plt.figure()
    plt.plot(time_list,trotter_error_list,label='trotter',marker = 'o')
    plt.plot(time_list,variational_error_list,label='variational',marker = 'o')
    plt.xlabel('final time')
    plt.ylabel('error')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(time_list,trotter_error_list,label='trotter',marker = 'o')
    plt.plot(time_list,variational_error_list,label='variational',marker = 'o')
    plt.xlabel('final time')
    plt.ylabel('error')
    plt.legend(loc='best')
    plt.show()

if observable_mode:
    # Get list of single Z string
    z_list_sparse = []
    for site in range(n-1):
        z_list_sparse.append(ed.get_sparse(x_list[site]*x_list[site+1]))
    z_t_continous = []
    z_t_trotter = []
    z_t_variational = []
    energy_trotter = []
    energy_variational = []
    # Get the observable for the continuous time evolution
    state_init = np.array([1.0+0.0j]+[0.0]*(2**n-1))
    energy = state_init.T@h_ising_matrix@state_init
    for time in time_list:
        variational_unitary = VariationalUnitary(h_ising, number_of_layer=3, time=time)
        variational_unitary.set_theta_to_trotter()
        trotter_unitary = copy.deepcopy(variational_unitary)
        opt = Optimizer()
        res,variational_unitary = opt.optimize_steps(hamiltonian=h_ising,final_time=time,order=2,dt_optimize=0.1)

    state_variational = copy.deepcopy(state_init)
    state_trotter= copy.deepcopy(state_init)
    state_continuous = copy.deepcopy(state_init)
    # Check that the norm is better for the variational unitary
    trotter_error = ed.get_error(trotter_unitary,hamiltonian=h_ising)
    variational_error = ed.get_error(variational_unitary=variational_unitary,hamiltonian=h_ising)
    print("trotter error", trotter_error)
    print("variational error", variational_error)

    for istep in range(10):
        state_trotter = ed.apply_variational_to_state(trotter_unitary,state_trotter)
        state_continuous = ed.apply_continuous_to_state(hamiltonian=h_ising,time=time,state = state_continuous)
        state_variational = ed.apply_variational_to_state(variational_unitary,state_variational)

        sz_expectation_value_trotter= expectation_value(state_trotter,z_list_sparse)
        sz_expectation_value_continuous= expectation_value(state_continuous,z_list_sparse)
        sz_expectation_value_variational = expectation_value(state_variational,z_list_sparse)

        z_t_continous.append(sz_expectation_value_continuous)
        z_t_trotter.append(sz_expectation_value_trotter)
        z_t_variational.append(sz_expectation_value_variational)

        energy_trotter.append(state_trotter.T.conj()@h_ising_matrix@state_trotter)
        energy_variational.append(state_variational.T.conj()@h_ising_matrix@state_variational)

    z_array_continuous = np.array(z_t_continous) 
    z_array_trotter = np.array(z_t_trotter) 
    z_array_variational = np.array(z_t_variational)
    energy_trotter = np.array(energy_trotter)
    energy_variational= np.array(energy_variational)

    plt.figure()
    plt.plot(z_array_continuous[:,0],label='continuous')
    plt.plot(z_array_trotter[:,0],label='trotter')
    plt.plot(z_array_variational[:,0],label ='variational')
    plt.legend(loc='best')
    plt.xlabel('step')
    plt.ylabel('Z(0)')
    plt.figure()
    plt.plot(z_array_continuous[:,int(n/2)],label='continuous')
    plt.plot(z_array_trotter[:,int(n/2)],label='trotter')
    plt.plot(z_array_variational[:,int(n/2)],label ='variational')
    plt.xlabel('time step')
    plt.ylabel(r'$\langle Z(L/2) \rangle$')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(np.abs(z_array_continuous[:,int(n/2)]-z_array_trotter[:,int(n/2)]),label='err trotter')
    plt.plot(np.abs(z_array_continuous[:,int(n/2)]-z_array_variational[:,int(n/2)]),label ='variational')
    plt.xlabel('time step')
    plt.ylabel('error Z(L/2)')
    plt.legend(loc='best')
    plt.show()
    plt.figure()
    plt.plot(np.abs(np.array(energy_trotter)-energy),label = 'trotter')
    plt.plot(np.abs(np.array(energy_variational)-energy),label = 'variational')
    plt.xlabel('time step')
    plt.ylabel('energy error')
    plt.legend(loc='best')
    plt.show()
        
