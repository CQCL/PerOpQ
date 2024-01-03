import numpy as np
import scipy
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli import Pauli, PauliString
from peropq.variational_unitary import VariationalUnitary as VUC
from peropq.unconstrained_variational_unitary import UnconstrainedVariationalUnitary as VU
from peropq.ed_module import ExactDiagonalization as ED
import matplotlib.pyplot as plt
import pickle
import copy


def expectation_value(state,obs_list):
    expectation_value_list = []
    for obs in obs_list:
        expectation_value_list.append(state.T.conj()@obs@state)
    return expectation_value_list
        

z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
bc_modifier = 1
nx = 3
ny = 3
n = nx*ny
for i in range(n):
    zi = PauliString.from_pauli_sequence(paulis=[Pauli.Z], start_qubit=i)
    z_list.append(zi)
    xi = PauliString.from_pauli_sequence(paulis=[Pauli.X], start_qubit=i)
    x_list.append(xi)
    yi = PauliString.from_pauli_sequence(paulis=[Pauli.Y], start_qubit=i)
    y_list.append(yi)
term_list = []
# for i in range(n):
#     term_list.append(1.0 * z_list[i])
for i in range(n):
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
time_list=[0.4]
ed =  ED(number_of_qubits=n)
h_ising_matrix = ed.get_hamiltonian_matrix(hamiltonian=h_ising)

# Get list of single Z string
z_list_sparse = []
for site in range(n-1):
    z_list_sparse.append(ed.get_sparse(x_list[site]*x_list[site+1]))
z_t_continous = []
z_t_trotter = []
z_t_variational = []
z_t_variational_c = []
energy_trotter = []
energy_variational = []
energy_variational_c = []
# Get the observable for the continuous time evolution
state_init = np.array([1.0+0.0j]+[0.0]*(2**n-1))
energy = state_init.T@h_ising_matrix@state_init
# print("unconstrained ")
# print("-------")
for time in time_list:
    variational_unitary = VU(h_ising, number_of_layer=3, time=time)
    variational_unitary.set_theta_to_trotter()
    # trotter_unitary = copy.deepcopy(variational_unitary)
    opt = Optimizer()
    res = opt.optimize_arbitrary(variational_unitary=variational_unitary,order=2,unconstrained=True)
    print("res ",res)
    c2 = variational_unitary.c2_squared(variational_unitary.theta)
    print("c2 ",c2)
    c2_trotter = variational_unitary.c2_squared(variational_unitary.get_initial_trotter_vector())
    # breakpoint()
# print("------")
# print("constrained")
for time in time_list:
    variational_unitary_c = VUC(h_ising,number_of_layer=3,time=time)
    variational_unitary_c.set_theta_to_trotter()
    trotter_unitary = copy.deepcopy(variational_unitary)
    opt = Optimizer()
    res = opt.optimize_arbitrary(variational_unitary=variational_unitary_c,order=2,unconstrained=False)
    # print("res ",res)
    # c2 = variational_unitary.c2_squared(variational_unitary.theta)
    # print("c2 ",c2)
    # c2_trotter = variational_unitary.c2_squared(variational_unitary.get_initial_trotter_vector())
# breakpoint()

state_variational = copy.deepcopy(state_init)
state_variational_c = copy.deepcopy(state_init)
state_trotter= copy.deepcopy(state_init)
state_continuous = copy.deepcopy(state_init)
# Check that the norm is better for the variational unitary
trotter_error = ed.get_error(trotter_unitary,hamiltonian=h_ising)
variational_error = ed.get_error(variational_unitary=variational_unitary,hamiltonian=h_ising)
variational_c_error = ed.get_error(variational_unitary=variational_unitary_c,hamiltonian=h_ising)
print("trotter error", trotter_error)
print("variational error", variational_error)
print("variational constrained error", variational_c_error)
breakpoint()

for istep in range(10):
    state_trotter = ed.apply_variational_to_state(trotter_unitary,state_trotter)
    state_continuous = ed.apply_continuous_to_state(hamiltonian=h_ising,time=time,state = state_continuous)
    state_variational = ed.apply_variational_to_state(variational_unitary,state_variational)
    state_variational_c = ed.apply_variational_to_state(variational_unitary_c,state_variational_c)

    sz_expectation_value_trotter= expectation_value(state_trotter,z_list_sparse)
    sz_expectation_value_continuous= expectation_value(state_continuous,z_list_sparse)
    sz_expectation_value_variational = expectation_value(state_variational,z_list_sparse)
    sz_expectation_value_variational_c = expectation_value(state_variational_c,z_list_sparse)

    z_t_continous.append(sz_expectation_value_continuous)
    z_t_trotter.append(sz_expectation_value_trotter)
    z_t_variational.append(sz_expectation_value_variational)
    z_t_variational_c.append(sz_expectation_value_variational_c)

    energy_trotter.append(state_trotter.T.conj()@h_ising_matrix@state_trotter)
    energy_variational.append(state_variational.T.conj()@h_ising_matrix@state_variational)
    energy_variational_c.append(state_variational_c.T.conj()@h_ising_matrix@state_variational_c)

z_array_continuous = np.array(z_t_continous) 
z_array_trotter = np.array(z_t_trotter) 
z_array_variational = np.array(z_t_variational)
z_array_variational_c = np.array(z_t_variational_c)
energy_trotter = np.array(energy_trotter)
energy_variational= np.array(energy_variational)
energy_variational_c= np.array(energy_variational_c)

plt.figure()
plt.plot(z_array_continuous[:,0],label='continuous')
plt.plot(z_array_trotter[:,0],label='trotter')
plt.plot(z_array_variational[:,0],label ='variational')
# plt.plot(z_array_variational_c[:,0],label ='variational constrained')
plt.legend(loc='best')
plt.figure()
plt.plot(z_array_continuous[:,int(n/2)],label='continuous')
plt.plot(z_array_trotter[:,int(n/2)],label='trotter')
plt.plot(z_array_variational[:,int(n/2)],label ='variational')
# plt.plot(z_array_variational_c[:,int(n/2)],label ='variational constrained')
plt.xlabel('time step')
plt.ylabel(r'$\langle Z \rangle$')
plt.legend(loc='best')
plt.figure()
plt.plot(np.abs(z_array_continuous[:,int(n/2)]-z_array_trotter[:,int(n/2)]),linestyle='--',marker = 'o',label='err trotter')
plt.plot(np.abs(z_array_continuous[:,int(n/2)]-z_array_variational[:,int(n/2)]),label ='variational')
# plt.plot(np.abs(z_array_continuous[:,int(n/2)]-z_array_variational_c[:,int(n/2)]),label ='variational constrained')
plt.xlabel('time step')
plt.ylabel('error')
plt.legend(loc='best')
plt.show()
plt.figure()
plt.plot(np.abs(np.array(energy_trotter)-energy),label = 'trotter')
plt.plot(np.abs(np.array(energy_variational)-energy),label = 'variational')
# plt.plot(np.abs(np.array(energy_variational_c)-energy),label = 'variational constrained')
plt.xlabel('time step')
plt.ylabel('energy error')
plt.legend(loc='best')
plt.show()
    
