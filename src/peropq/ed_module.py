import scipy.sparse as sp
import numpy as np
from numpy import typing as npt
from peropq.hamiltonian import Hamiltonian
from peropq.variational_unitary import VariationalUnitary
from peropq.pauli import Pauli, PauliString
from collections.abc import Sequence

class ExactDiagonalization:
    def __init__(self,number_of_qubits:int):
        """
        Initializing the ed module

        param: qubit_number number of qubits 
        """
        self.number_of_qubits = number_of_qubits

    def pauli_to_sparse(self,pauli:Pauli)->sp.spmatrix:
        """
        Converts a Pauli to a 2x2 sparse matrix

        param: pauli to be converted
        """
        match pauli:
            case Pauli.I:
                return sp.csc_matrix([[1.0,0],[0,1.0]],dtype=complex)
            case Pauli.X:
                return sp.csc_matrix([[0,1.0],[1.0,0]],dtype=complex)
            case Pauli.Y:
                return sp.csc_matrix([[0,-1j],[1j,0]],dtype=complex)
            case Pauli.Z:
                return sp.csc_matrix([[1.0,0.0],[0.0,-1.0]],dtype=complex)
        
    def get_sparse(self,pauli_string:PauliString)->sp.spmatrix:
        """
        Transforms PauliString into sparse matrix     
        
        param: pauli_string to be transformed into a sparse matrix
        """
        sparse_string:sp.spmatrix = pauli_string.coefficient*self.pauli_to_sparse(pauli_string.get_pauli(0))
        for qubit in range(1,self.number_of_qubits):
            sparse_string = sp.kron(sparse_string,self.pauli_to_sparse(pauli_string.get_pauli(qubit)))
        return sp.csc_matrix(sparse_string)
                        
        

    def get_continuous_time_evolution(self,hamiltonian:Hamiltonian,time:float)->npt.ArrayLike:
        """ 
        Get the continuous time evolution of an Hamiltonian.

        param: hamiltonian Which needs to be time evolved.
        param: time at which we want to time evolve.
        """
        hamiltonian_matrix = self.get_sparse(hamiltonian.pauli_string_list[0])
        for i_string in range(1,len(hamiltonian.pauli_string_list)):
            hamiltonian_matrix+= self.get_sparse(hamiltonian.pauli_string_list[i_string])
        return sp.linalg.expm(-1j*time*hamiltonian_matrix)

    def get_variational_evolution(self,variational_unitary:VariationalUnitary)->npt.ArrayLike:
        """
        Get the time evolution unitary from a variational unitary

        param: variational_unitary to be evolved  
        """

        u_sparse = sp.eye(2**self.number_of_qubits)
        for layer in range(variational_unitary.depth):
            for i_term,term in enumerate(variational_unitary.pauli_string_list):
                u_sparse = u_sparse @ sp.linalg.expm(-1j*variational_unitary.theta[layer,i_term]*self.get_sparse(term))
        return u_sparse

string1 = PauliString.from_pauli_sequence([Pauli.X, Pauli.Y], 1.0)
ed  = ExactDiagonalization(2)
h = Hamiltonian(pauli_string_list=[string1])
print(type(ed.get_continuous_time_evolution(h,2.0)))

z_list: list[PauliString] = []
x_list: list[PauliString] = []
y_list: list[PauliString] = []
n = 4
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
variational_unitary.set_theta_to_trotter()
ed = ExactDiagonalization(number_of_qubits=4)
print(ed.get_variational_evolution(variational_unitary))
