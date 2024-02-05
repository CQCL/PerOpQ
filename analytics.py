from peropq.pauli import Pauli,PauliString
from peropq import commutators

x = PauliString.from_pauli_sequence([Pauli.I,Pauli.X])
xy= PauliString.from_pauli_sequence([Pauli.Y,Pauli.Z])
print(commutators.get_commutator_pauli_tensors(x,xy))
