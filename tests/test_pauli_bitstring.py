import pytest
from peropq import pauli_bitstring as pb
import rich

def test_single_pauli_mult() -> None:
    assert pb.pauli_from_string(string='I',length=1)*pb.pauli_from_string(string='I',length=1)==pb.pauli_from_string(string='I',length=1)
    assert pb.pauli_from_string(string='X',length=1)*pb.pauli_from_string(string='X',length=1)==pb.pauli_from_string(string='I',length=1)
    assert pb.pauli_from_string(string='Y',length=1)*pb.pauli_from_string(string='Y',length=1)==pb.pauli_from_string(string='I',length=1)
    assert pb.pauli_from_string(string='X',length=1)*pb.pauli_from_string(string='I',length=1)==pb.pauli_from_string(string='X',length=1)
    ###
    assert pb.pauli_from_string(string='Y',length=1)*pb.pauli_from_string(string='I',length=1)==pb.pauli_from_string(string='Y',length=1)
    assert pb.pauli_from_string(string='Z',length=1)*pb.pauli_from_string(string='I',length=1)==pb.pauli_from_string(string='Z',length=1)
    assert pb.pauli_from_string(string='I',length=1)*pb.pauli_from_string(string='X',length=1)==pb.pauli_from_string(string='X',length=1)
    assert pb.pauli_from_string(string='X',length=1)*pb.pauli_from_string(string='Y',length=1)==pb.pauli_from_string(string='Z',length=1,coefficient=1j)
    assert pb.pauli_from_string(string='Y',length=1)*pb.pauli_from_string(string='X',length=1)==pb.pauli_from_string(string='Z',length=1,coefficient=-1j)
    assert pb.pauli_from_string(string='Y',length=1)*pb.pauli_from_string(string='Z',length=1)==pb.pauli_from_string(string='X',length=1,coefficient=1j)
    assert pb.pauli_from_string(string='Z',length=1)*pb.pauli_from_string(string='Z',length=1)==pb.pauli_from_string(string='I',length=1,coefficient=1.0)
    assert pb.pauli_from_string(string='Z',length=1)*pb.pauli_from_string(string='X',length=1)==pb.pauli_from_string(string='Y',length=1,coefficient=1j)


def test_pauli_string_mult() -> None:
    assert pb.pauli_from_string(string='XXZ',length=3)*pb.pauli_from_string(string='YYX',length=3)==pb.pauli_from_string(string='ZZY',length=3,coefficient=-1j)

    assert pb.pauli_from_string(string='XZ',length=3,start_qubit=1)*pb.pauli_from_string(string='YYX',length=3)==pb.pauli_from_string(string='YZY',length=3,coefficient=-1.0)
test_single_pauli_mult()
test_pauli_string_mult()
