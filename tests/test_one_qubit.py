from peropq.ed_module import ExactDiagonalization
from peropq.exact_norm import ExactUnitary
from peropq.hamiltonian import Hamiltonian
from peropq.optimizer import Optimizer
from peropq.pauli_bitstring import pauli_from_string
from peropq.unconstrained_variational_unitary import (
    UnconstrainedVariationalUnitary,
)


def test_one_qubit() -> None:
    z = pauli_from_string(string="Z", length=1)
    x = pauli_from_string(string="X", length=1)
    y = pauli_from_string(string="Y", length=1)
    term_list = []
    term_list.append(x)
    term_list.append(y)
    term_list.append(z)

    # Ising model
    h_ising = Hamiltonian(pauli_string_list=term_list)
    time = 0.4
    ed = ExactDiagonalization(number_of_qubits=1)

    # Do the approximate optimization
    variational_error_list = []
    for order in range(2, 5):
        variational_unitary = UnconstrainedVariationalUnitary(
            h_ising,
            number_of_layer=3,
            time=time,
        )
        variational_unitary.set_theta_to_trotter()
        opt = Optimizer()
        opt.optimize_arbitrary(
            variational_unitary=variational_unitary,
            order=order,
            unconstrained=True,
        )
        variational_error = ed.get_error(
            variational_unitary=variational_unitary,
            hamiltonian=h_ising,
        )
        variational_error_list.append(variational_error)

    assert variational_error_list[0] < 0.01
    assert variational_error_list[1] < 0.001
    assert variational_error_list[2] < 1e-04
    ########
    # Do the optimization with ExactUnitary
    exact_unitary = ExactUnitary(
        h_ising,
        number_of_layer=3,
        time=time,
        number_of_qubits=1,
    )
    exact_unitary.set_theta_to_trotter()
    exact_unitary.time = time
    opt = Optimizer()
    theta_flat = exact_unitary.flatten_theta(exact_unitary.theta)
    opt.optimize_exact(
        exact_unitary=exact_unitary,
        initial_guess=theta_flat,
    )

    variational_error = ed.get_error(
        variational_unitary=exact_unitary,
        hamiltonian=h_ising,
    )
    assert variational_error < 1e-5
