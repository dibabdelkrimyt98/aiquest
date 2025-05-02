import pennylane as qml
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np


def create_quantum_kernel(encoding_function, X_train, X_test=None):
    n_qubits = X_train.shape[1]
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        encoding_function(x1)
        qml.adjoint(encoding_function)(x2)
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        probs = kernel_circuit(x1, x2)
        return np.sum(probs)

    # Compute kernel matrix for training data
    kernel_matrix_train = pairwise_kernels(X_train, metric=kernel)

    if X_test is not None:
        # Compute kernel matrix between test and training data
        kernel_matrix_test = pairwise_kernels(X_test, X_train, metric=kernel)
        return kernel_matrix_train, kernel_matrix_test

    return kernel_matrix_train
