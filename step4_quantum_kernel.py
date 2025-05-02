import pennylane as qml
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np


def create_quantum_kernel(encoding_function, X_train, X_test=None):
    print("Starting kernel creation...")
    n_qubits = X_train.shape[1]
    print(f"Number of qubits: {n_qubits}")
    
    try:
        dev = qml.device('default.qubit', wires=n_qubits)
        print("Quantum device created")
        
        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            print(f"Circuit input shapes: x1={x1.shape}, x2={x2.shape}")
            encoding_function(x1)
            qml.adjoint(encoding_function)(x2)
            return qml.probs(wires=range(n_qubits))
            
        print("Quantum circuit defined")
        # ...rest of the code...
    except Exception as e:
        print(f"Error in kernel creation: {str(e)}")
        raise