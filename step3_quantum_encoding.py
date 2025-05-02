import pennylane as qml
import numpy as np


def create_quantum_encoding(n_features, encoding_type='angle'):
    n_qubits = min(n_features, 10)  # Limit to 10 qubits max
    wires = list(range(n_qubits))
    if encoding_type == 'angle':
        def encoding_function(x):
            x = np.array(x[:n_qubits])  # Réduire la taille de x
            qml.AngleEmbedding(features=x, wires=wires, rotation='Y')
        print(f"Created AngleEmbedding with {n_qubits} qubits")

    elif encoding_type == 'amplitude':
        def encoding_function(x):
            qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True)
        print(f"Created AmplitudeEmbedding with {n_qubits} qubits")

    elif encoding_type == 'qaoa':
        def encoding_function(x):
            weights = np.random.uniform(0, np.pi, size=(2, 3))
            qml.QAOAEmbedding(features=x, weights=weights, wires=wires)
        print(f"Created QAOAEmbedding with {n_qubits} qubits")

    else:
        raise ValueError(
            "Unsupported encoding type. Choose from "
            "'angle', 'amplitude', or 'qaoa'."
        )

    return encoding_function
