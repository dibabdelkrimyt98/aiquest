import pandas as pd
import pennylane as qml
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

class QuantumModel:
    def __init__(self, n_features=None, n_qubits=None, n_layers=2,
                 model_path='models/quantum/quantum_model.pkl',
                 scaler_path='models/quantum/scaler.pkl',
                 feature_map_path='models/quantum/feature_map.pkl',
                 kernel_path='models/quantum/kernel.pkl'):
        """Initialize the quantum predictor with all necessary components"""
        
        # Store model architecture parameters
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits is not None else n_features
        self.n_layers = n_layers
        
        # Initialize random weights if creating new model
        if n_features is not None:
            self.weights = np.random.randn(self.n_layers, self.n_qubits, 3)
            self._create_quantum_circuit()
            return

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Check if files exist, create default ones if missing
        if not all(os.path.exists(f) for f in [model_path, scaler_path, feature_map_path, kernel_path]):
            print("\nCreating missing model files...")
            
            if not os.path.exists(scaler_path):
                default_scaler = StandardScaler()
                joblib.dump(default_scaler, scaler_path)
                print(f"Created default scaler at {scaler_path}")
            
            if not os.path.exists(model_path):
                from sklearn.linear_model import LinearRegression
                default_model = LinearRegression()
                joblib.dump(default_model, model_path)
                print(f"Created default model at {model_path}")
                
            if not os.path.exists(feature_map_path):
                class DefaultFeatureMap:
                    def generate_features(self, X):
                        return X
                joblib.dump(DefaultFeatureMap(), feature_map_path)
                print(f"Created default feature map at {feature_map_path}")
                
            if not os.path.exists(kernel_path):
                class DefaultKernel:
                    def evaluate(self, X):
                        return np.ones(len(X))
                joblib.dump(DefaultKernel(), kernel_path)
                print(f"Created default kernel at {kernel_path}")
        
        # Load components
        self._load_components(model_path, scaler_path, feature_map_path, kernel_path)
        
    def _load_components(self, model_path, scaler_path, feature_map_path, kernel_path):
        """Load all model components"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_map = joblib.load(feature_map_path)
        self.kernel = joblib.load(kernel_path)
        
    def _create_quantum_circuit(self):
        """Create the quantum circuit"""
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev)
        def circuit(inputs, weights):
            # Encode input features
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply variational layers
            for layer in range(self.n_layers):
                # Rotation gates
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
    
    def predict(self, X):
        """Make predictions using the quantum circuit"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        for x in X:
            x_padded = np.pad(x, (0, max(0, self.n_qubits - len(x))))[:self.n_qubits]
            output = self.circuit(x_padded, self.weights)
            predictions.append(np.mean(output))
        return np.array(predictions)
    
    def save(self, model_dir='models/quantum'):
        """Save model parameters"""
        os.makedirs(model_dir, exist_ok=True)
        
        params = {
            'n_features': self.n_features,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'weights': self.weights
        }
        
        params_path = os.path.join(model_dir, 'quantum_params.pkl')
        joblib.dump(params, params_path)
        print(f"Model parameters saved to {params_path}")
    
    @classmethod
    def load(cls, model_dir='models/quantum'):
        """Load a saved model"""
        params_path = os.path.join(model_dir, 'quantum_params.pkl')
        params = joblib.load(params_path)
        
        model = cls(
            n_features=params['n_features'],
            n_qubits=params['n_qubits'],
            n_layers=params['n_layers']
        )
        model.weights = params['weights']
        return model

def create_and_save_model(n_features, n_qubits=None, n_layers=2):
    """Create and save a new quantum model"""
    model_dir = 'models/quantum'
    os.makedirs(model_dir, exist_ok=True)
    
    model = QuantumModel(n_features=n_features, n_qubits=n_qubits, n_layers=n_layers)
    model.save()
    return model

if __name__ == "__main__":
    # Model parameters
    n_features = 5  # Number of input features
    n_qubits = 4    # Number of qubits
    n_layers = 2    # Number of variational layers
    
    print("Creating quantum model...")
    model = create_and_save_model(n_features, n_qubits, n_layers)
    
    # Test predictions
    test_input = np.random.rand(3, n_features)
    predictions = model.predict(test_input)
    print("\nTest predictions:", predictions)
    
    # Verify saved model
    print("\nVerifying saved model...")
    loaded_model = QuantumModel.load()
    loaded_predictions = loaded_model.predict(test_input)
    print("Loaded model predictions:", loaded_predictions)
    
    # Verify predictions match
    np.testing.assert_array_almost_equal(predictions, loaded_predictions)
    print("\nVerification successful! Original and loaded models produce identical predictions.")