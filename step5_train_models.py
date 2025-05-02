import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


def train_models(
    X_train, y_train, classical_output_dir, quantum_output_dir, qml_model=None
):
    """
    STEP 5: Train classical and quantum models.

    Args:
        X_train: Training features
        y_train: Training targets
        classical_output_dir: Directory to save classical model
        quantum_output_dir: Directory to save quantum model
        qml_model: Precompiled PennyLane quantum model (optional)

    Returns:
        models: Dictionary of trained models
    """
    models = {}

    # --- Classical Model: Linear Regression ---
    print("\nTraining Classical Model...")
    classical_model = LinearRegression()
    classical_model.fit(X_train, y_train)
    models['Classical Model'] = classical_model

    os.makedirs(classical_output_dir, exist_ok=True)
    model_path = os.path.join(classical_output_dir, 'classical_model.pkl')
    joblib.dump(classical_model, model_path)
    print(f"Saved classical model to {model_path}")

    # --- Quantum Model: PennyLane QNode or KerasLayer ---
    if qml_model:
        print("\nTraining Quantum Model...")
        # Assume qml_model is a compiled model
        y_train_np = np.array(y_train).reshape(-1, 1)

        qml_model.fit(
            X_train,
            y_train_np,
            epochs=20,
            batch_size=5,
            verbose=1
        )
        models['Quantum Model'] = qml_model

        os.makedirs(quantum_output_dir, exist_ok=True)
        model_path = os.path.join(quantum_output_dir, 'quantum_model.keras')
        qml_model.save(model_path)
        print(f"Saved quantum model to {model_path}")
    else:
        print("Quantum model not provided. Skipping quantum training.")

    return models
