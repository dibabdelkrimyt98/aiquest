import os
import argparse

# Import the steps we have implemented
from create_model import QuantumModel
from step1_load_data import load_data
from step2_select_features import select_features
from step3_quantum_encoding import create_quantum_encoding
from step4_quantum_kernel import create_quantum_kernel
from step5_train_models import train_models
from step6_evaluate_models import evaluate_models
from step7_save_results import save_results


def build_quantum_kernel(n_features, X_train):
    # STEP 3: Quantum encoding
    print("\nSTEP 3: Quantum feature encoding")
    feature_map = create_quantum_encoding(n_features=n_features)

    # STEP 4: Quantum kernel construction
    print("\nSTEP 4: Quantum kernel construction")
    return create_quantum_kernel(
        encoding_function=feature_map,
        X_train=X_train
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Quantum Agriculture Forecasting'
    )
    parser.add_argument('--input', type=str, default='data/ai_features.csv', help='Path to input CSV file')
    parser.add_argument('--features', type=int, default=5, help='Number of top features to select')
    parser.add_argument('--scaling', type=str, choices=['minmax', 'standard'], default='minmax', help='Scaling method')
    parser.add_argument('--kernel', type=str, choices=['zz', 'pauli'], default='zz', help='Quantum feature map type')
    parser.add_argument('--reps', type=int, default=2, help='Number of repetitions in quantum circuit')
    args = parser.parse_args()

    # Create necessary directories
    dirs = ['data', 'results', 'plots', 'models/classical', 'models/quantum']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # STEP 1: Load and preprocess data
    print("\nSTEP 1: Load and preprocess data")
    df = load_data(args.input)

    # STEP 2: Feature selection and scaling
    print("\nSTEP 2: Feature selection and scaling")
    X_train, X_test, y_train, y_test, selected_features = select_features(
        df, args.features, args.scaling
    )

    # STEP 3 & 4: Quantum feature map & kernel
    kernel = build_quantum_kernel(n_features=len(selected_features), X_train=X_train)

    # STEP 5: Train models
    print("\nSTEP 5: Train models")
    models = train_models(
        X_train,
        y_train,
        'models/classical',
        'models/quantum',
        kernel
    )

    # STEP 6: Evaluate models
    print("\nSTEP 6: Evaluate models")
    metrics, predictions, times = evaluate_models(
        models,
        X_test,
        y_test
    )

    # STEP 7: Save results
    print("\nSTEP 7: Save results")
    save_results(
        metrics,
        predictions,
        times,
        y_test,
        selected_features,
        'results'
    )

    print("\nQuantum agriculture forecasting pipeline completed successfully!")


# Run main function
if __name__ == "__main__":
    main()
