import os
import argparse

# Import the steps we have implemented
from step1_load_data import load_data
from step2_select_features import select_features
from step3_quantum_encoding import create_quantum_encoding
from step4_quantum_kernel import create_quantum_kernel
from step5_train_models import train_models
from step6_evaluate_models import evaluate_models
from step7_save_results import save_results


def build_quantum_kernel(n_features):
    print("\nBuilding Quantum Kernel...")
    
    # Create quantum feature map
    feature_map = create_quantum_encoding(n_features=n_features)
    
    # Create and return quantum kernel
    return create_quantum_kernel(
        encoding_function=feature_map,
        X_train=None  # Will be set during training
    )


def main():
    """Main function to run the quantum agriculture forecasting pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Quantum Agriculture Forecasting'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/ai_features.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=5,
        help='Number of top features to select'
    )
    parser.add_argument(
        '--scaling',
        type=str,
        choices=['minmax', 'standard'],
        default='minmax',
        help='Scaling method'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        choices=['zz', 'pauli'],
        default='zz',
        help='Quantum feature map type'
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=2,
        help='Number of repetitions in quantum circuit'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    dirs = [
        'data', 'results', 'plots',
        'models/classical', 'models/quantum'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load and preprocess data
    print("\nSTEP 1: Load and preprocess data")
    df = load_data(args.input)
    
    # Select features and scale data
    print("\nSTEP 2: Feature selection and scaling")
    X_train, X_test, y_train, y_test, selected_features = select_features(
        df, args.features, args.scaling
    )
    
    # Build quantum kernel
    print("\nSTEP 3: Build quantum kernel")
    kernel = build_quantum_kernel(n_features=len(selected_features))
    
    # Train models
    print("\nSTEP 4: Train models")
    models = train_models(
        X_train,
        y_train,
        'models/classical',
        'models/quantum',
        kernel
    )
    
    # Evaluate models
    print("\nSTEP 5: Evaluate models")
    metrics, predictions, times = evaluate_models(
        models,
        X_test,
        y_test
    )
    
    # Save results
    print("\nSTEP 6: Save results")
    save_results(
        metrics,
        predictions,
        times,
        y_test,
        selected_features,
        'results'
    )
    
    print("\nQuantum agriculture forecasting pipeline completed successfully!")


if __name__ == "__main__":
    main()
