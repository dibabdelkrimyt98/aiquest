import numpy as np
import pandas as pd
from step_prediction import QuantumPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_quantum_predictions(test_data_path='data/ai_features.csv', n_predictions=10):
    """
    Get quantum predictions for agricultural yields
    
    Args:
        test_data_path: Path to test data CSV
        n_predictions: Number of predictions to make
    """
    try:
        # Load test data
        test_data = pd.read_csv(test_data_path)
        
        # Initialize quantum predictor
        predictor = QuantumPredictor()
        
        # Select random samples for prediction
        sample_indices = np.random.choice(len(test_data), n_predictions, replace=False)
        test_samples = test_data.iloc[sample_indices]
        
        # Make predictions
        predictions = []
        confidences = []
        actuals = test_samples['Yield'].values if 'Yield' in test_samples.columns else None
        
        print("\nMaking quantum predictions...")
        for _, row in test_samples.iterrows():
            # Prepare input features
            input_features = {
                'temperature': row['temperature'],
                'rainfall': row['rainfall'],
                'soil_moisture': row['soil_moisture'],
                'humidity': row['humidity'],
                'solar_radiation': row['solar_radiation']
            }
            
            # Get prediction and confidence
            result = predictor.predict(input_features)
            confidence = predictor.get_prediction_confidence(input_features)
            
            predictions.append(result)
            confidences.append(confidence)
        
        # Calculate metrics if actuals are available
        if actuals is not None:
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            print("\nPrediction Metrics:")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R² Score: {r2:.4f}")
        
        # Display predictions
        print("\nPrediction Results:")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            print(f"\nPrediction {i+1}:")
            print(f"Predicted Yield: {pred:.2f}")
            print(f"Confidence: {conf:.2%}")
            if actuals is not None:
                print(f"Actual Yield: {actuals[i]:.2f}")
                print(f"Error: {abs(actuals[i] - pred):.2f}")
        
        return predictions, confidences, actuals
        
    except Exception as e:
        print(f"Error in quantum prediction: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # Run predictions
    predictions, confidences, actuals = get_quantum_predictions()