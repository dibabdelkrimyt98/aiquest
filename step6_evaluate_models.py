import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_models(models, X_test, y_test):
   
    metrics = {}
    predictions = {}
    times = {}
    print("\nEvaluating model performance:")

    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    for name, model in models.items():
        print(f"\nPredicting with {name}...")
        start_time = time.time()

        try:
            # Ensure compatibility with PennyLane interfaces
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                y_pred = model(X_test)

            pred_time = time.time() - start_time
            times[name] = pred_time
            predictions[name] = y_pred

            model_metrics = compute_metrics(y_test, y_pred)
            metrics[name] = model_metrics

            print(f"{name} Performance:")
            for metric, value in model_metrics.items():
                print(f"{metric}: {value:.4f}")
            print(f"Prediction time: {pred_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            metrics[name] = None
            predictions[name] = None
            times[name] = None

    return metrics, predictions, times