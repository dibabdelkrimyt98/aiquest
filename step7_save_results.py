import os
import pandas as pd
import matplotlib.pyplot as plt


def save_results(metrics, predictions, times, y_test, output_dir, plots_dir):
    """
    Save model results, metrics, and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save predictions
    results_df = pd.DataFrame({'True_Values': y_test})
    for model_name, preds in predictions.items():
        results_df[f'{model_name}_Predictions'] = preds
    results_df.to_csv(
        os.path.join(output_dir, 'prediction_results.csv'),
        index=False
    )
    print(f"Saved prediction results to {output_dir}/prediction_results.csv")

    # Save metrics
    metric_rows = []
    for metric in ['MSE', 'RMSE', 'MAE', 'R²']:
        row = {'Metric': metric}
        for model_name in metrics:
            row[model_name] = metrics[model_name][metric]
        if len(metrics) > 1:
            q = [k for k in metrics if 'Quantum' in k][0]
            c = [k for k in metrics if 'Classical' in k][0]
            q_val = metrics[q][metric]
            c_val = metrics[c][metric]
            if metric == 'R²':
                row['Improvement (%)'] = (q_val - c_val) / abs(c_val) * 100
            else:
                row['Improvement (%)'] = (c_val - q_val) / c_val * 100
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(output_dir, 'qml_metrics.csv'), index=False)
    print(f"Saved metrics to {output_dir}/qml_metrics.csv")

    # Save times
    pd.DataFrame({
        'Model': list(times.keys()),
        'Prediction_Time_Seconds': list(times.values())
    }).to_csv(os.path.join(output_dir, 'prediction_times.csv'), index=False)
    print(f"Saved times to {output_dir}/prediction_times.csv")

    # Accuracy plot for first model
    first_model = list(predictions.keys())[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions[first_model], label=first_model, alpha=0.6)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        '--',
        color='gray'
    )
    plt.xlabel('True Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, 'qml_accuracy.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved accuracy plot to {plot_path}")
