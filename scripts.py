import numpy as np
import pandas as pd
import time
import psutil
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
from typing import Dict, List, Any

@dataclass
class PerformanceMetrics:
    mse: float
    rmse: float
    mae: float
    r2_score: float
    training_time: float
    inference_time: float
    
@dataclass
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    storage_size: float

class MLComparison:
    def __init__(self):
        self.metrics = {
            'classical': {},
            'quantum': {}
        }
        
    def measure_performance(self, model_type: str, metrics: PerformanceMetrics):
        """Record performance metrics for each approach"""
        self.metrics[model_type]['performance'] = {
            'MSE': metrics.mse,
            'RMSE': metrics.rmse,
            'MAE': metrics.mae,
            'R2': metrics.r2_score,
            'Training Time': metrics.training_time,
            'Inference Time': metrics.inference_time
        }
        
    def measure_resources(self, model_type: str):
        """Measure computational resources"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        # Add GPU measurement if available
        
        self.metrics[model_type]['resources'] = {
            'CPU Usage (%)': cpu_percent,
            'Memory Usage (%)': memory,
            'Storage (MB)': self._get_model_size(model_type)
        }
        
    def analyze_scalability(self, model_type: str, dataset_sizes: List[int]):
        """Analyze scalability with different dataset sizes"""
        scaling_metrics = []
        for size in dataset_sizes:
            # Simulate training with different sizes
            training_time = self._measure_training_time(model_type, size)
            scaling_metrics.append({
                'Size': size,
                'Time': training_time
            })
        self.metrics[model_type]['scalability'] = scaling_metrics
        
    def calculate_costs(self, model_type: str):
        """Calculate implementation costs"""
        # Example cost calculation
        computing_cost = self._estimate_computing_cost(model_type)
        development_cost = self._estimate_development_cost(model_type)
        
        self.metrics[model_type]['costs'] = {
            'Computing Cost': computing_cost,
            'Development Cost': development_cost,
            'Total Cost': computing_cost + development_cost
        }
    def _plot_performance_comparison(self, output_path: str):
        """Create performance comparison plots"""
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        classical_vals = [self.metrics['classical']['performance'][m] for m in metrics]
        quantum_vals = [self.metrics['quantum']['performance'][m] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        plt.bar(x - 0.2, classical_vals, 0.4, label='Classical')
        plt.bar(x + 0.2, quantum_vals, 0.4, label='Quantum')
        plt.xticks(x, metrics)
        plt.title('Performance Metrics Comparison')
        plt.legend()
        plt.savefig(f'{output_path}/performance_comparison.png')
       # Add these methods to the MLComparison class:

    def _plot_resource_usage(self, output_path: str):
        """Create resource usage comparison plots"""
        if 'resources' not in self.metrics['classical'] or 'resources' not in self.metrics['quantum']:
            print("No resource metrics available for plotting")
            return
            
        metrics = ['CPU Usage (%)', 'Memory Usage (%)', 'Storage (MB)']
        classical_vals = [self.metrics['classical']['resources'][m] for m in metrics]
        quantum_vals = [self.metrics['quantum']['resources'][m] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        plt.bar(x - 0.2, classical_vals, 0.4, label='Classical')
        plt.bar(x + 0.2, quantum_vals, 0.4, label='Quantum')
        plt.xticks(x, metrics, rotation=45)
        plt.title('Resource Usage Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/resource_usage.png')
        plt.close()
    
    def generate_report(self, output_path: str):
    # Create lists of metrics with matching lengths
        metrics = []
        classical_values = []
        quantum_values = []
    
        # Add performance metrics
        if 'performance' in self.metrics['classical']:
            for metric, value in self.metrics['classical']['performance'].items():
                metrics.append(f"Performance_{metric}")
                classical_values.append(value)
                quantum_values.append(self.metrics['quantum'].get('performance', {}).get(metric, 0))
        
    # Add resource metrics
        if 'resources' in self.metrics['classical']:
                for metric, value in self.metrics['classical']['resources'].items():
                    metrics.append(f"Resource_{metric}")
                    classical_values.append(value)
                    quantum_values.append(self.metrics['quantum'].get('resources', {}).get(metric, 0))
            
                # Create DataFrame with aligned data
                report = pd.DataFrame({
                    'Metric': metrics,
                    'Classical': classical_values,
                    'Quantum': quantum_values
                })
                
                # Create output directory if it doesn't exist
                if output_path:
                    os.makedirs(output_path, exist_ok=True)
                    csv_path = f'{output_path}/comparison_report.csv'
                else:
                    csv_path = 'comparison_report.csv'
                
                # Save to CSV
                report.to_csv(csv_path, index=False)
                print(f"Report saved to: {csv_path}")
                
                # Generate visualizations if we have data
                if metrics:
                    self._plot_performance_comparison(output_path or '.')
                    self._plot_resource_usage(output_path or '.')
                    self._plot_scalability_analysis(output_path or '.')  
    def _plot_scalability_analysis(self, output_path: str):
        if 'scalability' not in self.metrics['classical'] or 'scalability' not in self.metrics['quantum']:
            print("No scalability metrics available for plotting")
            return
            
        classical_data = pd.DataFrame(self.metrics['classical']['scalability'])
        quantum_data = pd.DataFrame(self.metrics['quantum']['scalability'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(classical_data['Size'], classical_data['Time'], 
                'o-', label='Classical', color='blue')
        plt.plot(quantum_data['Size'], quantum_data['Time'], 
                'o-', label='Quantum', color='orange')
        plt.xlabel('Dataset Size')
        plt.ylabel('Training Time (s)')
        plt.title('Scalability Analysis')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_path}/scalability_analysis.png')
        plt.close() 
    def _measure_training_time(self, model_type: str, size: int) -> float:
        """Measure training time for different dataset sizes"""
        start_time = time.time()
        # Simulate training here
        end_time = time.time()
        return end_time - start_time
        
    def _get_model_size(self, model_type: str) -> float:
        """Get model size in MB"""
        # Implementation specific to your models
        return 0.0
        
    def _estimate_computing_cost(self, model_type: str) -> float:
        """Estimate computing costs"""
        # Implementation specific to your infrastructure
        return 0.0
        
    def _estimate_development_cost(self, model_type: str) -> float:
        """Estimate development costs"""
        # Implementation specific to your team
        return 0.0

# At the bottom of the file, update the main section:
if __name__ == "__main__":
    comparison = MLComparison()
    
    # Measure Classical ML
    classical_perf = PerformanceMetrics(
        mse=116740298,
        rmse=10804,
        mae=8571,
        r2_score=0.054,
        training_time=120.5,
        inference_time=0.5
    )
    comparison.measure_performance('classical', classical_perf)
    comparison.measure_resources('classical')
    
    # Measure Quantum ML (example metrics)
    quantum_perf = PerformanceMetrics(
        mse=98650298,
        rmse=9932,
        mae=7891,
        r2_score=0.068,
        training_time=180.3,
        inference_time=0.8
    )
    comparison.measure_performance('quantum', quantum_perf)
    comparison.measure_resources('quantum')
    
    # Generate report
    comparison.generate_report('rapport')
    
    print("Comparison analysis completed. Check 'rapport' directory for reports.")
    comparison = MLComparison()
    
    # Measure Classical ML
    classical_perf = PerformanceMetrics(
        mse=116740298,
        rmse=10804,
        mae=8571,
        r2_score=0.054,
        training_time=0.0,
        inference_time=0.0
    )
    comparison.measure_performance('classical', classical_perf)
    comparison.measure_resources('classical')
    
    # Measure Quantum ML
    # Add your quantum metrics here
    
    # Generate report
    comparison.generate_report('')
    
    print("Comparison analysis completed. Check 'results' directory for reports.")