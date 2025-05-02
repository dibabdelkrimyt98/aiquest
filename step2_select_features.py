import pandas as pd
import numpy as np  # Required for array conversion
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
import os


def select_features(
    df: pd.DataFrame,
    n_features: int,
    scaling_method: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    target_col = df.columns[-1]
    print(f"Using '{target_col}' as target variable")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Feature selection
    print(f"Selecting top {n_features} features...")
    k = min(n_features, len(X.columns))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    
    # Get feature scores
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': scores
    })
    feature_scores = feature_scores.sort_values('F_Score', ascending=False)
    print("\nFeature importance scores:")
    for idx, row in enumerate(feature_scores.itertuples(), 1):
        print(f"{idx}. {row.Feature}: {row.F_Score:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    scores_array = feature_scores['F_Score'].to_numpy()
    features_list = feature_scores['Feature'].tolist()
    indices = np.arange(len(scores_array))
    
    plt.barh(indices, scores_array, align='center')
    plt.yticks(indices, features_list)
    plt.xlabel('F-Score')
    plt.title('Feature Importance')
    
    # Create plots directory if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save feature importance plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    print(f"Selected features: {', '.join(selected_features)}")
    
    # Keep only selected features
    X = X[selected_features]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Scale features
    if scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        print("Using MinMax scaling to range [-1, 1]")
    else:
        scaler = StandardScaler()
        print("Using Standard scaling")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot data distribution before and after scaling
    plt.subplots(1, 2, figsize=(12, 5))
    fig = plt.gcf()
    axes = fig.axes
    
    # Before scaling
    axes[0].boxplot(X_train)
    axes[0].set_title('Feature Distribution Before Scaling')
    axes[0].set_xticklabels(selected_features, rotation=45)
    
    # After scaling
    axes[1].boxplot(X_train_scaled)
    axes[1].set_title('Feature Distribution After Scaling')
    axes[1].set_xticklabels(selected_features, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scaling_comparison.png'))
    plt.close()
    
    return (
        X_train_scaled, 
        X_test_scaled,
        y_train.values,
        y_test.values,
        selected_features
    )
