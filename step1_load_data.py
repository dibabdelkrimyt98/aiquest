import pandas as pd
# Removed unused import: import numpy as np


def load_data(input_path):

    print(f"Loading data from {input_path}...")
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        # Fill missing values with median for numeric columns
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(
        exclude=['float64', 'int64']
    ).columns.tolist()
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns: {', '.join(non_numeric_cols)}")
        # Drop non-numeric columns
        df = df.drop(columns=non_numeric_cols)
    
    # Show statistical summary
    print("\nData Summary:")
    print(df.describe().transpose()[['min', 'mean', 'max', 'std']])
    
    return df
