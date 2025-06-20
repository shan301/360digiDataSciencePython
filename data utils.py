# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:11:39 2025

@author: Shantanu
"""

"""data_utils.py
This script provides utility functions for data processing in data science and automation tasks. It includes functions for loading datasets, cleaning data, encoding variables, scaling features, and exporting results, designed to work with datasets like `sales.csv` and `hr_data.csv` in the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn
- Datasets: `sales.csv`, `hr_data.csv` in the `data/` directory
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Ensure output directory exists
output_dir = '../data/output'
os.makedirs(output_dir, exist_ok=True)

def load_dataset(file_name):
    """Load a CSV file from the data directory.
    
    Args:
        file_name (str): Name of the CSV file (e.g., 'sales.csv').
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = os.path.join('../data', file_name)
    return pd.read_csv(file_path)

def save_dataset(df, file_name):
    """Save a DataFrame to a CSV file in the output directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_name (str): Name of the output CSV file.
    """
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f'Saved dataset to {file_path}')

def handle_missing_values(df, strategy='mean'):
    """Handle missing values in numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy for filling missing values ('mean', 'median', 'drop').
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if strategy == 'drop':
        df = df.dropna(subset=numerical_cols)
    elif strategy in ['mean', 'median']:
        fill_values = df[numerical_cols].agg(strategy)
        df[numerical_cols] = df[numerical_cols].fillna(fill_values)
    return df

def encode_categorical(df, columns):
    """Encode categorical columns using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of categorical column names.
        
    Returns:
        pd.DataFrame: DataFrame with encoded columns.
    """
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df

def one_hot_encode(df, columns):
    """Perform one-hot encoding on categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of categorical column names.
        
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)

def scale_features(df, columns):
    """Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of numerical column names to scale.
        
    Returns:
        pd.DataFrame: DataFrame with scaled columns.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def split_dataset(df, target_column, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def compute_summary_stats(df):
    """Compute summary statistics for numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Summary statistics.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return df[numerical_cols].describe()

def remove_outliers(df, column, threshold=3):
    """Remove outliers from a numerical column using Z-score.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the column to check for outliers.
        threshold (float): Z-score threshold.
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df = df.copy()
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def merge_datasets(df1, df2, key):
    """Merge two DataFrames on a common key.
    
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        key (str): Column name to merge on.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    return pd.merge(df1, df2, on=key, how='inner')

def group_and_aggregate(df, group_col, agg_dict):
    """Group by a column and apply aggregation functions.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Column to group by.
        agg_dict (dict): Dictionary of columns and aggregation functions.
        
    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    return df.groupby(group_col).agg(agg_dict).reset_index()

def convert_dtypes(df, column_types):
    """Convert columns to specified data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_types (dict): Dictionary of column names and target types.
        
    Returns:
        pd.DataFrame: DataFrame with converted types.
    """
    df = df.copy()
    for col, dtype in column_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df

def check_duplicates(df):
    """Check for duplicate rows in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        int: Number of duplicate rows.
    """
    return df.duplicated().sum()

def drop_columns(df, columns):
    """Drop specified columns from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to drop.
        
    Returns:
        pd.DataFrame: DataFrame with dropped columns.
    """
    return df.drop(columns=[col for col in columns if col in df.columns], axis=1)

# Example Usage
if __name__ == "__main__":
    # Load datasets
    sales = load_dataset('sales.csv')
    hr = load_dataset('hr_data.csv')
    
    # Example: Handle missing values
    sales_clean = handle_missing_values(sales, strategy='median')
    print('Sales Data After Handling Missing Values:')
    print(sales_clean.head())
    
    # Example: Encode categorical column
    hr_encoded = encode_categorical(hr, ['department'])
    print('\nHR Data After Encoding Department:')
    print(hr_encoded.head())
    
    # Example: Scale numerical columns
    sales_scaled = scale_features(sales_clean, ['sales_amount', 'marketing_spend'])
    print('\nSales Data After Scaling:')
    print(sales_scaled.head())
    
    # Example: Save processed data
    save_dataset(sales_scaled, 'processed_sales.csv')
    
    # Example: Compute summary statistics
    summary = compute_summary_stats(sales)
    print('\nSales Summary Statistics:')
    print(summary)