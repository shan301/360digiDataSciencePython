# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:22:57 2025

@author: Shantanu
"""

"""ml_utils.py
This script provides utility functions for machine learning tasks in data science and automation. It includes functions for data preprocessing, model training, evaluation, and hyperparameter tuning, designed to work with datasets like `sales.csv` and `hr_data.csv` in the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn, xgboost, joblib
- Datasets: `sales.csv`, `hr_data.csv` in the `data/` directory
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
import joblib
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

def save_model(model, file_name):
    """Save a trained model to the output directory.
    
    Args:
        model: Trained machine learning model.
        file_name (str): Name of the output file (e.g., 'model.pkl').
    """
    file_path = os.path.join(output_dir, file_name)
    joblib.dump(model, file_path)
    print(f'Saved model to {file_path}')

def preprocess_data(df, target_column, categorical_cols=None, numerical_cols=None):
    """Preprocess data by encoding categorical columns and scaling numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        categorical_cols (list, optional): List of categorical column names.
        numerical_cols (list, optional): List of numerical column names.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = df.copy()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df.drop(columns=[target_column]), df[target_column]

def split_dataset(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest Classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of trees.
        random_state (int): Random seed.
        
    Returns:
        RandomForestClassifier: Trained model.
    """
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    return rf_clf

def train_random_forest_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of trees.
        random_state (int): Random seed.
        
    Returns:
        RandomForestRegressor: Trained model.
    """
    rf_reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_reg.fit(X_train, y_train)
    return rf_reg

def train_xgboost_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """Train an XGBoost Classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
        
    Returns:
        XGBClassifier: Trained model.
    """
    xgb_clf = XGBClassifier(n_estimators=n_estimators, random_state=random_state, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def train_xgboost_regressor(X_train, y_train, n_estimators=100, random_state=42):
    """Train an XGBoost Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed.
        
    Returns:
        XGBRegressor: Trained model.
    """
    xgb_reg = XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    xgb_reg.fit(X_train, y_train)
    return xgb_reg

def evaluate_classifier(model, X_test, y_test):
    """Evaluate a classifier model with accuracy and classification report.
    
    Args:
        model: Trained classifier model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        dict: Evaluation metrics (accuracy and classification report).
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {'accuracy': accuracy, 'report': report}

def evaluate_regressor(model, X_test, y_test):
    """Evaluate a regressor model with MSE and R² score.
    
    Args:
        model: Trained regressor model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        dict: Evaluation metrics (MSE and R² score).
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'mse': mse, 'r2': r2}

def get_feature_importance(model, feature_names):
    """Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        
    Returns:
        pd.Series: Feature importance scores.
    """
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

def perform_grid_search(model, param_grid, X_train, y_train, cv=5):
    """Perform grid search for hyperparameter tuning.
    
    Args:
        model: Machine learning model.
        param_grid (dict): Parameter grid for grid search.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv (int): Number of cross-validation folds.
        
    Returns:
        GridSearchCV: Fitted grid search object.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """Perform cross-validation on a model.
    
    Args:
        model: Machine learning model.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (e.g., 'accuracy', 'r2').
        
    Returns:
        np.ndarray: Cross-validation scores.
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)

def create_binary_target(df, column, threshold=None):
    """Create a binary target column based on a threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to binarize.
        threshold (float, optional): Threshold value (default: median).
        
    Returns:
        pd.DataFrame: DataFrame with new binary target column.
    """
    df = df.copy()
    if threshold is None:
        threshold = df[column].median()
    df[f'{column}_binary'] = (df[column] > threshold).astype(int)
    return df

def predict_new_data(model, X_new):
    """Make predictions on new data using a trained model.
    
    Args:
        model: Trained machine learning model.
        X_new (pd.DataFrame): New data for prediction.
        
    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X_new)

# Example Usage
if __name__ == "__main__":
    # Load datasets
    sales = load_dataset('sales.csv')
    hr = load_dataset('hr_data.csv')
    
    # Example: Preprocess sales data
    X_sales, y_sales = preprocess_data(
        sales, 
        target_column='sales_amount', 
        numerical_cols=['marketing_spend', 'store_size']
    )
    print('Preprocessed Sales Features Head:')
    print(X_sales.head())
    
    # Example: Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X_sales, y_sales)
    
    # Example: Train Random Forest Regressor
    rf_reg = train_random_forest_regressor(X_train, y_train)
    print('\nTrained Random Forest Regressor')
    
    # Example: Evaluate regressor
    metrics = evaluate_regressor(rf_reg, X_test, y_test)
    print('Regressor Metrics:', metrics)
    
    # Example: Save model
    save_model(rf_reg, 'rf_regressor.pkl')
    
    # Example: Feature importance
    importance = get_feature_importance(rf_reg, X_sales.columns)
    print('\nFeature Importance:')
    print(importance)