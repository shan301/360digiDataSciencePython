# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:06:49 2025

@author: Shantanu
"""

"""16. Advanced Machine Learning
This script explores advanced machine learning concepts, focusing on ensemble methods such as Random Forest and XGBoost. It covers model training, hyperparameter tuning, feature importance, and model evaluation, using sample datasets from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- Datasets: `hr_data.csv`, `sales.csv` from the `data/` directory
"""

# 16.1. Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
import joblib

# Load sample datasets
hr_df = pd.read_csv('../data/hr_data.csv')
sales_df = pd.read_csv('../data/sales.csv')
print('HR Dataset Head:')
print(hr_df.head())
print('\nSales Dataset Head:')
print(sales_df.head())

"""16.2. Introduction to Advanced Machine Learning
Advanced machine learning involves sophisticated techniques like ensemble methods, which combine multiple models to improve performance. Key concepts:
- Ensemble Methods: Bagging (e.g., Random Forest) and Boosting (e.g., XGBoost).
- Hyperparameter Tuning: Optimizing model parameters for better performance.
- Feature Importance: Understanding which features drive predictions.
This script focuses on Random Forest and XGBoost for classification and regression tasks."""

"""16.3. Data Preprocessing
Prepare datasets by handling missing values, encoding categorical variables, and scaling features."""
# Handle missing values in hr dataset
hr_df.fillna(hr_df.median(numeric_only=True), inplace=True)

# Encode categorical variables (e.g., 'department' in hr dataset)
hr_df = pd.get_dummies(hr_df, columns=['department'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = hr_df.select_dtypes(include=['int64', 'float64']).columns.drop('salary', errors='ignore')
hr_df[numerical_cols] = scaler.fit_transform(hr_df[numerical_cols])
print('Preprocessed HR Dataset:')
print(hr_df.head())

"""16.4. Random Forest Classifier
Random Forest is a bagging ensemble method for classification, using multiple decision trees."""
# Prepare hr dataset for classification (predict high/low salary)
hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
X = hr_df.drop(['salary', 'high_salary'], axis=1)
y = hr_df['high_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Random Forest Classifier Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Random Forest Classifier: Confusion Matrix')
plt.show()

"""16.5. Random Forest Regressor
Random Forest can also be used for regression tasks, such as predicting continuous outcomes."""
# Prepare sales dataset for regression (predict sales amount)
X = sales_df[['marketing_spend', 'store_size']]
y = sales_df['sales_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Random Forest Regressor MSE: {mse:.2f}')
print(f'Random Forest Regressor R^2 Score: {r2:.2f}')

# Plot predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales Amount')
plt.ylabel('Predicted Sales Amount')
plt.title('Random Forest Regressor: Actual vs Predicted')
plt.show()

"""16.6. XGBoost Classifier
XGBoost is a boosting ensemble method that sequentially improves weak learners for classification."""
# Train XGBoost Classifier on hr dataset
xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Classifier Accuracy: {xgb_accuracy:.2f}')
print('XGBoost Classification Report:')
print(classification_report(y_test, y_pred))

"""16.7. XGBoost Regressor
XGBoost can also handle regression tasks with high accuracy."""
# Train XGBoost Regressor on sales dataset
xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_reg.predict(X_test)
xgb_mse = mean_squared_error(y_test, y_pred)
xgb_r2 = r2_score(y_test, y_pred)
print(f'XGBoost Regressor MSE: {xgb_mse:.2f}')
print(f'XGBoost Regressor R^2 Score: {xgb_r2:.2f}')

"""16.8. Feature Importance
Feature importance reveals which variables contribute most to predictions."""
# Feature importance for Random Forest Classifier
rf_feature_importance = pd.Series(rf_clf.feature_importances_, index=X.columns)
print('Random Forest Feature Importance:\n', rf_feature_importance.sort_values(ascending=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
rf_feature_importance.sort_values().plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.show()

"""16.9. Hyperparameter Tuning
Grid search optimizes ensemble model parameters."""
# Grid search for Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best Parameters for Random Forest:', grid_search.best_params_)
print(f'Best Score: {grid_search.best_score_:.2f}')

"""16.10. Cross-Validation
Cross-validation ensures robust model performance across data subsets."""
# Perform 5-fold cross-validation on XGBoost Classifier
cv_scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')
print('XGBoost Cross-Validation Accuracy Scores:', cv_scores)
print(f'Average Accuracy: {cv_scores.mean():.2f}')

"""16.11. Saving and Loading Models
Save trained models for reuse."""
# Save Random Forest model
joblib.dump(rf_clf, '../scripts/ml_utils_rf.pkl')

# Load model
loaded_rf = joblib.load('../scripts/ml_utils_rf.pkl')
print('Loaded Random Forest Feature Importances:', loaded_rf.feature_importances_)

"""16. Advanced Machine Learning Exercises"""

"""Exercise 1: Load and Inspect Dataset
Load `hr_data.csv` and print its first 5 rows and data types."""
def exercise_1():
    df = pd.read_csv('../data/hr_data.csv')
    print('First 5 rows:')
    print(df.head())
    print('\nData Types:')
    print(df.dtypes)

exercise_1()

"""Exercise 2: Handle Missing Values
Fill missing values in `hr_data.csv` with the mean of numerical columns."""
def exercise_2():
    df = pd.read_csv('../data/hr_data.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print('Missing values after filling:', df.isnull().sum())

exercise_2()

"""Exercise 3: Encode Categorical Variables
Encode the `department` column in `hr_data.csv` using one-hot encoding."""
def exercise_3():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    print('Encoded dataset head:')
    print(df.head())

exercise_3()

"""Exercise 4: Train-Test Split
Split `hr_data.csv` into training (80%) and testing (20%) sets for predicting `high_salary`."""
def exercise_4():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set shape:', X_train.shape)

exercise_4()

"""Exercise 5: Random Forest Classifier
Train a Random Forest Classifier on `hr_data.csv` to predict `high_salary` and compute accuracy."""
def exercise_5():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

exercise_5()

"""Exercise 6: Random Forest Regressor
Train a Random Forest Regressor on `sales.csv` to predict `sales_amount` and compute MSE."""
def exercise_6():
    df = pd.read_csv('../data/sales.csv')
    X = df[['marketing_spend', 'store_size']]
    y = df['sales_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse:.2f}')

exercise_6()

"""Exercise 7: XGBoost Classifier
Train an XGBoost Classifier on `hr_data.csv` to predict `high_salary` and print classification report."""
def exercise_7():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print('XGBoost Classification Report:')
    print(classification_report(y_test, y_pred))

exercise_7()

"""Exercise 8: Feature Scaling
Scale numerical features in `sales.csv` before training a model."""
def exercise_8():
    df = pd.read_csv('../data/sales.csv')
    X = df[['marketing_spend', 'store_size']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print('Scaled features head:')
    print(X_scaled_df.head())

exercise_8()

"""Exercise 9: Grid Search
Perform grid search to tune `n_estimators` and `max_depth` for Random Forest on `hr_data.csv`."""
def exercise_9():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print('Best Parameters:', grid_search.best_params_)
    print(f'Best Score: {grid_search.best_score_:.2f}')

exercise_9()

"""Exercise 10: Save Model
Save the trained XGBoost Regressor model to `ml_utils_xgb.pkl`."""
def exercise_10():
    df = pd.read_csv('../data/sales.csv')
    X = df[['marketing_spend', 'store_size']]
    y = df['sales_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(X_train, y_train)
    joblib.dump(xgb_reg, '../scripts/ml_utils_xgb.pkl')
    print('Model saved successfully')

exercise_10()

"""Exercise 11: Feature Importance
Plot feature importance for the XGBoost Classifier on `hr_data.csv`."""
def exercise_11():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X, y)
    feature_importance = pd.Series(xgb_clf.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().plot(kind='barh')
    plt.title('XGBoost Feature Importance')
    plt.show()

exercise_11()

"""Exercise 12: Cross-Validation
Perform 5-fold cross-validation on Random Forest Regressor for `sales.csv`."""
def exercise_12():
    df = pd.read_csv('../data/sales.csv')
    X = df[['marketing_spend', 'store_size']]
    y = df['sales_amount']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf_reg, X, y, cv=5, scoring='r2')
    print('Cross-Validation R^2 Scores:', cv_scores)
    print(f'Average R^2 Score: {cv_scores.mean():.2f}')

exercise_12()

"""Exercise 13: Predict New Data
Use the Random Forest Classifier to predict `high_salary` for a new data point from `hr_data.csv`."""
def exercise_13():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['high_salary'], axis=1)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    new_data = X_test.iloc[0:1]
    prediction = rf_clf.predict(new_data)
    print('Prediction for new data:', prediction)

exercise_13()

"""Exercise 14: Confusion Matrix
Plot the confusion matrix for the XGBoost Classifier on `hr_data.csv`."""
def exercise_14():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('XGBoost Classifier: Confusion Matrix')
    plt.show()

exercise_14()

"""Exercise 15: Compare Models
Compare Random Forest and XGBoost Classifier accuracies on `hr_data.csv`."""
def exercise_15():
    df = pd.read_csv('../data/hr_data.csv')
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    X = df.drop(['salary', 'high_salary'], axis=1)
    y = df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    rf_clf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    xgb_pred = xgb_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
    print(f'XGBoost Accuracy: {xgb_accuracy:.2f}')

exercise_15()