# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:47:54 2025

@author: Shantanu
"""
"""14. Machine Learning Basics
This script introduces fundamental machine learning concepts using scikit-learn, focusing on regression and classification tasks. It covers data preprocessing, model training, evaluation, hyperparameter tuning, and model interpretation, using sample datasets from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Datasets: `cars.csv`, `sales.csv`, `hr_data.csv` from the `data/` directory
"""

# 14.1. Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import joblib

# Load sample datasets
cars_df = pd.read_csv('../data/cars.csv')
sales_df = pd.read_csv('../data/sales.csv')
hr_df = pd.read_csv('../data/hr_data.csv')
print('Cars Dataset Head:')
print(cars_df.head())
print('\nSales Dataset Head:')
print(sales_df.head())
print('\nHR Dataset Head:')
print(hr_df.head())

"""14.2. Introduction to Machine Learning
Machine learning (ML) enables systems to learn from data. Key types:
- Supervised Learning: Labeled data (regression, classification).
- Unsupervised Learning: Unlabeled data (clustering).
- Reinforcement Learning: Reward-based learning.
This script focuses on supervised learning with scikit-learn."""

"""14.3. Data Preprocessing
Preprocessing ensures data is suitable for ML models by handling missing values, encoding categorical variables, and scaling features."""
# Handle missing values in cars dataset
cars_df.fillna(cars_df.mean(numeric_only=True), inplace=True)

# Encode categorical variables (e.g., 'fuel_type' in cars dataset)
cars_df = pd.get_dummies(cars_df, columns=['fuel_type'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['mileage', 'engine_size']
cars_df[numerical_cols] = scaler.fit_transform(cars_df[numerical_cols])
print('Preprocessed Cars Dataset:')
print(cars_df.head())

"""14.4. Train-Test Split
Splitting data into training and testing sets evaluates model performance on unseen data."""
# Features and target for regression (predicting car price)
X = cars_df[['mileage', 'engine_size', 'fuel_type_Diesel']]
y = cars_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

"""14.5. Linear Regression
Linear regression models continuous outcomes (e.g., car price) using a linear relationship."""
# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression MSE: {mse:.2f}')
print(f'Linear Regression R^2 Score: {r2:.2f}')

# Plot predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

"""14.6. Logistic Regression
Logistic regression predicts binary outcomes (e.g., high/low sales)."""
# Prepare sales dataset (binary classification: high/low sales)
sales_df['is_high_sale'] = (sales_df['sales_amount'] > sales_df['sales_amount'].median()).astype(int)
X = sales_df[['marketing_spend', 'store_size']]
y = sales_df['is_high_sale']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Logistic Regression: Confusion Matrix')
plt.show()

"""14.7. Decision Tree Classifier
Decision trees split data based on feature thresholds for classification."""
# Train decision tree classifier on sales data
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')
print('Decision Tree Classification Report:')
print(classification_report(y_test, y_pred_dt))

"""14.8. Model Evaluation Metrics
- Regression: MSE, RMSE, R² score.
- Classification: Accuracy, precision, recall, F1-score, confusion matrix."""

"""14.9. Cross-Validation
Cross-validation assesses model stability across data subsets."""
# Perform 5-fold cross-validation on linear regression
cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
print('Linear Regression Cross-Validation R^2 Scores:', cv_scores)
print(f'Average R^2 Score: {cv_scores.mean():.2f}')

"""14.10. Hyperparameter Tuning
Grid search optimizes model parameters."""
# Grid search for logistic regression
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best Parameters for Logistic Regression:', grid_search.best_params_)
print(f'Best Score: {grid_search.best_score_:.2f}')

"""14.11. Feature Importance
Feature importance reveals which variables drive predictions."""
# Feature importance for linear regression coefficients
feature_importance = X.columns
feature_importance = pd.Series([lr_model.coef_, index=X.columns])
print('Linear Regression Feature Importance:\n', feature_importance)

"""14.12. Saving and Loading Models
Models are saved for reuse using joblib."""
# Save model
joblib.dump(lr_model, '../scripts/ml_utils_lr.pkl')

# Load model
loaded_model = joblib.load('../scripts/ml_utils_lr.pkl')
print('Loaded Linear Regression Coefficients:', loaded_model.coef_)

"""14. Machine Learning Exercises"""

"""Exercise 1: Load and Inspect Dataset
Load `hr_data.csv` and print its first 5 rows and column names."""
def exercise_1():
    hr_df = pd.read_csv('../data/hr_data.csv')
    print('First 5 rows:')
    print(hr_df.head())
    print('\nColumns:', hr_df.columns.tolist())

exercise_1()

"""Exercise 2: Handle Missing Values
Fill missing values in `hr_data.csv` with the median of numerical columns."""
def exercise_2():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df.fillna(hr_df.median(numeric_only=True), inplace=True)
    print('Missing values after filling:', hr_df.isnull().sum())

exercise_2()

"""Exercise 3: Encode Categorical Variables
Encode the `department` column in `hr_data.csv` using one-hot encoding."""
def exercise_3():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df = pd.get_dummies(hr_df, columns=['department'], drop_first=True)
    print('Encoded dataset head:')
    print(hr_df.head())

exercise_3()

"""Exercise 4: Train-Test Split
Split `hr_data.csv` into training (80%) and testing (20%) sets to predict `salary`."""
def exercise_4():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set shape:', X_train.shape)

exercise_4()

"""Exercise 5: Linear Regression Model
Train a linear regression model on `hr_data.csv` to predict `salary` and compute MSE."""
def exercise_5():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse:.2f}')

exercise_5()

"""Exercise 6: Logistic Regression Model
Create a binary target in `hr_data.csv` (e.g., `high_salary` if above median) and train a logistic regression model."""
def exercise_6():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
    X = hr_df.drop(['salary', 'high_salary'], axis=1)
    y = hr_df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

exercise_6()

"""Exercise 7: Confusion Matrix
Compute and plot the confusion matrix for the logistic regression model above."""
def exercise_7():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
    X = hr_df.drop(['salary', 'high_salary'], axis=1)
    y = hr_df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

exercise_7()

"""Exercise 8: Feature Scaling
Scale numerical features in `hr_data.csv` before training a model."""
def exercise_8():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print('Scaled features head:')
    print(X.head())

exercise_8()

"""Exercise 9: Grid Search
Perform grid search to tune `C` for the logistic regression model on `hr_data.csv`."""
def exercise_9():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
    X = hr_df.drop(['salary', 'high_salary'], axis=1)
    y = hr_df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print('Best Parameters:', grid_search.best_params_)

exercise_9()

"""Exercise 10: Save Model
Save the trained linear regression model to `ml_utils_hr_model.pkl`."""
def exercise_10():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, '../scripts/ml_utils_hr_model.pkl')
    print('Model saved successfully')

exercise_10()

"""Exercise 11: R² Score
Compute the R² score for the linear regression model on `hr_data.csv`."""
def exercise_11():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2:.2f}')

exercise_11()

"""Exercise 12: Feature Importance
Print the coefficients of the linear regression model to assess feature importance."""
def exercise_12():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    feature_importance = pd.Series(lr_model.coef_, index=X.columns)
    print('Feature Importance:\n', feature_importance)

exercise_12()

"""Exercise 13: Predict New Data
Use the logistic regression model to predict `high_salary` for a new data point."""
def exercise_13():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
    X = hr_df.drop(['salary', 'high_salary'], axis=1)
    y = hr_df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    new_data = X_test[0:1]
    prediction = log_model.predict(new_data)
    print('Prediction for new data:', prediction)

exercise_13()

"""Exercise 14: Cross-Validation
Perform 5-fold cross-validation on the linear regression model for `hr_data.csv`."""
def exercise_14():
    hr_df = pd.read_csv('../data/hr_data.csv')
    X = hr_df.drop('salary', axis=1)
    y = hr_df['salary']
    lr_model = LinearRegression()
    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
    print('Cross-Validation R^2 Scores:', cv_scores)
    print(f'Average R^2 Score: {cv_scores.mean():.2f}')

exercise_14()

"""Exercise 15: Decision Tree Model
Train a decision tree classifier on `hr_data.csv` to predict `high_salary` and compute accuracy."""
def exercise_15():
    hr_df = pd.read_csv('../data/hr_data.csv')
    hr_df['high_salary'] = (hr_df['salary'] > hr_df['salary'].median()).astype(int)
    X = hr_df.drop(['salary', 'high_salary'], axis=1)
    y = hr_df['high_salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Decision Tree Accuracy: {accuracy:.2f}')

exercise_15()

