# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:58:10 2025

@author: Shantanu
"""

"""17. Deep Learning
This script introduces fundamental deep learning concepts using TensorFlow/Keras. It covers neural network architecture, training, evaluation, and hyperparameter tuning, using sample datasets from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, tensorflow, scikit-learn, matplotlib, seaborn
- Datasets: `hr_data.csv`, `sales.csv` from the `data/` directory
"""

# 17.1. Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Load sample datasets
hr_df = pd.read_csv('../data/hr_data.csv')
sales_df = pd.read_csv('../data/sales.csv')
print('HR Dataset Head:')
print(hr_df.head())
print('\nSales Dataset Head:')
print(sales_df.head())

"""17.2. Introduction to Deep Learning
Deep learning uses neural networks with multiple layers to model complex patterns. Key concepts:
- Neural Networks: Layers of interconnected nodes (neurons) for feature learning.
- Activation Functions: Introduce non-linearity (e.g., ReLU, sigmoid).
- Loss Functions: Measure model error (e.g., MSE for regression, binary crossentropy for classification).
This script focuses on building and evaluating neural networks with TensorFlow/Keras."""

"""17.3. Data Preprocessing
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

"""17.4. Neural Network for Regression
Build a neural network to predict continuous outcomes (e.g., salary in hr dataset)."""
# Prepare hr dataset for regression
X = hr_df.drop('salary', axis=1)
y = hr_df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build regression model
reg_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile model
reg_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = reg_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Predict and evaluate
y_pred = reg_model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Neural Network Regression MSE: {mse:.2f}')
print(f'Neural Network Regression R^2 Score: {r2:.2f}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Regression Model: Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""17.5. Neural Network for Classification
Build a neural network for binary classification (e.g., high/low sales in sales dataset)."""
# Prepare sales dataset for classification
sales_df['high_sales'] = (sales_df['sales_amount'] > sales_df['sales_amount'].median()).astype(int)
X = sales_df[['marketing_spend', 'store_size']]
y = sales_df['high_sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build classification model
clf_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
clf_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = clf_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Predict and evaluate
y_pred = (clf_model.predict(X_test) > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Neural Network Classification Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Neural Network Classification: Confusion Matrix')
plt.show()

"""17.6. Activation Functions
Common activation functions include:
- ReLU: Introduces non-linearity (max(0, x)).
- Sigmoid: Maps outputs to [0, 1] for binary classification."""

"""17.7. Dropout Regularization
Dropout prevents overfitting by randomly deactivating neurons during training."""
# (Demonstrated in classification model with Dropout(0.2))

"""17.8. Model Evaluation
Evaluate models using metrics like MSE, R² for regression, and accuracy, confusion matrix for classification."""
# Plot classification training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Classification Model: Training History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""17.9. Hyperparameter Tuning
Manually adjust or use grid search for hyperparameters like learning rate or number of neurons."""
# Example: Adjust learning rate
reg_model_lr = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
reg_model_lr.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
reg_model_lr.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_lr = reg_model_lr.predict(X_test).flatten()
print(f'MSE with Learning Rate 0.01: {mean_squared_error(y_test, y_pred_lr):.2f}')

"""17.10. Saving and Loading Models
Save trained models for reuse."""
# Save regression model
reg_model.save('../scripts/dl_utils_reg_model.h5')

# Load model
from tensorflow.keras.models import load_model
loaded_model = load_model('../scripts/dl_utils_reg_model.h5')
print('Loaded model summary:')
loaded_model.summary()

"""17. Deep Learning Exercises"""

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
Fill missing values in `hr_data.csv` with the median of numerical columns."""
def exercise_2():
    df = pd.read_csv('../data/hr_data.csv')
    df.fillna(df.median(numeric_only=True), inplace=True)
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
Split `hr_data.csv` into training (80%) and testing (20%) sets for predicting `salary`."""
def exercise_4():
    df = pd.read_csv('../data/hr_data.csv')
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set shape:', X_train.shape)

exercise_4()

"""Exercise 5: Neural Network Regression
Build and train a neural network to predict `salary` in `hr_data.csv` and compute MSE."""
def exercise_5():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse:.2f}')

exercise_5()

"""Exercise 6: Neural Network Classification
Build a neural network to predict `high_sales` in `sales.csv` and compute accuracy."""
def exercise_6():
    df = pd.read_csv('../data/sales.csv')
    df['high_sales'] = (df['sales_amount'] > df['sales_amount'].median()).astype(int)
    X = df[['marketing_spend', 'store_size']]
    y = df['high_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

exercise_6()

"""Exercise 7: Confusion Matrix
Plot the confusion matrix for the neural network classification model on `sales.csv`."""
def exercise_7():
    df = pd.read_csv('../data/sales.csv')
    df['high_sales'] = (df['sales_amount'] > df['sales_amount'].median()).astype(int)
    X = df[['marketing_spend', 'store_size']]
    y = df['high_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

exercise_7()

"""Exercise 8: Feature Scaling
Scale numerical features in `hr_data.csv` before training a neural network."""
def exercise_8():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print('Scaled features head:')
    print(X_scaled_df.head())

exercise_8()

"""Exercise 9: Add Dropout
Add a Dropout layer to the regression neural network and compute MSE."""
def exercise_9():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE with Dropout: {mse:.2f}')

exercise_9()

"""Exercise 10: Save Model
Save the trained classification neural network to `dl_utils_clf_model.h5`."""
def exercise_10():
    df = pd.read_csv('../data/sales.csv')
    df['high_sales'] = (df['sales_amount'] > df['sales_amount'].median()).astype(int)
    X = df[['marketing_spend', 'store_size']]
    y = df['high_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    model.save('../scripts/dl_utils_clf_model.h5')
    print('Model saved successfully')

exercise_10()

"""Exercise 11: Plot Training History
Plot the training and validation loss for the regression model."""
def exercise_11():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

exercise_11()

"""Exercise 12: Adjust Learning Rate
Train a regression neural network with a learning rate of 0.01 and compute R² score."""
def exercise_12():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2:.2f}')

exercise_12()

"""Exercise 13: Predict New Data
Use the classification neural network to predict `high_sales` for a new data point from `sales.csv`."""
def exercise_13():
    df = pd.read_csv('../data/sales.csv')
    df['high_sales'] = (df['sales_amount'] > df['sales_amount'].median()).astype(int)
    X = df[['marketing_spend', 'store_size']]
    y = df['high_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    new_data = X_test[0:1]
    prediction = (model.predict(new_data) > 0.5).astype(int).flatten()
    print('Prediction for new data:', prediction)

exercise_13()

"""Exercise 14: Add Hidden Layer
Add an additional hidden layer to the regression neural network and compute MSE."""
def exercise_14():
    df = pd.read_csv('../data/hr_data.csv')
    df = pd.get_dummies(df, columns=['department'], drop_first=True)
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE with Extra Layer: {mse:.2f}')

exercise_14()

"""Exercise 15: Classification Report
Generate a classification report for the neural network classification model on `sales.csv`."""
def exercise_15():
    df = pd.read_csv('../data/sales.csv')
    df['high_sales'] = (df['sales_amount'] > df['sales_amount'].median()).astype(int)
    X = df[['marketing_spend', 'store_size']]
    y = df['high_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

exercise_15()