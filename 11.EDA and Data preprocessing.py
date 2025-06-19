# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 2024
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""11. Exploratory Data Analysis (EDA) and Data Preprocessing
EDA and preprocessing prepare raw data for analysis or modeling using libraries like pandas, numpy, seaborn, scipy, and sklearn. These techniques ensure data quality and compatibility with algorithms.

11.1. Type Casting
Convert column data types to appropriate formats."""
import pandas as pd

data = pd.DataFrame({
    'EmpID': [101, 102, 103],
    'Salaries': [50000.75, 60000.50, 55000.25],
    'Age': [25, 30, 35]
})
data['EmpID'] = data['EmpID'].astype('str')
data['Salaries'] = data['Salaries'].astype('int64')
data['Age'] = data['Age'].astype(float)
print(f"Data Types:\n{data.dtypes}")
# Output:
# EmpID       object
# Salaries     int64
# Age        float64

"""11.2. Handling Duplicates
Identify and remove duplicate rows or redundant columns."""
data = pd.DataFrame({
    'EmpID': ['101', '101', '102'],
    'Name': ['Alice', 'Alice', 'Bob'],
    'Salaries': [50000, 50000, 60000]
})
print(f"Duplicates:\n{data[data.duplicated()]}")
# Output:
# EmpID   Name  Salaries
# 1    101  Alice    50000
data_cleaned = data.drop_duplicates()
print(f"\nAfter removing duplicates:\n{data_cleaned}")
# Output:
# EmpID   Name  Salaries
# 0    101  Alice    50000
# 2    102    60000     Bob

"""11.3. Outlier Detection with IQR
Detect and treat outliers using the Interquartile Range method."""
import numpy as np

data = pd.DataFrame({'Salaries': [30000, 50000, 60000, 55000, 200000]})
Q1 = data['Salaries'].quantile(0.25)
Q3 = data['Salaries'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
data_cleaned = data[(data['Salaries'] >= lower_limit) & (data['Salaries'] <= upper_limit)]
print(f"Data without outliers:\n{data_cleaned}")
# Output:
#    Salaries
# 0    30000
# 1    50000
# 2    60000
# 3    55000

"""11.4. Zero and Low Variance
Identify columns with near-zero variance."""
data = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3]})
print(f"Variance:\n{data.var()}")
# Output:
# A    0.0
# B    1.0
print("Column 'A' has zero variance and can be dropped.")

"""11.5. Discretization
Convert continuous data into categorical bins."""
data = pd.DataFrame({'Salaries': [20000, 30000, 50000, 70000]})
data['Salary_bin'] = pd.cut(data['Salaries'], bins=[0, 35000, 60000, float('inf')], labels=['Low', 'Medium', 'High'], include_lowest=True)
print(f"Data with bins:\n{data}")
# Output:
#    Salaries Salary_bin
# 0    20000       Low
# 1    30000       Low
# 2    50000    Medium
# 3    70000      High

"""11.6. Dummy Variables and Encoding
Encode categorical data into numerical format."""
data = pd.DataFrame({'Department': ['HR', 'IT', 'Finance']})
dummies = pd.get_dummies(data, columns=['Department'], drop_first=True)
print(f"Dummy Variables:\n{dummies}")
# Output:
#    Department_IT  Department_Finance
# 0             0                  0
# 1             1                  0
# 2             0                  1

"""11.7. Handling Missing Values
Impute missing values using various strategies."""
from sklearn.impute import SimpleImputer

data = pd.DataFrame({'Salaries': [50000, 60000, np.nan, 55000, np.nan]})
imputer = SimpleImputer(strategy='mean')
data['Salaries'] = imputer.fit_transform(data[['Salaries']])
print(f"Data with imputed values:\n{data}")
# Output:
#      Salaries
# 0  55000.000000
# 1  60000.000000
# 2  55000.000000
# 3  55000.000000
# 4  55000.000000

"""11. Normality Checking
Check data distribution using Q-Q plots."""
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.DataFrame({'Salaries': [50000, 55000, 60000, 65000, 70000]})
stats.probplot(data['Salaries'], dist="norm", plot=plt)
plt.title("Q-Q Plot for Salaries")
plt.show()
print("Q-Q plot displayed")

"""11.9. Standardization
Scale data to zero mean and unit variance."""
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 70000, 80000]})
scaler = StandardScaler()
data['Salaries_std'] = scaler.fit_transform(data[['Salaries']])
print(f"Standardized Data:\n{data}")
# Output:
#    Salaries  Salaries_std
# 0    50000    -1.264911
# 1    60000    -0.632456
# 2    55000    -0.948683
# 3    70000     0.000000
# 4    80000     0.632456

"""11.10. Normalization
Scale data to a fixed range, typically [0, 1]."""
from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 70000, 80000]})
scaler = MinMaxScaler()
data['Salaries_norm'] = scaler.fit_transform(data[['Salaries']])
print(f"Normalized Data:\n{data}")
# Output:
#    Salaries  Salaries_norm
#  0    50000       0.000000
# 1    60000       0.333333
# 2    55000       0.166667
# 3    70000       0.666667
# 4    80000       1.000000

"""11.11. Robust Scaling
Scale data robust to outliers using RobustScaler."""
from sklearn.preprocessing import RobustScaler

data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 200000, 80000]})
scaler = RobustScaler()
data['Salaries_robust'] = scaler.fit_transform(data[['Salaries']])
print(f"Robust Scaled Data:\n{data}")
# Output:
#    Salaries  Salaries_robust
# 0    50000       -0.666667
# 1    60000        0.000000
# 2    55000       -0.333333
# 3   200000        9.666667
# 4    80000        1.333333

"""11. Handling Imbalanced Data
Address class imbalance using resampling techniques."""
from imblearn.over_sampling import SMOTE

data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5, 6],
    'Target': [0, 0, 0, 1, 1, 0]
})
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(data[['Feature']], data['Target'])
resampled = pd.DataFrame({'Feature': X_res.flatten(), 'Target': y_res})
print(f"Resampled Data:\n{resampled}")
# Output: (balanced Target classes)

"""11.13. Data Sampling
Use random or stratified sampling for large datasets."""
data = pd.DataFrame({'Value': range(100), 'Class': ['A']*50 + ['B']*50})
sample = data.sample(n=10, random_state=42)
print(f"Random Sample:\n{sample}")
# Output: (10 rows)

"""11. EDA and Preprocessing Exercises
Exercise 1: Type Casting
Convert EmpID to string, Salaries to int, and Age to float."""
data = pd.DataFrame({
    'EmpID': [101, 102, 103],
    'Salaries': [50000.75, 60000.50, 55000.25],
    'Age': [25, 30, 35]
})
data['EmpID'] = data['EmpID'].astype('str')
data['Salaries'] = data['Salaries'].astype('int64')
data['Age'] = data['Age'].astype('float32')
print(f"Data Types:\n{data.dtypes}")
# Output:
# EmpID       object
# Salaries     int64
# Age        float32

"""Exercise 2: Remove Duplicates
Identify and remove duplicate rows, keeping the first occurrence."""
data = pd.DataFrame({
    'EmpID': ['101', '101', '102'],
    'Name': ['Alice', 'Alice', 'Bob'],
    'Salaries': [50000, 50000, 60000]
})
print(f"Duplicates:\n{data[data.duplicated()]}")
data_clean = data.drop_duplicates()
print(f"\nCleaned Data:\n{data_clean}")

"""Exercise 3: Outlier Detection
Remove outliers in Salaries using IQR."""
data = pd.DataFrame({'Salaries': [30000, 50000, 60000, 55000, 200000]})
Q1 = data['Salaries'].quantile(0.25)
Q3 = data['Salaries'].quantile(0.75)
IQR = Q3 - Q1
data_clean = data[(data['Salaries'] >= Q1 - 1.5 * IQR) & (data['Salaries'] <= Q3 + 1.5 * IQR)]
print(f"Data without outliers:\n{data_clean}")

"""Exercise 4: Create Dummy Variables
Convert Department column into dummy variables."""
data = pd.DataFrame({'Department': ['HR', 'IT', 'Finance']})
dummies = pd.get_dummies(data, columns=['Department'], drop_first=True)
print(f"Dummies:\n{dummies}")

"""Exercise 5: Impute Missing Values
Replace missing Salaries with mean."""
data = pd.DataFrame({'Salaries': [50000, 60000, np.nan, 55000, np.nan]})
imputer = SimpleImputer(strategy='mean')
data['Salaries'] = imputer.fit_transform(data[['Salaries']])
print(f"Imputed Data:\n{data}")

"""Exercise 6: Standardize Data
Standardize Salaries using StandardScaler."""
data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 70000, 80000]})
scaler = StandardScaler()
data['Salaries_std'] = scaler.fit_transform(data[['Salaries']])
print(f"Standardized Data:\n{data}")

"""Exercise 7: Normalize Data
Normalize Salaries using MinMaxScaler."""
data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 70000, 80000]})
scaler = MinMaxScaler()
data['Salaries_norm'] = scaler.fit_transform(data[['Salaries']])
print(f"Normalized Data:\n{data}")

"""Exercise 8: Robust Scaling
Scale Salaries using RobustScaler."""
data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 200000, 80000]})
scaler = RobustScaler()
data['Salaries_robust'] = scaler.fit_transform(data[['Salaries']])
print(f"Robust Scaled Data:\n{data}")

"""Exercise 9: Discretize Salaries
Bin Salaries into Low, Medium, High categories."""
data = pd.DataFrame({'Salaries': [20000, 30000, 50000, 70000]})
data['Salary_bin'] = pd.cut(data['Salaries'], bins=[0, 35000, 60000, float('inf')], labels=['Low', 'Medium', 'High'], include_lowest=True)
print(f"Binned Data:\n{data}")

"""Exercise 10: Check Normality
Create a Q-Q plot for Salaries."""
data = pd.DataFrame({'Salaries': [50000, 55000, 60000, 65000, 70000]})
stats.probplot(data['Salaries'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
print("Q-Q plot displayed")

"""Exercise 11: Label Encoding
Encode Department column using LabelEncoder."""
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({'Department': ['HR', 'IT', 'Finance']})
le = LabelEncoder()
data['Department_Code'] = le.fit_transform(data['Department'])
print(f"Encoded Data:\n{data}")

"""Exercise 12: Handle Imbalanced Data
Use SMOTE to balance a dataset."""
data = pd.DataFrame({
    'Feature': [1, 2, 3, 4, 5, 6],
    'Target': [0, 0, 0, 1, 1, 0]
})
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(data[['Feature']], data['Target'])
resampled = pd.DataFrame({'Feature': X_res.flatten(), 'Target': y_res})
print(f"Resampled Data:\n{resampled}")

"""Exercise 13: Random Sampling
Sample 5 rows randomly from a DataFrame."""
data = pd.DataFrame({'Value': range(20)})
sample = data.sample(n=5, random_state=42)
print(f"Sampled Data:\n{sample}")

"""Exercise 14: Correlation Analysis
Compute and visualize correlation matrix."""
import seaborn as sns

data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [5, 4, 3, 2, 1]
})
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
print("Correlation matrix displayed")

"""Exercise 15: Winsorization
Cap extreme values in Salaries using winsorization."""
from scipy.stats import mstats

data = pd.DataFrame({'Salaries': [30000, 50000, 60000, 55000, 200000]})
data['Salaries_winsor'] = mstats.winsorize(data['Salaries'], limits=[0.05, 0.05])
print(f"Winsorized Data:\n{data}")

