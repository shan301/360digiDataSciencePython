# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:02:05 2025

@author: Shantanu
"""

"""Data preprocessing techniques in Python using pandas, numpy, seaborn, scipy.stats, and sklearn.preprocessing. 
Let me break it down into key sections and explain each part.

1. Type Casting
Changing data types of columns using .astype()

Example: data.EmpID = data.EmpID.astype('str') converts employee IDs from integer to string.

Used for ensuring appropriate data types, e.g., categorical data should not be stored as integers.

2. Identifying and Handling Duplicates
data.duplicated() finds duplicate rows.

data.drop_duplicates() removes duplicates.

data.corr() helps identify redundant columns using correlation.

3. Outlier Detection and Treatment
Boxplots (sns.boxplot(df.Salaries)) help visualize outliers.

Interquartile Range (IQR) Method:

Detects outliers beyond 1.5 * IQR from Q1/Q3.

Options to handle:

Trimming: Remove outliers.

Replacing: Cap outliers to limits.

Winsorization: Using Winsorizer() to cap extreme values.

4. Zero and Near-Zero Variance
If a column has zero variance, it provides no useful information.

df.var() helps detect such columns.

5. Discretization
Converting continuous variables into categorical bins using pd.cut().

Example:

python
Copy
Edit
data['Salaries_new'] = pd.cut(data['Salaries'], bins=[min(data.Salaries), data.Salaries.mean(), max(data.Salaries)], labels=["Low", "High"])
Handles issues using include_lowest=True.

6. Dummy Variables and Encoding
pd.get_dummies(df, drop_first=True) converts categorical data into numerical format.

One-Hot Encoding: Uses OneHotEncoder().

Label Encoding: Uses LabelEncoder().

Ordinal Encoding: Uses OrdinalEncoder().

7. Handling Missing Values
df.isna().sum() finds missing values.

Strategies for Filling Missing Values:

Mean/Median (SimpleImputer(strategy='mean') or 'median').

Mode (for categorical variables).

Constant (strategy='constant').

Random Imputation (RandomSampleImputer()).

8. Normality Checking & Transformation
Q-Q Plots (stats.probplot()).

Box-Cox Transformation (for positive values only).

Yeo-Johnson Transformation (works with zero/negative values).

9. Standardization & Normalization
Standardization: StandardScaler() (Mean = 0, Std Dev = 1).

Normalization:

MinMaxScaler() scales between [0,1].

RobustScaler() handles outliers."""




"""Exercise 1: Type Casting
Task:
Convert the following columns in a DataFrame:

EmpID to string

Salaries to integer

age to float32"""
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'EmpID': [101, 102, 103],
    'Salaries': [50000.75, 60000.50, 55000.25],
    'age': [25, 30, 35]
})

# Type Casting
data['EmpID'] = data['EmpID'].astype('str')
data['Salaries'] = data['Salaries'].astype('int64')
data['age'] = data['age'].astype('float32')

# Display Data Types
print(data.dtypes)

"""Exercise 2: Finding and Removing Duplicates
Task:
Identify duplicate rows in the dataset.
Remove them while keeping the first occurrence."""
# Sample Data
data = pd.DataFrame({
    'EmpID': ['101', '102', '103', '101'],
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
    'Salaries': [50000, 60000, 55000, 50000]
})

# Identify duplicates
duplicates = data.duplicated()
print("Duplicate Rows:\n", data[duplicates])

# Remove duplicates (keep first occurrence)
data_cleaned = data.drop_duplicates()
print("\nData After Removing Duplicates:\n", data_cleaned)

"""Exercise 3: Outlier Detection Using IQR
Task:
Identify and remove outliers in the Salaries column using the IQR method."""
import numpy as np

# Sample Data
data = pd.DataFrame({'Salaries': [30000, 50000, 60000, 55000, 200000]})

# Calculate IQR
Q1 = data['Salaries'].quantile(0.25)
Q3 = data['Salaries'].quantile(0.75)
IQR = Q3 - Q1

# Define limits
lower_limit = Q1 - (1.5 * IQR)
upper_limit = Q3 + (1.5 * IQR)

# Remove Outliers
data_cleaned = data[(data['Salaries'] >= lower_limit) & (data['Salaries'] <= upper_limit)]
print(data_cleaned)

"""Exercise 4: Creating Dummy Variables
Task:
Convert the Department column into dummy variables."""
# Sample Data
data = pd.DataFrame({'Department': ['HR', 'IT', 'Finance', 'HR', 'IT']})

# Create Dummy Variables
data_dummies = pd.get_dummies(data, drop_first=True)
print(data_dummies)

"""Exercise 5: Handling Missing Values
Task:
Replace missing values in the Salaries column with the mean value."""
import numpy as np
from sklearn.impute import SimpleImputer

# Sample Data
data = pd.DataFrame({'Salaries': [50000, 60000, np.nan, 55000, np.nan]})

# Mean Imputation
imputer = SimpleImputer(strategy='mean')
data['Salaries'] = imputer.fit_transform(data[['Salaries']])

print(data)

"""Exercise 6: Standardization & Normalization
Task:
Standardize the Salaries column using StandardScaler().
Normalize it using MinMaxScaler()."""
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample Data
data = pd.DataFrame({'Salaries': [50000, 60000, 55000, 70000, 80000]})

# Standardization
scaler = StandardScaler()
data['Salaries_Standardized'] = scaler.fit_transform(data[['Salaries']])

# Normalization
minmax = MinMaxScaler()
data['Salaries_Normalized'] = minmax.fit_transform(data[['Salaries']])

print(data)

"""

