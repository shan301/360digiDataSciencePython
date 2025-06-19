# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:01:23 2025
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""9. Data Analytics Using Python
This module covers essential libraries and techniques for data analytics, including numerical operations (NumPy), data manipulation (pandas), and visualization (Matplotlib, Seaborn).

9.1. NumPy Basics
NumPy provides efficient array operations for numerical computations."""
import numpy as np

# Create and manipulate array
arr = np.array([10, 21, 3, 14, 15, 16])
print(f"Array * 2: {arr * 2}")  # Output: Array * 2: [20 42  6 28 30 32]
print(f"Values > 10: {arr[arr > 10]}")  # Output: Values > 10: [21 14 15 16]

# Random number generation
print(f"Random integer (3-9): {np.random.randint(3, 9)}")  # Output: Varies, e.g., 5

"""9.2. Python Arrays
Arrays are lightweight alternatives to lists for numerical data."""
from array import array

arr = array('i', [10, 20, 30, 40, 50])
arr.insert(1, 60)
arr.remove(40)
arr[2] = 80
print(f"Array: {arr.tolist()}")  # Output: Array: [10, 60, 80, 30, 50]

"""9.3. 2D Arrays and Matrices
Handle multi-dimensional data structures."""
# 2D array
T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5]]
print(f"Element at [1][2]: {T[1][2]}")  # Output: Element at [1][2]: 10

# Matrix using NumPy
matrix = np.array([[1, 2], [3, 4]])
print(f"Matrix:\n{matrix}")  # Output: Matrix: [[1 2], [3 4]]

"""9.4. Pandas DataFrames
Pandas is ideal for tabular data manipulation."""
import pandas as pd

df = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})
print(f"DataFrame:\n{df}")
# Output:
#    X1  X2
# 0   1   4
# 1   2   8
# 2   3  12

"""9.5. Accessing Data
Access columns, rows, or specific elements."""
print(f"Column X1: {df['X1'].tolist()}")  # Output: Column X1: [1, 2, 3]
print(f"Row 0-2, Column X2: {df.iloc[0:3, 1].tolist()}")  # Output: Row 0-2, Column X2: [4, 8, 12]

"""9.6. Data Statistics
Compute summary statistics."""
print(f"Mean of X1: {df['X1'].mean()}")  # Output: Mean of X1: 2.0
print(f"Summary:\n{df.describe()}")
# Output:
#              X1         X2
# count  3.000000   3.000000
# mean   2.000000   8.000000
# std    1.000000   4.000000
# ...

"""9.7. Merging and Concatenation
Combine DataFrames using merge or concat."""
df1 = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})
df2 = pd.DataFrame({"X1": [1, 2, 3, 4], "X3": [14, 18, 112, 15]})
merged = pd.merge(df1, df2, on="X1")
print(f"Merged:\n{merged}")
# Output:
#    X1  X2   X3
# 0   1   4   14
# 1   2   8   18
# 2   3  12  112

"""9.8. Handling Missing Data
Identify and manage missing values."""
df_missing = pd.DataFrame({"grade1": [1, 2, 3, 4, np.nan], "grade2": [np.nan, 11, 12, 100, 200]})
print(f"Missing Values:\n{df_missing.isna().sum()}")
# Output:
# grade1    1
# grade2    1

"""9.9. Data Transformation with apply/map
Apply custom functions to transform data."""
df['X1_squared'] = df['X1'].apply(lambda x: x**2)
print(f"DataFrame with X1_squared:\n{df}")
# Output:
#    X1  X2  X1_squared
# 0   1   4           1
# 1   2   8           4
# 2   3  12           9

"""9.10. Handling Categorical Data
Encode categorical variables for analysis."""
df_cat = pd.DataFrame({"Category": ["A", "B", "A", "C"]})
df_cat['Category_Code'] = df_cat['Category'].astype('category').cat.codes
print(f"Categorical Data:\n{df_cat}")
# Output:
#   Category  Category_Code
# 0        A              0
# 1        B              1
# 2        A              0
# 3        C              2

"""9.11. Time Series Basics
Work with datetime data for temporal analysis."""
dates = pd.date_range("2023-01-01", periods=3, freq="D")
df_time = pd.DataFrame({"Value": [10, 15, 20]}, index=dates)
print(f"Time Series DataFrame:\n{df_time}")
# Output:
#             Value
# 2023-01-01     10
# 2023-01-02     15
# 2023-01-03     20

"""9.12. Reading External Files
Read data from CSV or Excel files."""
try:
    df_csv = pd.DataFrame({'workex': [1, 2, 3], 'gmat': [600, 650, 700]})  # Mock data
    print(f"Mock CSV Data:\n{df_csv}")
except FileNotFoundError:
    print("Error: CSV file not found")

"""9.13. Exploratory Data Analysis (EDA)
Compute measures of central tendency and dispersion."""
print(f"Mean workex: {df_csv['workex'].mean()}")  # Output: Mean workex: 2.0
print(f"Std gmat: {df_csv['gmat'].std()}")       # Output: Std gmat: 50.0

"""9.14. Data Visualization
Visualize data using Matplotlib and Seaborn."""
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot
plt.bar(np.arange(len(df_csv)), df_csv['gmat'], color='skyblue')
plt.title("GMAT Scores")
plt.show()

# Seaborn boxplot
sns.boxplot(data=df_csv['gmat'])
plt.title("GMAT Distribution")
plt.show()

"""9. Data Analytics Exercises
Exercise 1: NumPy Even Numbers
Create a NumPy array from 10 to 50 and print even numbers."""
arr = np.arange(10, 51)
print(f"Even numbers: {arr[arr % 2 == 0]}")
# Output: Even numbers: [10 12 14 ... 48 50]

"""Exercise 2: NumPy 3x3 Matrix
Create a 3x3 NumPy array with values from 1 to 9."""
matrix = np.arange(1, 10).reshape(3, 3)
print(f"3x3 Matrix:\n{matrix}")
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

"""Exercise 3: NumPy Binary Matrix
Generate a 4x4 random matrix and replace values > 0.5 with 1, others with 0."""
np.random.seed(42)  # For reproducibility
arr = np.random.rand(4, 4)
arr[arr > 0.5] = 1
arr[arr <= 0.5] = 0
print(f"Binary Matrix:\n{arr}")
# Output: (varies based on seed)

"""Exercise 4: Create DataFrame
Create a DataFrame with Name, Age, and City columns for 5 rows."""
data = {
    "Name": ["Amit", "Sara", "John", "Lisa", "Raj"],
    "Age": [25, 30, 27, 22, 29],
    "City": ["Mumbai", "Delhi", "New York", "London", "Bangalore"]
}
df = pd.DataFrame(data)
print(f"DataFrame:\n{df}")
# Output:
#     Name  Age       City
# 0   Amit   25     Mumbai
# 1   Sara   30      Delhi
# ...

"""Exercise 5: Add Salary Column
Add a Salary column with random values between 40000 and 80000."""
np.random.seed(42)
df['Salary'] = np.random.randint(40000, 80000, size=len(df))
print(f"DataFrame with Salary:\n{df}")
# Output: (varies based on seed)

"""Exercise 6: Filter by Age
Filter rows where Age is greater than 25."""
filtered = df[df['Age'] > 25]
print(f"Filtered DataFrame:\n{filtered}")
# Output:
#     Name  Age       City  Salary
# 1   Sara   30      Delhi   ...
# 2   John   27  New York   ...
# 4    Raj   29  Bangalore  ...

"""Exercise 7: Average Salary
Compute the average Salary."""
print(f"Average Salary: {df['Salary'].mean()}")
# Output: Average Salary: (varies)

"""Exercise 8: Handle Missing Values
Create a DataFrame with missing values and fill with column means."""
data_missing = {"A": [1, 2, np.nan], "B": [np.nan, 5, 6]}
df_missing = pd.DataFrame(data_missing)
df_missing.fillna(df_missing.mean(), inplace=True)
print(f"Filled DataFrame:\n{df_missing}")
# Output:
#      A    B
# 0  1.0  5.5
# 1  2.0  5.0
# 2  1.5  6.0

"""Exercise 9: Apply Transformation
Add a column with Age doubled using apply."""
df['Age_Doubled'] = df['Age'].apply(lambda x: x * 2)
print(f"DataFrame with Age_Doubled:\n{df}")
# Output: (includes Age_Doubled column)

"""Exercise 10: Encode Categorical Data
Encode the City column as categorical codes."""
df['City_Code'] = df['City'].astype('category').cat.codes
print(f"DataFrame with City_Code:\n{df}")
# Output: (includes City_Code column)

"""Exercise 11: Time Series Resampling
Create a time series DataFrame and resample to compute daily mean."""
dates = pd.date_range("2023-01-01", periods=6, freq="H")
df_time = pd.DataFrame({"Value": [10, 12, 14, 16, 18, 20]}, index=dates)
daily_mean = df_time.resample('D').mean()
print(f"Daily Mean:\n{daily_mean}")
# Output:
#             Value
# 2023-01-01   15.0

"""Exercise 12: Bar Plot
Create a bar plot of City counts."""
df['City'].value_counts().plot(kind="bar", color="skyblue")
plt.xlabel("City")
plt.ylabel("Count")
plt.title("Number of People by City")
plt.show()
print("Bar plot displayed")

"""Exercise 13: Histogram
Create a histogram of Ages."""
plt.hist(df['Age'], bins=5, color="green", edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()
print("Histogram displayed")

"""Exercise 14: Box Plot
Create a box plot of Salaries."""
sns.boxplot(data=df['Salary'])
plt.title("Salary Distribution")
plt.show()
print("Box plot displayed")

"""Exercise 15: Merge DataFrames
Merge two DataFrames on a common column."""
df1 = pd.DataFrame({"ID": [1, 2, 3], "Name": ["Amit", "Sara", "John"]})
df2 = pd.DataFrame({"ID": [1, 2, 4], "Score": [85, 90, 88]})
merged = pd.merge(df1, df2, on="ID", how="inner")
print(f"Merged DataFrame:\n{merged}")
# Output:
#    ID  Name  Score
# 0   1  Amit     85
# 1   2  Sara     90
