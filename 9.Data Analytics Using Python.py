# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:01:23 2025

@author: Shantanu
"""

""" Data Analytics using python-
fundamental libraries such as:

*NumPy - For numerical operations, array manipulations, memory efficiency, broadcasting, and matrix operations.
*Pandas - For data manipulation and analysis, including creating data frames, accessing elements, 
performing operations like merging, concatenation, and handling missing data.
*Matplotlib - For data visualization, including bar plots, histograms, and box plots.
*Seaborn - For advanced visualization, including detecting outliers in data.

1. NumPy (Numerical Python)
NumPy is a powerful library for handling numerical data, offering multi-dimensional arrays and various mathematical functions.

Key Features
Efficient array operations
Broadcasting (operations between different-sized arrays)
Memory efficiency
Random number generation """

import numpy as np

# Creating a NumPy array
x = np.array([10, 21, 3, 14, 15, 16])
print(x * 2)  # Multiplies each element by 2

# Boolean filtering
print(x[x > 10])  # Returns elements greater than 10

# Random integer generation
from numpy import random
print(random.randint(3, 9))  # Generates a random integer between 3 and 9

# Memory sharing example
a = np.arange(10)
b = a[::2]  # b is a view of a, sharing the same memory
print(np.shares_memory(a, b))

"""2. Data Structures
Python provides multiple ways to store and manipulate data, such as arrays, lists, and matrices.

Arrays """

from array import array

arr = array('i', [10, 20, 30, 40, 50])
arr.insert(1, 60)  # Insert 60 at index 1
arr.remove(40)  # Remove 40
arr[2] = 80  # Update index 2
print(arr)

"""2D Arrays"""

T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5]]
print(T[1][2])  # Access value at row 1, column 2

"""Matrices"""

import numpy as np

m = np.matrix('1 2; 3 4')  # Creating a 2x2 matrix
print(m)


"""3. Pandas (Data Manipulation)
Pandas is essential for working with tabular data.

Creating DataFrames"""

import pandas as pd

df = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})
print(df)

"""Accessing Data"""

print(df.X1)  # Access column using dot notation
print(df["X1"])  # Access column using bracket notation
print(df.iloc[0:3, 1])  # Access rows 0 to 3 in column at index 1
"""Statistics"""

print(df['X1'].mean())  # Mean
print(df.describe())  # Summary statistics

"""Merging & Concatenation"""

df1 = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 8, 12]})
df2 = pd.DataFrame({"X1": [1, 2, 3, 4], "X3": [14, 18, 112, 15]})

merged = pd.merge(df1, df2, on="X1")  # Merge on X1
print(merged)

concatenated = pd.concat([df1, df2])  # Concatenate dataframes
print(concatenated)

"""Handling Missing Data"""

df = pd.DataFrame({"grade1": [1, 2, 3, 4, np.nan], "grade2": [np.nan, 11, 12, 100, 200]})
print(df.isna().sum())  # Count missing values
df.dropna()  # Remove rows with missing values


"""4. Reading External Files"""

df = pd.read_csv("education.csv")  # Read CSV file
print(df.info())  # Get data summary
print(df.describe())  # Get statistics


"""5. Exploratory Data Analysis (EDA)
Measures of Central Tendency"""

print(df['workex'].mean())  # Mean
print(df['workex'].median())  # Median
print(df['workex'].mode())  # Mode

"""Measures of Dispersion"""

print(df['workex'].var())  # Variance
print(df['workex'].std())  # Standard Deviation

"""Skewness & Kurtosis"""

print(df['workex'].skew())  # Skewness
print(df['workex'].kurt())  # Kurtosis


"""6. Data Visualization
Matplotlib"""

import matplotlib.pyplot as plt
import numpy as np

# Bar plot
plt.bar(height=df['gmat'], x=np.arange(len(df)))
plt.show()

# Histogram
plt.hist(df['gmat'])
plt.show()

# Box plot
plt.boxplot(df['gmat'])
plt.show()

"""Seaborn"""

import seaborn as sns

sns.boxplot(df['gmat'])  # Box plot for outliers
plt.show()



"""Conclusion
This script covers essential data analytics concepts:

NumPy for efficient numerical computations.
Pandas for data manipulation and analysis.
Matplotlib & Seaborn for visualization.
Exploratory Data Analysis (EDA) for insights."""


"""Exercises-
 1. NumPy Exercises
Q1: Create a NumPy array of integers from 10 to 50. Print only even numbers from it."""
import numpy as np

arr = np.arange(10, 51)  # Create array from 10 to 50
even_numbers = arr[arr % 2 == 0]  # Select even numbers
print(even_numbers)

"""Q2: Create a 3x3 NumPy array with values ranging from 1 to 9."""

arr = np.arange(1, 10).reshape(3, 3)
print(arr)

"""Q3: Generate a random 4x4 matrix and replace all values greater than 0.5 with 1, and the rest with 0."""

arr = np.random.rand(4, 4)  # Generate random numbers
arr[arr > 0.5] = 1
arr[arr <= 0.5] = 0
print(arr)


"""2. Pandas Exercises
Q4: Create a DataFrame with 3 columns: "Name", "Age", and "City". Add 5 rows of data."""

import pandas as pd

data = {
    "Name": ["Amit", "Sara", "John", "Lisa", "Raj"],
    "Age": [25, 30, 27, 22, 29],
    "City": ["Mumbai", "Delhi", "New York", "London", "Bangalore"]
}

df = pd.DataFrame(data)
print(df)

"""Q5: Load a CSV file and display its first 5 rows."""

df = pd.read_csv("education.csv")  # Replace with your file path
print(df.head())  # Show first 5 rows

"""Q6: Add a new column "Salary" to the DataFrame in Q4 with random values between 40,000 and 80,000. """

import numpy as np

df["Salary"] = np.random.randint(40000, 80000, size=len(df))
print(df)

"""Q7: Filter out rows where Age is greater than 25."""

filtered_df = df[df["Age"] > 25]
print(filtered_df)

"""Q8: Find the average salary from the DataFrame."""

average_salary = df["Salary"].mean()
print("Average Salary:", average_salary)



"""3. Data Visualization Exercises
Q9: Create a bar plot showing the number of people from each city in Q4."""

import matplotlib.pyplot as plt

df["City"].value_counts().plot(kind="bar", color="skyblue")
plt.xlabel("City")
plt.ylabel("Count")
plt.title("Number of People in Each City")
plt.show()

"""Q10: Create a histogram of ages from the DataFrame in Q4."""

plt.hist(df["Age"], bins=5, color="green", edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()

"""Q11: Create a box plot to visualize salary distribution from the DataFrame in Q4."""

import seaborn as sns

sns.boxplot(df["Salary"])
plt.title("Salary Distribution")
plt.show()



"""Q12: Load the CSV file ("education.csv"), find missing values, and replace them with the column mean."""

df = pd.read_csv("education.csv")
print("Missing Values:\n", df.isna().sum())  # Count missing values

df.fillna(df.mean(), inplace=True)  # Replace with column mean
print("After Filling Missing Values:\n", df.isna().sum())  # Check again
