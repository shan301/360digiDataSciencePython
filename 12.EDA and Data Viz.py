# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 11:33:37 2025

@author: Shantanu
"""

"""This Python script focuses on Exploratory Data Analysis (EDA) and Data Visualization using matplotlib, seaborn, numpy, and pandas. Here's a breakdown of what the script does:

1. Importing Libraries
matplotlib.pyplot: Used for creating static, animated, and interactive visualizations.

numpy: For numerical operations.

pandas: For handling tabular data.

seaborn: A statistical visualization library built on matplotlib.


2. Reading and Exploring Data
Reads education data from a CSV file (education.csv).

Uses .shape to check the number of rows and columns.

3. Univariate Data Visualization
Bar Plot: Displays GMAT scores using plt.bar().

Histogram:

Plots GMAT scores (plt.hist()).

Plots work experience in different configurations.

Box Plot: Visualizes the spread and outliers of GMAT scores.

Density Plot: Estimates the distribution of GMAT scores using sns.kdeplot().

Descriptive Statistics: Uses education.describe() to get central tendency, dispersion, and shape of data.



4. Bivariate Data Visualization
Reads car data from Cars.csv.

Uses .info() to check data types and missing values.

Scatter Plots:

Horsepower (HP) vs. Miles per Gallon (MPG).

Horsepower (HP) vs. Sale Price (SP)."""




"""To make the file path relative, you need to reference it relative to your script's location instead of using an absolute path. Here’s how:

1. Using os.path
Modify your script to use the os module:"""
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the relative path to the CSV file
education_path = os.path.join(script_dir, "education.csv")
cars_path = os.path.join(script_dir, "Cars.csv")

# Read CSV files using relative paths
education = pd.read_csv(education_path)
cars = pd.read_csv(cars_path)

"""2. Using Path from pathlib (Recommended)
The pathlib module provides an easier way to handle file paths:"""
from pathlib import Path

# Get the directory of the script
script_dir = Path(__file__).parent

# Construct the relative path
education_path = script_dir / "education.csv"
cars_path = script_dir / "Cars.csv"

# Read CSV files
education = pd.read_csv(education_path)
cars = pd.read_csv(cars_path)

"""Why Use Relative Paths?
✅ Works across different computers.
✅ No need to change paths manually when moving files.
✅ Improves script portability."""




""""Exercise 1: Load Data Using a Relative Path 
Task: Modify the script to use a relative path instead of an absolute path when loading education.csv and Cars.csv.

✅ Solution:
Use pathlib for better path management:
"""
from pathlib import Path
import pandas as pd

# Get the script directory
script_dir = Path(__file__).parent

# Construct relative paths
education_path = script_dir / "education.csv"
cars_path = script_dir / "Cars.csv"

# Load data
education = pd.read_csv(education_path)
cars = pd.read_csv(cars_path)

print(education.head())  # Display first 5 rows
print(cars.head())  


"""Exercise 2: Identify Missing Values
👉 Task: Check if there are any missing values in both datasets.

✅ Solution:
Use .isnull().sum() to count missing values per column:"""
print("Missing values in education dataset:")
print(education.isnull().sum())

print("\nMissing values in cars dataset:")
print(cars.isnull().sum())


"""Exercise 3: Plot a Histogram with Custom Bins
👉 Task: Create a histogram for GMAT scores with bins of size 20 and a blue color."""
import matplotlib.pyplot as plt

plt.hist(education.gmat, bins=20, color='blue', edgecolor='black')
plt.xlabel("GMAT Score")
plt.ylabel("Frequency")
plt.title("Distribution of GMAT Scores")
plt.show()


"""Exercise 4: Create a Box Plot for workex
👉 Task: Generate a box plot to check for outliers in the workex column."""
plt.boxplot(education.workex)
plt.xlabel("Work Experience")
plt.title("Box Plot of Work Experience")
plt.show()


"""Exercise 5: Scatter Plot with Labels
👉 Task: Create a scatter plot for Horsepower (HP) vs. Miles Per Gallon (MPG) from the cars dataset, including labels."""
plt.scatter(x=cars['HP'], y=cars['MPG'], color='red')
plt.xlabel("Horsepower (HP)")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("Horsepower vs. MPG")
plt.show()


"""Exercise 6: Replace distplot() with histplot()
👉 Task: Modify the deprecated sns.distplot() function to sns.histplot().

✅ Solution:
Replace:"""
sns.distplot(education.gmat)  # Deprecated

import seaborn as sns
sns.histplot(education.gmat, kde=True)
plt.xlabel("GMAT Score")
plt.title("Distribution of GMAT Scores with KDE")
plt.show()


"""Exercise 7: Compute Summary Statistics
👉 Task: Print summary statistics (mean, median, mode) for GMAT scores."""
print("Mean GMAT Score:", education.gmat.mean())
print("Median GMAT Score:", education.gmat.median())
print("Mode GMAT Score:", education.gmat.mode()[0])

