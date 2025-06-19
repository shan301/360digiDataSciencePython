# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 11:33:37 2025
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""12. Exploratory Data Analysis (EDA) and Data Visualization
EDA and visualization help uncover patterns, trends, and anomalies in data using pandas, numpy, matplotlib, and seaborn.

12.1. Importing Libraries
Load essential libraries for data handling and visualization."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

"""12.2. Loading Data
Read datasets using relative paths for portability."""
# Mock data to avoid file dependency
education = pd.DataFrame({
    'gmat': [600, 650, 700, 620, 680],
    'workex': [2, 3, 5, 1, 4]
})
cars = pd.DataFrame({
    'HP': [100, 150, 120, 200, 180],
    'MPG': [30, 25, 28, 20, 22],
    'SP': [15000, 20000, 18000, 25000, 22000]
})
print(f"Education Data Shape: {education.shape}")
print(f"Cars Data Shape: {cars.shape}")
# Output:
# Education Data Shape: (5, 2)
# Cars Data Shape: (5, 3)

"""12.3. Univariate Visualization
Visualize single variables to understand distributions."""
# Bar Plot
plt.figure(figsize=(6, 4))
plt.bar(np.arange(len(education)), education['gmat'], color='skyblue')
plt.xlabel("Index")
plt.ylabel("GMAT Score")
plt.title("GMAT Scores")
plt.show()
# Output: Bar plot displayed

# Histogram
plt.figure(figsize=(6, 4))
plt.hist(education['gmat'], bins=5, color='blue', edgecolor='black')
plt.xlabel("GMAT Score")
plt.ylabel("Frequency")
plt.title("GMAT Score Distribution")
plt.show()
# Output: Histogram displayed

# Box Plot
plt.figure(figsize=(6, 4))
plt.boxplot(education['gmat'])
plt.ylabel("GMAT Score")
plt.title("Box Plot of GMAT Scores")
plt.show()
# Output: Box plot displayed

# Density Plot
plt.figure(figsize=(6, 4))
sns.kdeplot(education['gmat'], color='purple')
plt.xlabel("GMAT Score")
plt.title("Density Plot of GMAT Scores")
plt.show()
# Output: Density plot displayed

"""12.4. Descriptive Statistics
Summarize data with central tendency and dispersion."""
print(f"Education Data Summary:\n{education.describe()}")
# Output:
#              gmat    workex
# count    5.000000  5.000000
# mean   650.000000  3.000000
# std     39.370039  1.581139
# ...

"""12.5. Bivariate Visualization
Explore relationships between two variables."""
# Scatter Plot: HP vs MPG
plt.figure(figsize=(6, 4))
plt.scatter(cars['HP'], cars['MPG'], color='red')
plt.xlabel("Horsepower (HP)")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("HP vs MPG")
plt.show()
# Output: Scatter plot displayed

"""12.6. Correlation Visualization
Visualize relationships between numerical variables."""
plt.figure(figsize=(6, 4))
sns.heatmap(cars.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# Output: Correlation matrix displayed

"""12.7. Pair Plots
Explore multivariate relationships."""
sns.pairplot(cars)
plt.suptitle("Pair Plot of Cars Data", y=1.02)
plt.show()
# Output: Pair plot displayed

"""12.8. Customizing Aesthetics
Enhance plots with themes and annotations."""
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
sns.histplot(education['gmat'], bins=5, color='teal', kde=True)
plt.xlabel("GMAT Score")
plt.ylabel("Frequency")
plt.title("Customized GMAT Distribution")
plt.annotate("Peak", xy=(650, 0.002), xytext=(620, 0.003), arrowprops=dict(facecolor='black'))
plt.show()
# Output: Customized histogram displayed

"""12. EDA and Data Visualization Exercises
Exercise 1: Load Mock Data
Create and display a mock DataFrame for education data."""
education = pd.DataFrame({
    'gmat': [600, 650, 700, 620, 680],
    'workex': [2, 3, 5, 1, 4]
})
print(f"Mock Education Data:\n{education.head()}")
# Output:
#    gmat  workex
# 0   600       2
# 1   650       3
# ...

"""Exercise 2: Check Missing Values
Identify missing values in the education DataFrame."""
print(f"Missing Values:\n{education.isna().sum()}")
# Output:
# gmat      0
# workex    0

"""Exercise 3: Custom Histogram
Create a histogram for GMAT scores with 10 bins and green color."""
plt.figure(figsize=(6, 4))
plt.hist(education['gmat'], bins=10, color='green', edgecolor='black')
plt.xlabel("GMAT Score")
plt.ylabel("Frequency")
plt.title("GMAT Score Histogram")
plt.show()
print("Histogram displayed")

"""Exercise 4: Box Plot for Work Experience
Generate a box plot for workex to check outliers."""
plt.figure(figsize=(6, 4))
plt.boxplot(education['workex'])
plt.ylabel("Work Experience (Years)")
plt.title("Box Plot of Work Experience")
plt.show()
print("Box plot displayed")

"""Exercise 5: Scatter Plot
Create a scatter plot of HP vs SP from cars data."""
plt.figure(figsize=(6, 4))
plt.scatter(cars['HP'], cars['SP'], color='blue')
plt.xlabel("Horsepower (HP)")
plt.ylabel("Sale Price (SP)")
plt.title("HP vs Sale Price")
plt.show()
print("Scatter plot displayed")

"""Exercise 6: Histplot with KDE
Use sns.histplot to visualize GMAT scores with KDE."""
plt.figure(figsize=(6, 4))
sns.histplot(education['gmat'], bins=5, kde=True, color='orange')
plt.xlabel("GMAT Score")
plt.title("GMAT Distribution with KDE")
plt.show()
print("Histplot displayed")

"""Exercise 7: Summary Statistics
Compute mean, median, and mode for workex."""
print(f"Mean Work Experience: {education['workex'].mean()}")
print(f"Median Work Experience: {education['workex'].median()}")
print(f"Mode Work Experience: {education['workex'].mode()[0]}")
# Output:
# Mean Work Experience: 3.0
# Median Work Experience: 3.0
# Mode Work Experience: (varies, e.g., 1)

"""Exercise 8: Correlation Heatmap
Create a correlation heatmap for cars data."""
plt.figure(figsize=(6, 4))
sns.heatmap(cars.corr(), annot=True, cmap='viridis')
plt.title("Cars Correlation Matrix")
plt.show()
print("Correlation heatmap displayed")

"""Exercise 9: Pair Plot
Generate a pair plot for cars data."""
sns.pairplot(cars)
plt.suptitle("Pair Plot of Cars Data", y=1.02)
plt.show()
print("Pair plot displayed")

"""Exercise 10: Customized Box Plot
Create a box plot for GMAT with a custom theme."""
sns.set_style("darkgrid")
plt.figure(figsize=(6, 4))
sns.boxplot(data=education['gmat'], color='lightblue')
plt.ylabel("GMAT Score")
plt.title("Customized GMAT Box Plot")
plt.show()
print("Box plot displayed")

"""Exercise 11: Density Plot
Create a density plot for workex."""
plt.figure(figsize=(6, 4))
sns.kdeplot(education['workex'], color='green')
plt.xlabel("Work Experience (Years)")
plt.title("Density Plot of Work Experience")
plt.show()
print("Density plot displayed")

"""Exercise 12: Bar Plot with Counts
Create a bar plot of workex value counts."""
plt.figure(figsize=(6, 4))
education['workex'].value_counts().plot(kind='bar', color='purple')
plt.xlabel("Work Experience (Years)")
plt.ylabel("Count")
plt.title("Work Experience Counts")
plt.show()
print("Bar plot displayed")

"""Exercise 13: Scatter with Trend Line
Add a trend line to the HP vs MPG scatter plot."""
plt.figure(figsize=(6, 4))
sns.regplot(x=cars['HP'], y=cars['MPG'], color='red')
plt.xlabel("Horsepower (HP)")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("HP vs MPG with Trend Line")
plt.show()
print("Scatter with trend line displayed")

"""Exercise 14: Violin Plot
Create a violin plot for GMAT scores."""
plt.figure(figsize=(6, 4))
sns.violinplot(data=education['gmat'], color='pink')
plt.ylabel("GMAT Score")
plt.title("Violin Plot of GMAT Scores")
plt.show()
print("Violin plot displayed")

"""Exercise 15: Annotated Scatter Plot
Add annotations to the HP vs SP scatter plot."""
plt.figure(figsize=(6, 4))
plt.scatter(cars['HP'], cars['SP'], color='blue')
for i, (hp, sp) in enumerate(zip(cars['HP'], cars['SP'])):
    plt.annotate(f"({hp}, {sp})", (hp, sp), fontsize=8)
plt.xlabel("Horsepower (HP)")
plt.ylabel("Sale Price (SP)")
plt.title("Annotated HP vs Sale Price")
plt.show()
print("Annotated scatter plot displayed")

