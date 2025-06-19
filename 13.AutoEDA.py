# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:36:44 2025

@author: Shantanu
"""

"""13. Automated Exploratory Data Analysis (AutoEDA)
AutoEDA uses Python libraries to automate the process of exploring and visualizing datasets, generating insights quickly. Tools like pandas, ydata-profiling, Sweetviz, AutoViz, and D-Tale are commonly used.

13.1. Basic Dataset Summary with Pandas
Pandas provides methods to generate quick summaries of a dataset."""
import pandas as pd

data = {"age": [25, 30, 35], "salary": [50000, 60000, 75000]}
df = pd.DataFrame(data)
print("Dataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

"""13.2. Missing Value Analysis
Identifying missing values is critical in AutoEDA."""
data = {"name": ["Alice", None, "Charlie"], "score": [85, 90, None]}
df = pd.DataFrame(data)
print("Missing Values:")
print(df.isnull().sum())

"""13.3. Correlation Analysis
Correlation matrices help identify relationships between numerical variables."""
data = {"x": [1, 2, 3, 4], "y": [2, 4, 5, 8]}
df = pd.DataFrame(data)
correlation = df.corr()
print("Correlation Matrix:")
print(correlation)

"""13.4. Automated EDA with ydata-profiling
ydata-profiling generates a comprehensive HTML report for EDA."""
# Note: For demonstration, this is commented out as it requires installation and file output
# from ydata_profiling import ProfileReport
# data = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 75000]})
# profile = ProfileReport(data, title="EDA Report")
# profile.to_file("eda_report.html")
print("ydata-profiling can generate a detailed EDA report (commented for simplicity).")

"""13.5. Automated EDA with Sweetviz
Sweetviz creates visual HTML reports for analyzing a single dataset or comparing two datasets."""
# Note: For demonstration, this is commented out as it requires installation
# import sweetviz as sv
# data = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 75000]})
# report = sv.analyze(data)
# report.show_html("sweetviz_report.html")
print("Sweetviz can generate a visual EDA report (commented for simplicity).")

"""13.6. Automated EDA with AutoViz
AutoViz automatically visualizes datasets with minimal code, producing charts and insights."""
# Note: For demonstration, this is commented out as it requires installation
# from autoviz.AutoViz_Class import AutoViz_Class
# AV = AutoViz_Class()
# data = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 75000]})
# AV.AutoViz("", dfte=data)
print("AutoViz can generate automated visualizations (commented for simplicity).")

"""13.7. Automated EDA with D-Tale
D-Tale provides an interactive web-based interface for exploring pandas DataFrames."""
# Note: For demonstration, this is commented out as it requires installation
# import dtale
# data = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 75000]})
# dtale.show(data)
print("D-Tale provides an interactive EDA interface (commented for simplicity).")

"""13. AutoEDA Exercises
Exercise 1: Dataset Summary
Write a program that creates a small DataFrame and prints its summary statistics."""
import pandas as pd

data = {"product": ["A", "B", "C"], "price": [100, 150, 200], "quantity": [10, 5, 8]}
df = pd.DataFrame(data)
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

"""Exercise 2: Missing Value Check
Write a program that creates a DataFrame with missing values and reports the count of missing values."""
data = {"name": ["Alice", None, "Charlie", "David"], "age": [25, 30, None, 40]}
df = pd.DataFrame(data)
print("Missing Values:")
print(df.isnull().sum())

"""Exercise 3: Correlation Analysis
Write a program that creates a DataFrame with numerical columns and prints its correlation matrix."""
data = {"sales": [100, 120, 140, 160], "advertising": [10, 15, 12, 20], "profit": [50, 60, 70, 80]}
df = pd.DataFrame(data)
print("Correlation Matrix:")
print(df.corr())

"""Exercise 4: Sweetviz Single Dataset Report
Write a program that creates a DataFrame and generates a Sweetviz report (commented for execution)."""
# Note: Commented out to avoid execution errors without Sweetviz installed
# import sweetviz as sv
data = {"age": [25, 30, 35, 40], "salary": [50000, 60000, 75000, 80000]}
df = pd.DataFrame(data)
# report = sv.analyze(df)
# report.show_html("sweetviz_single_report.html")
print("Sweetviz report would be generated for the dataset (commented for simplicity).")
print("Dataset preview:")
print(df.head())

"""Exercise 5: Sweetviz Compare Datasets
Write a program that creates two DataFrames and compares them using Sweetviz (commented for execution)."""
# Note: Commented out to avoid execution errors without Sweetviz installed
# import sweetviz as sv
data1 = {"score": [85, 90, 88], "grade": ["A", "A", "B"]}
data2 = {"score": [78, 92, 80], "grade": ["B", "A", "C"]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
# report = sv.compare([df1, "Dataset 1"], [df2, "Dataset 2"])
# report.show_html("sweetviz_compare_report.html")
print("Sweetviz comparison report would be generated (commented for simplicity).")
print("Dataset 1 preview:")
print(df1.head())
print("Dataset 2 preview:")
print(df2.head())

"""Exercise 6: AutoViz Visualization
Write a program that creates a DataFrame and generates AutoViz visualizations (commented for execution)."""
# Note: Commented out to avoid execution errors without AutoViz installed
# from autoviz.AutoViz_Class import AutoViz_Class
data = {"x": [1, 2, 3, 4], "y": [10, 20, 25, 30]}
df = pd.DataFrame(data)
# AV = AutoViz_Class()
# AV.AutoViz("", dfte=df)
print("AutoViz visualizations would be generated (commented for simplicity).")
print("Dataset preview:")
print(df.head())

"""Exercise 7: D-Tale Interactive EDA
Write a program that creates a DataFrame and opens a D-Tale interface (commented for execution)."""
# Note: Commented out to avoid execution errors without D-Tale installed
# import dtale
data = {"product": ["A", "B", "C"], "sales": [100, 150, 200]}
df = pd.DataFrame(data)
# dtale.show(df)
print("D-Tale interactive interface would be opened (commented for simplicity).")
print("Dataset preview:")
print(df.head())

"""Exercise 8: Unique Values Count
Write a program that creates a DataFrame and counts unique values in a categorical column."""
data = {"city": ["NY", "LA", "NY", "Chicago", "LA"], "population": [8.4, 3.9, 8.4, 2.7, 3.9]}
df = pd.DataFrame(data)
print("Unique Values in 'city':")
print(df["city"].value_counts())

"""Exercise 9: Group By Analysis
Write a program that creates a DataFrame and computes the mean of a numerical column grouped by a categorical column."""
data = {"department": ["HR", "IT", "HR", "IT"], "salary": [50000, 70000, 55000, 80000]}
df = pd.DataFrame(data)
print("Mean Salary by Department:")
print(df.groupby("department")["salary"].mean())

"""Exercise 10: Outlier Detection
Write a program that creates a DataFrame and identifies outliers in a numerical column using IQR."""
data = {"values": [10, 12, 14, 100, 13, 15, 200]}
df = pd.DataFrame(data)
Q1 = df["values"].quantile(0.25)
Q3 = df["values"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["values"] < (Q1 - 1.5 * IQR)) | (df["values"] > (Q3 + 1.5 * IQR))]
print("Outliers:")
print(outliers)

"""Exercise 11: Automated Summary with Custom Function
Write a program that creates a DataFrame and defines a custom function to print a summary (rows, columns, missing values)."""
def custom_summary(df):
    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum()
    }
    return summary

data = {"a": [1, None, 3], "b": ["x", "y", None]}
df = pd.DataFrame(data)
print("Custom Summary:")
print(custom_summary(df))

"""Exercise 12: Categorical Encoding Check
Write a program that creates a DataFrame with a categorical column and prints its encoded values using pandas."""
data = {"grade": ["A", "B", "A", "C", "B"]}
df = pd.DataFrame(data)
df["grade_encoded"] = df["grade"].astype("category").cat.codes
print("Original and Encoded Grades:")
print(df[["grade", "grade_encoded"]])