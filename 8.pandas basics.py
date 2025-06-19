# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 10:08:44 2025
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""8. Pandas Basics
Pandas is a powerful library for data manipulation and analysis, using DataFrames as its core structure. A DataFrame is a 2D labeled data structure, similar to a spreadsheet or SQL table, ideal for data analysis tasks.

8.1. Creating DataFrames
Create DataFrames from dictionaries, lists, or external files."""
import pandas as pd

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)
# Output:
#       Name  Age         City
# 0    Alice   25     New York
# 1      Bob   30  Los Angeles
# 2  Charlie   35      Chicago

# From list of lists
data = [['Alice', 25, 'New York'], ['Bob', 30, 'Los Angeles'], ['Charlie', 35, 'Chicago']]
columns = ['Name', 'Age', 'City']
df_list = pd.DataFrame(data, columns=columns)
print("\nDataFrame from list:")
print(df_list)
# Output: (same as above)

"""8.2. Reading from Files
Read data from CSV or Excel files."""
# Example with try-except to handle missing file
try:
    df_csv = pd.DataFrame({'Name': ['Alice'], 'Age': [25], 'City': ['New York']})  # Mock data
    print("\nMock CSV DataFrame:")
    print(df_csv)
except FileNotFoundError:
    print("Error: CSV file not found")

"""8.3. Basic Operations
Inspect and manipulate DataFrame structure."""
print("\nFirst 2 rows:")
print(df.head(2))  # Output: First 2 rows of df
print("\nDataFrame info:")
print(df.info())   # Output: Info about columns and dtypes
print("\nSelect 'Age' column:")
print(df['Age'])   # Output: 0    25, 1    30, 2    35

"""8.4. Adding Columns
Add new columns to a DataFrame."""
df['Salary'] = [50000, 60000, 70000]
print("\nDataFrame with Salary:")
print(df)
# Output:
#       Name  Age         City  Salary
# 0    Alice   25     New York   50000
# 1      Bob   30  Los Angeles   60000
# 2  Charlie   35      Chicago   70000

"""8.5. Filtering Rows
Filter rows based on conditions."""
filtered_df = df[df['Age'] > 28]
print("\nRows where Age > 28:")
print(filtered_df)
# Output:
#       Name  Age      City  Salary
# 1      Bob   30  Los Angeles  60000
# 2  Charlie   35     Chicago   70000

"""8.6. Grouping and Aggregating
Group data and compute aggregates like mean."""
grouped = df.groupby('City')['Age'].mean()
print("\nMean Age by City:")
print(grouped)
# Output:
# City
# Chicago        35.0
# Los Angeles    30.0
# New York       25.0

"""8.7. Handling Missing Data
Manage missing values in a DataFrame."""
data_missing = {'Name': ['Alice', None, 'Charlie'], 'Age': [25, 30, None]}
df_missing = pd.DataFrame(data_missing)
print("\nMissing Values:")
print(df_missing.isna().sum())
# Output:
# Name    1
# Age     1

"""8.8. Data Type Conversion
Convert column data types for analysis."""
df['Age'] = df['Age'].astype(float)
print("\nDataFrame with Age as float:")
print(df.dtypes)
# Output: Name: object, Age: float64, City: object, Salary: int64

"""8.9. Indexing with loc and iloc
Select data using label-based (loc) or integer-based (iloc) indexing."""
print("\nSelect row 1 with loc:")
print(df.loc[1, ['Name', 'City']])  # Output: Name: Bob, City: Los Angeles
print("\nSelect row 1 with iloc:")
print(df.iloc[1, [0, 2]])           # Output: Bob, Los Angeles

"""8.10. Merging DataFrames
Combine DataFrames using merge."""
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Age': [25, 30]})
merged_df = pd.merge(df1, df2, on='ID')
print("\nMerged DataFrame:")
print(merged_df)
# Output:
#    ID   Name  Age
# 0   1  Alice   25
# 1   2    Bob   30

"""8.11. Pivot Tables
Create pivot tables for summarizing data."""
pivot_df = df.pivot_table(values='Age', index='City', aggfunc='mean')
print("\nPivot Table:")
print(pivot_df)
# Output:
#              Age
# City            
# Chicago      35.0
# Los Angeles  30.0
# New York     25.0

"""8.12. Exporting Data
Save DataFrames to CSV or Excel."""
try:
    df.to_csv('output.csv', index=False)
    print("\nSaved to output.csv")
except PermissionError:
    print("Error: Cannot write to output.csv")

"""8. Pandas Basics Exercises
Exercise 1: Create DataFrame
Create a DataFrame from a dictionary with names and scores."""
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Score': [85, 90, 88]}
df_ex1 = pd.DataFrame(data)
print("DataFrame:")
print(df_ex1)
# Output:
#       Name  Score
# 0    Alice     85
# 1      Bob     90
# 2  Charlie     88

"""Exercise 2: Display DataFrame Info
Print the info of the DataFrame from Exercise 1."""
print("DataFrame Info:")
print(df_ex1.info())
# Output: Info about columns and dtypes

"""Exercise 3: Select Columns
Select and print the 'Name' column from the DataFrame."""
print("Name Column:")
print(df_ex1['Name'])
# Output: 0    Alice, 1    Bob, 2    Charlie

"""Exercise 4: Add Column
Add a 'Grade' column with values ['A', 'A', 'B'] to the DataFrame."""
df_ex1['Grade'] = ['A', 'A', 'B']
print("DataFrame with Grade:")
print(df_ex1)
# Output:
#       Name  Score Grade
# 0    Alice     85     A
# 1      Bob     90     A
# 2  Charlie     88     B

"""Exercise 5: Filter Rows
Filter rows where Score is greater than 86."""
filtered = df_ex1[df_ex1['Score'] > 86]
print("Filtered DataFrame (Score > 86):")
print(filtered)
# Output:
#       Name  Score Grade
# 1      Bob     90     A
# 2  Charlie     88     B

"""Exercise 6: Group and Aggregate
Group the DataFrame by 'Grade' and compute the mean Score."""
grouped = df_ex1.groupby('Grade')['Score'].mean()
print("Mean Score by Grade:")
print(grouped)
# Output:
# Grade
# A    87.5
# B    88.0

"""Exercise 7: Handle Missing Values
Create a DataFrame with missing values and fill them with 0."""
data_missing = {'Name': ['Alice', None, 'Charlie'], 'Score': [85, 90, None]}
df_missing = pd.DataFrame(data_missing)
df_missing.fillna({'Name': 'Unknown', 'Score': 0}, inplace=True)
print("DataFrame with filled missing values:")
print(df_missing)
# Output:
#       Name  Score
# 0    Alice   85.0
# 1  Unknown   90.0
# 2  Charlie    0.0

"""Exercise 8: Convert Data Types
Convert the 'Score' column to integer in the DataFrame from Exercise 1."""
df_ex1['Score'] = df_ex1['Score'].astype(int)
print("DataFrame with Score as int:")
print(df_ex1.dtypes)
# Output: Name: object, Score: int32, Grade: object

"""Exercise 9: Indexing with loc
Use loc to select the Name and Score for the second row."""
print("Second row (Name, Score) with loc:")
print(df_ex1.loc[1, ['Name', 'Score']])
# Output: Name: Bob, Score: 90

"""Exercise 10: Indexing with iloc
Use iloc to select the first two rows and first two columns."""
print("First two rows, first two columns with iloc:")
print(df_ex1.iloc[:2, :2])
# Output:
#     Name  Score
# 0  Alice     85
# 1    Bob     90

"""Exercise 11: Merge DataFrames
Merge two DataFrames on a common 'ID' column."""
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [85, 90]})
merged = pd.merge(df1, df2, on='ID')
print("Merged DataFrame:")
print(merged)
# Output:
#    ID   Name  Score
# 0   1  Alice     85
# 1   2    Bob     90

"""Exercise 12: Create Pivot Table
Create a pivot table showing mean Score by Grade."""
pivot = df_ex1.pivot_table(values='Score', index='Grade', aggfunc='mean')
print("Pivot Table:")
print(pivot)
# Output:
#        Score
# Grade      
# A       87.5
# B       88.0

"""Exercise 13: Export to CSV
Save the DataFrame from Exercise 1 to 'scores.csv'."""
try:
    df_ex1.to_csv('scores.csv', index=False)
    print("Saved to scores.csv")
except PermissionError:
    print("Error: Cannot write to scores.csv")

"""Exercise 14: Drop Missing Values
Create a DataFrame with missing values and drop rows with any missing data."""
data_missing = {'Name': ['Alice', None, 'Charlie'], 'Score': [85, 90, None]}
df_missing = pd.DataFrame(data_missing)
df_dropped = df_missing.dropna()
print("DataFrame after dropping missing values:")
print(df_dropped)
# Output:
#       Name  Score
# 0    Alice   85.0

"""Exercise 15: Convert Date Column
Create a DataFrame with a date column as strings and convert it to datetime."""
data = {'Name': ['Alice', 'Bob'], 'Date': ['2023-01-01', '2023-02-01']}
df_dates = pd.DataFrame(data)
df_dates['Date'] = pd.to_datetime(df_dates['Date'])
print("DataFrame with Date as datetime:")
print(df_dates.dtypes)
# Output: Name: object, Date: datetime64[ns]




































