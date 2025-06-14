# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 10:08:44 2025

@author: Shantanu
"""
"""
DataFrames in Python are a core feature of the pandas library, widely used for data manipulation and analysis.
A DataFrame is a 2-dimensional labeled data structure with rows and columns, similar to an Excel spreadsheet or a SQL table.

Importing pandas:"""
import pandas as pd

"""Creating a DataFrame:
From a dictionary:"""
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)

"""From a list of lists:"""
data = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Los Angeles'],
    ['Charlie', 35, 'Chicago']
]
columns = ['Name', 'Age', 'City']
df = pd.DataFrame(data, columns=columns)
print(df)

"""From a CSV file:"""
df = pd.read_csv('data.csv')
print(df)





"""Basic Operations:
Display the first few rows:"""
print(df.head())

"""Display DataFrame info:"""
print(df.info())

"""Access columns:"""
print(df['Age'])  # Select one column
print(df[['Name', 'City']])  # Select multiple columns

"""Add a new column:"""
df['Salary'] = [50000, 60000, 70000]
print(df)

"""Filter rows:"""
filtered_df = df[df['Age'] > 28]
print(filtered_df)

"""Group and aggregate:"""
grouped = df.groupby('City')['Age'].mean()
print(grouped)





"""Save/Export Data:
Save to CSV:"""
df.to_csv('output.csv', index=False)

""""Save to Excel:"""
df.to_excel('output.xlsx', index=False)





"""Advanced Features:
Merge and join DataFrames:"""
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Age': [25, 30]})
merged_df = pd.merge(df1, df2, on='ID')
print(merged_df)

"""Pivot tables:"""
pivot_df = df.pivot_table(values='Age', index='City', aggfunc='mean')
print(pivot_df)





































