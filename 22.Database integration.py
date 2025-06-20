# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:04:04 2025

@author: Shantanu
"""

"""22. Database Integration
This script introduces fundamental database integration concepts using SQLite and pandas. It covers database creation, data insertion, querying, updating, and joining, using the `sales.csv` and `hr_data.csv` datasets from the `data/` directory stored in an SQLite database.

Prerequisites:
- Python 3.9+
- Libraries: sqlite3, pandas
- Datasets: `sales.csv`, `hr_data.csv` from the `data/` directory
"""

# 22.1. Setup
import sqlite3
import pandas as pd
import os

# Create output directory
output_dir = '../data/output'
os.makedirs(output_dir, exist_ok=True)

# Initialize SQLite database
db_path = '../data/sample.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Load sample datasets
sales_df = pd.read_csv('../data/sales.csv')
hr_df = pd.read_csv('../data/hr_data.csv')
print('Sales Dataset Head:')
print(sales_df.head())
print('\nHR Dataset Head:')
print(hr_df.head())

"""22.2. Introduction to Database Integration
Database integration enables Python to interact with databases for storing and querying data. Key concepts:
- SQLite: Lightweight, file-based database.
- CRUD Operations: Create, Read, Update, Delete.
- SQL Queries: Structured Query Language for data manipulation.
- Pandas Integration: Combining pandas with SQL for data analysis.
This script uses SQLite to manage data from `sales.csv` and `hr_data.csv`."""

"""22.3. Create Database and Tables
Create tables in SQLite to store dataset contents."""
# Create sales table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        transaction_id TEXT,
        sales_amount REAL,
        marketing_spend REAL,
        store_size INTEGER
    )
''')

# Create hr table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS hr (
        employee_id INTEGER,
        department TEXT,
        salary REAL
    )
''')
conn.commit()

# Insert data into tables
sales_df.to_sql('sales', conn, if_exists='replace', index=False)
hr_df.to_sql('hr', conn, if_exists='replace', index=False)
print('Tables created and data inserted.')

"""22.4. Basic Querying
Retrieve data using SQL SELECT statements."""
# Fetch first 5 rows from sales table
cursor.execute('SELECT * FROM sales LIMIT 5')
sales_rows = cursor.fetchall()
print('First 5 Sales Rows:')
for row in sales_rows:
    print(row)

"""22.5. Filtering Data
Use WHERE clause to filter data."""
# Fetch sales with amount > 1000
cursor.execute('SELECT * FROM sales WHERE sales_amount > 1000 LIMIT 5')
high_sales = cursor.fetchall()
print('\nHigh Sales (>1000):')
for row in high_sales:
    print(row)

"""22.6. Aggregations
Perform aggregations like SUM, AVG, and COUNT."""
# Compute average salary by department
cursor.execute('SELECT department, AVG(salary) as avg_salary FROM hr GROUP BY department')
avg_salary = cursor.fetchall()
print('\nAverage Salary by Department:')
for row in avg_salary:
    print(row)

"""22.7. Updating Data
Modify records using UPDATE statements."""
# Update sales_amount by increasing it 10% for store_size > 1000
cursor.execute('UPDATE sales SET sales_amount = sales_amount * 1.1 WHERE store_size > 1000')
conn.commit()
print('Updated sales_amount for store_size > 1000.')

"""22.8. Joining Tables
Combine data from multiple tables."""
# Mock a department_id for joining
sales_df['department_id'] = (sales_df.index % 3 + 1).astype(str)
hr_df['department_id'] = hr_df['department'].astype(str)
sales_df.to_sql('sales', conn, if_exists='replace', index=False)
hr_df.to_sql('hr', conn, if_exists='replace', index=False)

# Join sales and hr tables
cursor.execute('''
    SELECT s.transaction_id, s.sales_amount, h.department, h.salary
    FROM sales s
    INNER JOIN hr h ON s.department_id = h.department_id
    LIMIT 5
''')
joined_data = cursor.fetchall()
print('\nJoined Sales and HR Data:')
for row in joined_data:
    print(row)

"""22.9. Pandas with SQL
Use pandas to execute SQL queries and process results."""
# Query sales data with pandas
sales_query = pd.read_sql_query('SELECT * FROM sales WHERE sales_amount > 1000', conn)
output_path = os.path.join(output_dir, 'high_sales.csv')
sales_query.to_csv(output_path, index=False)
print(f'\nSaved high sales data to {output_path}')

# Plot average salary by department
avg_salary_df = pd.read_sql_query('SELECT department, AVG(salary) as avg_salary FROM hr GROUP BY department', conn)
plt.figure(figsize=(10, 6))
sns.barplot(x='department', y='avg_salary', data=avg_salary_df)
plt.title('Average Salary by Department')
plt.xticks(rotation=45)
plt.show()

"""22.10. Database Integration Exercises"""

"""Exercise 1: Create Database
Create a new SQLite database named `test.db` and print the connection status."""
def exercise_1():
    test_conn = sqlite3.connect('../data/test.db')
    print('Connected to test.db successfully')
    test_conn.close()

exercise_1()

"""Exercise 2: Create Table
Create a table named `employees` with columns for id, name, and salary."""
def exercise_2():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            salary REAL
        )
    ''')
    conn.commit()
    print('Created employees table')
    conn.close()

exercise_2()

"""Exercise 3: Insert Data
Insert a sample employee record into the `employees` table."""
def exercise_3():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO employees (id, name, salary) VALUES (?, ?, ?)', (1, 'John Doe', 50000))
    conn.commit()
    print('Inserted sample employee')
    conn.close()

exercise_3()

"""Exercise 4: Query Table
Fetch all records from the `sales` table and print the first 3."""
def exercise_4():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sales LIMIT 3')
    rows = cursor.fetchall()
    print('First 3 Sales Records:')
    for row in rows:
        print(row)
    conn.close()

exercise_4()

"""Exercise 5: Filter Data
Fetch records from `hr` where salary > 60000."""
def exercise_5():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM hr WHERE salary > 60000 LIMIT 5')
    rows = cursor.fetchall()
    print('Employees with Salary > 60000:')
    for row in rows:
        print(row)
    conn.close()

exercise_5()

"""Exercise 6: Aggregate Data
Compute the total sales_amount from the `sales` table."""
def exercise_6():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT SUM(sales_amount) as total_sales FROM sales')
    total_sales = cursor.fetchone()[0]
    print(f'Total Sales Amount: {total_sales:.2f}')
    conn.close()

exercise_6()

"""Exercise 7: Update Data
Update `sales` to set marketing_spend to 0 where sales_amount < 500."""
def exercise_7():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('UPDATE sales SET marketing_spend = 0 WHERE sales_amount < 500')
    conn.commit()
    print('Updated marketing_spend for low sales')
    conn.close()

exercise_7()

"""Exercise 8: Join Tables
Join `sales` and `hr` on department_id and fetch 5 records."""
def exercise_8():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.transaction_id, s.sales_amount, h.department, h.salary
        FROM sales s
        INNER JOIN hr h ON s.department_id = h.department_id
        LIMIT 5
    ''')
    rows = cursor.fetchall()
    print('Joined Sales and HR Records:')
    for row in rows:
        print(row)
    conn.close()

exercise_8()

"""Exercise 9: Pandas Query
Use pandas to query `sales` for store_size > 1000 and save to CSV."""
def exercise_9():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM sales WHERE store_size > 1000', conn)
    output_path = os.path.join('../data/output', 'large_stores.csv')
    df.to_csv(output_path, index=False)
    print(f'Saved large stores data to {output_path}')
    conn.close()

exercise_9()

"""Exercise 10: Count Records
Count the number of employees per department in `hr`."""
def exercise_10():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT department, COUNT(*) as emp_count FROM hr GROUP BY department')
    rows = cursor.fetchall()
    print('Employee Count by Department:')
    for row in rows:
        print(row)
    conn.close()

exercise_10()

"""Exercise 11: Delete Records
Delete records from `sales` where sales_amount < 200."""
def exercise_11():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sales WHERE sales_amount < 200')
    conn.commit()
    print('Deleted sales records with amount < 200')
    conn.close()

exercise_11()

"""Exercise 12: Create Index
Create an index on the `salary` column in `hr` for faster queries."""
def exercise_12():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_salary ON hr(salary)')
    conn.commit()
    print('Created index on salary column')
    conn.close()

exercise_12()

"""Exercise 13: Pandas Aggregation
Use pandas to compute the maximum sales_amount by store_size and save to CSV."""
def exercise_13():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT store_size, sales_amount FROM sales', conn)
    max_sales = df.groupby('store_size')['sales_amount'].max().reset_index()
    output_path = os.path.join('../data/output', 'max_sales_by_store.csv')
    max_sales.to_csv(output_path, index=False)
    print(f'Saved max sales to {output_path}')
    conn.close()

exercise_13()

"""Exercise 14: Update with Join
Update `sales` to increase sales_amount by 5% where department_id matches high-salary departments (>60000)."""
def exercise_14():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE sales
        SET sales_amount = sales_amount * 1.05
        WHERE department_id IN (SELECT department_id FROM hr WHERE salary > 60000)
    ''')
    conn.commit()
    print('Updated sales_amount for high-salary departments')
    conn.close()

exercise_14()

"""Exercise 15: Visualize Data
Query average sales_amount by store_size and plot a bar chart."""
def exercise_15():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT store_size, AVG(sales_amount) as avg_sales FROM sales GROUP BY store_size', conn)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='store_size', y='avg_sales', data=df)
    plt.title('Average Sales Amount by Store Size')
    plt.show()
    conn.close()

exercise_15()

# Close database connection
conn.close()