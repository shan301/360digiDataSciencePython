# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:02:26 2025

@author: Shantanu
"""

"""19. Big Data Processing
This script introduces fundamental big data processing concepts using PySpark and Spark SQL. It covers data loading, transformation, aggregation, and querying for large-scale datasets, using the `sales.csv` and `hr_data.csv` datasets from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pyspark
- Datasets: `sales.csv`, `hr_data.csv` from the `data/` directory
"""

# 19.1. Setup
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, when, max, min
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("BigDataProcessing").getOrCreate()

# Load sample datasets
sales_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("sales_amount", DoubleType(), True),
    StructField("marketing_spend", DoubleType(), True),
    StructField("store_size", IntegerType(), True)
])
sales_df = spark.read.csv('../data/sales.csv', header=True, schema=sales_schema)
hr_df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)

print('Sales Dataset Schema:')
sales_df.printSchema()
print('HR Dataset Schema:')
hr_df.printSchema()

"""19.2. Introduction to Big Data Processing
Big data processing handles large-scale datasets using distributed computing. PySpark enables scalable data processing with:
- DataFrames: Distributed tabular data structures.
- Spark SQL: SQL-like querying for big data.
- RDDs: Resilient Distributed Datasets for low-level operations.
This script focuses on DataFrame operations and Spark SQL for data analysis."""

"""19.3. Data Loading and Inspection
Load and inspect large datasets with Spark DataFrames."""
# Show first 5 rows of sales dataset
print('Sales Dataset Sample:')
sales_df.show(5)

# Show first 5 rows of hr dataset
print('HR Dataset Sample:')
hr_df.show(5)

"""19.4. Data Cleaning
Handle missing values and data types in Spark DataFrames."""
# Drop rows with missing values in sales dataset
sales_df_clean = sales_df.na.drop()

# Fill missing values in hr dataset with median for numerical columns
median_salary = hr_df.approxQuantile("salary", [0.5], 0.25)[0]
hr_df_clean = hr_df.na.fill({"salary": median_salary})

print('Sales Dataset After Cleaning (Count):', sales_df_clean.count())
print('HR Dataset After Cleaning (Sample):')
hr_df_clean.show(5)

"""19.5. Data Transformation
Transform data using filtering, grouping, and column operations."""
# Filter sales with amount > 1000
high_sales = sales_df_clean.filter(col("sales_amount") > 1000)
print('High Sales Transactions:')
high_sales.show(5)

# Add a new column for sales category
sales_df_clean = sales_df_clean.withColumn("sales_category", 
    when(col("sales_amount") > 1000, "High").otherwise("Low"))
print('Sales Dataset with Category:')
sales_df_clean.show(5)

"""19.6. Aggregations
Perform aggregations like sum, average, and count."""
# Group by sales category and compute average marketing spend
sales_agg = sales_df_clean.groupBy("sales_category").agg(
    avg("marketing_spend").alias("avg_marketing_spend"),
    count("transaction_id").alias("transaction_count")
)
print('Sales Aggregations:')
sales_agg.show()

"""19.7. Spark SQL
Use SQL queries for data analysis."""
# Create temporary views for SQL queries
sales_df_clean.createOrReplaceTempView("sales")
hr_df_clean.createOrReplaceTempView("hr")

# SQL query: Average salary by department
avg_salary_query = spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM hr
    GROUP BY department
""")
print('Average Salary by Department:')
avg_salary_query.show()

"""19.8. Joins
Join datasets to combine information."""
# Assuming hr dataset has a department_id and sales has a department_id
hr_df_clean = hr_df_clean.withColumn("department_id", col("department").cast("string"))
sales_df_clean = sales_df_clean.withColumn("department_id", 
    (col("sales_amount") % 3 + 1).cast("string"))  # Mock department_id
joined_df = sales_df_clean.join(hr_df_clean, "department_id", "inner")
print('Joined Dataset Sample:')
joined_df.show(5)

"""19.9. Visualization
Convert Spark DataFrames to Pandas for visualization."""
# Convert sales aggregations to Pandas for plotting
sales_agg_pd = sales_agg.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="sales_category", y="avg_marketing_spend", data=sales_agg_pd)
plt.title('Average Marketing Spend by Sales Category')
plt.show()

"""19.10. Saving Results
Save processed data or models for reuse."""
# Save aggregated sales data as CSV
sales_agg.write.csv('../data/sales_agg_output', mode="overwrite", header=True)
print('Aggregated sales data saved to ../data/sales_agg_output')

"""19. Big Data Processing Exercises"""

"""Exercise 1: Load and Inspect Dataset
Load `sales.csv` as a Spark DataFrame and print its schema."""
def exercise_1():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    print('Sales Dataset Schema:')
    df.printSchema()

exercise_1()

"""Exercise 2: Count Rows
Count the number of rows in `hr_data.csv`."""
def exercise_2():
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    print('Row Count:', df.count())

exercise_2()

"""Exercise 3: Handle Missing Values
Drop rows with missing values in `sales.csv` and print the new count."""
def exercise_3():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    df_clean = df.na.drop()
    print('Row Count After Dropping Missing Values:', df_clean.count())

exercise_3()

"""Exercise 4: Filter Data
Filter `hr_data.csv` for employees with salary > 50000."""
def exercise_4():
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    high_salary_df = df.filter(col("salary") > 50000)
    print('Employees with Salary > 50000:')
    high_salary_df.show(5)

exercise_4()

"""Exercise 5: Group By Aggregation
Group `sales.csv` by store_size and compute total sales_amount."""
def exercise_5():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    sales_by_store = df.groupBy("store_size").agg(sum("sales_amount").alias("total_sales"))
    print('Total Sales by Store Size:')
    sales_by_store.show()

exercise_5()

"""Exercise 6: Spark SQL Query
Use Spark SQL to find the maximum salary in `hr_data.csv`."""
def exercise_6():
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    df.createOrReplaceTempView("hr")
    max_salary = spark.sql("SELECT MAX(salary) as max_salary FROM hr")
    print('Maximum Salary:')
    max_salary.show()

exercise_6()

"""Exercise 7: Add Column
Add a column to `sales.csv` indicating if marketing_spend is above 500."""
def exercise_7():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    df_with_flag = df.withColumn("high_marketing", when(col("marketing_spend") > 500, "Yes").otherwise("No"))
    print('Sales with Marketing Flag:')
    df_with_flag.show(5)

exercise_7()

"""Exercise 8: Join Datasets
Join `sales.csv` and `hr_data.csv` on a mock department_id."""
def exercise_8():
    sales = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    hr = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    sales = sales.withColumn("department_id", (col("sales_amount") % 3 + 1).cast("string"))
    hr = hr.withColumn("department_id", col("department").cast("string"))
    joined = sales.join(hr, "department_id", "inner")
    print('Joined Dataset Sample:')
    joined.show(5)

exerciseFuseSource: https://spark.apache.org/docs/latest/api/python/index.html
exercise_8()

"""Exercise 9: Aggregation with SQL
Use Spark SQL to compute the average sales_amount by store_size in `sales.csv`."""
def exercise_9():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    df.createOrReplaceTempView("sales")
    avg_sales = spark.sql("""
        SELECT store_size, AVG(sales_amount) as avg_sales
        FROM sales
        GROUP BY store_size
    """)
    print('Average Sales by Store Size:')
    avg_sales.show()

 упраж_9()

"""Exercise 10: Save DataFrame
Save the filtered high_salary employees from `hr_data.csv` as a CSV."""
def exercise_10():
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    high_salary_df = df.filter(col("salary") > 50000)
    high_salary_df.write.csv('../data/high_salary_output', mode="overwrite", header=True)
    print('High salary data saved to ../data/high_salary_output')

exercise_10()

"""Exercise 11: Count Distinct Values
Count distinct departments in `hr_data.csv`."""
def exercise_11():
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    distinct_depts = df.select("department").distinct().count()
    print('Distinct Departments:', distinct_depts)

exercise_11()

"""Exercise 12: Window Function
Use a window function to rank employees in `hr_data.csv` by salary within each department."""
def exercise_12():
    from pyspark.sql.window import Window
    from pyspark.sql.functions import rank
    df = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    window_spec = Window.partitionBy("department").orderBy(col("salary").desc())
    ranked_df = df.withColumn("salary_rank", rank().over(window_spec))
    print('Ranked Employees by Department:')
    ranked_df.show(5)

exercise_12()

"""Exercise 13: Visualization
Convert `sales_agg` to Pandas and plot total transactions by sales_category."""
def exercise_13():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    df = df.withColumn("sales_category", when(col("sales_amount") > 1000, "High").otherwise("Low"))
    sales_agg = df.groupBy("sales_category").agg(count("transaction_id").alias("transaction_count"))
    sales_agg_pd = sales_agg.toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="sales_category", y="transaction_count", data=sales_agg_pd)
    plt.title('Total Transactions by Sales Category')
    plt.show()

exercise_13()

"""Exercise 14: Filter and Aggregate
Filter `sales.csv` for sales_amount > 2000 and compute average marketing_spend."""
def exercise_14():
    df = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    filtered_df = df.filter(col("sales_amount") > 2000)
    avg_marketing = filtered_df.agg(avg("marketing_spend").alias("avg_marketing_spend"))
    print('Average Marketing Spend for Sales > 2000:')
    avg_marketing.show()

exercise_14()

"""Exercise 15: Join and Analyze
Join `sales.csv` and `hr_data.csv`, then compute total sales_amount per department."""
def exercise_15():
    sales = spark.read.csv('../data/sales.csv', header=True, inferSchema=True)
    hr = spark.read.csv('../data/hr_data.csv', header=True, inferSchema=True)
    sales = sales.withColumn("department_id", (col("sales_amount") % 3 + 1).cast("string"))
    hr = hr.withColumn("department_id", col("department").cast("string"))
    joined = sales.join(hr, "department_id", "inner")
    sales_by_dept = joined.groupBy("department").agg(sum("sales_amount").alias("total_sales"))
    print('Total Sales by Department:')
    sales_by_dept.show()

exercise_15()

# Stop Spark session
spark.stop()