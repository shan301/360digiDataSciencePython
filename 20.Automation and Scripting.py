# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:54:57 2025

@author: Shantanu
"""

"""20. Automation and Scripting
This script introduces fundamental automation and scripting concepts using Python. It covers file handling, process automation, scheduling tasks, and web scraping, with practical examples for managing files in the `data/` directory and automating web interactions.

Prerequisites:
- Python 3.9+
- Libraries: os, shutil, subprocess, schedule, selenium, pandas, time
- Tools: ChromeDriver for Selenium (ensure installed and in PATH)
- Directory: `data/` for file operations
"""

# 20.1. Setup
import os
import shutil
import subprocess
import schedule
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Create a sample output directory
output_dir = '../data/output'
os.makedirs(output_dir, exist_ok=True)

"""20.2. Introduction to Automation and Scripting
Automation and scripting streamline repetitive tasks using Python. Key concepts:
- File Handling: Managing files and directories (e.g., create, move, delete).
- Process Automation: Running system commands or scripts.
- Task Scheduling: Automating recurring tasks.
- Web Scraping: Extracting data from websites.
This script demonstrates these concepts with practical examples."""

"""20.3. File Handling
Read, write, and organize files in the `data/` directory."""
# List files in data directory
data_dir = '../data'
files = os.listdir(data_dir)
print('Files in data directory:', files)

# Copy a file (e.g., sales.csv) to output directory
source_file = os.path.join(data_dir, 'sales.csv')
dest_file = os.path.join(output_dir, 'sales_copy.csv')
shutil.copy(source_file, dest_file)
print(f'Copied {source_file} to {dest_file}')

"""20.4. File Processing
Process a CSV file using pandas and save results."""
# Read and process sales.csv
sales_df = pd.read_csv('../data/sales.csv')
sales_summary = sales_df.groupby('store_size').agg({'sales_amount': 'sum'}).reset_index()
output_summary = os.path.join(output_dir, 'sales_summary.csv')
sales_summary.to_csv(output_summary, index=False)
print(f'Saved sales summary to {output_summary}')

"""20.5. Process Automation
Run system commands or scripts using subprocess."""
# Run a system command (e.g., list directory contents)
result = subprocess.run(['ls', data_dir], capture_output=True, text=True)
print('Directory contents via subprocess:\n', result.stdout)

"""20.6. Task Scheduling
Schedule recurring tasks using the schedule library."""
def backup_data():
    """Backup data directory to a zip file."""
    backup_file = os.path.join(output_dir, f'backup_{time.strftime("%Y%m%d_%H%M%S")}.zip')
    shutil.make_archive(backup_file[:-4], 'zip', data_dir)
    print(f'Created backup: {backup_file}')

# Schedule backup every 10 seconds (for demo purposes)
schedule.every(10).seconds.do(backup_data)

# Run scheduler for 30 seconds
end_time = time.time() + 30
while time.time() < end_time:
    schedule.run_pending()
    time.sleep(1)

"""20.7. Web Scraping
Automate web interactions using Selenium."""
# Setup Selenium with headless Chrome
chrome_options = Options()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)

# Scrape a sample website (e.g., a public page)
url = 'https://example.com'
driver.get(url)
title = driver.title
content = driver.find_element(By.TAG_NAME, 'h1').text
print(f'Webpage Title: {title}')
print(f'Webpage H1 Content: {content}')
driver.quit()

"""20.8. Automation and Scripting Exercises"""

"""Exercise 1: List Directory Contents
List all files in the `data/` directory."""
def exercise_1():
    files = os.listdir('../data')
    print('Files in data directory:', files)

exercise_1()

"""Exercise 2: Create Directory
Create a new directory named `temp` in the `data/` directory."""
def exercise_2():
    temp_dir = '../data/temp'
    os.makedirs(temp_dir, exist_ok=True)
    print(f'Created directory: {temp_dir}')

exercise_2()

"""Exercise 3: Copy File
Copy `hr_data.csv` to the `output/` directory as `hr_data_backup.csv`."""
def exercise_3():
    src = '../data/hr_data.csv'
    dst = '../data/output/hr_data_backup.csv'
    shutil.copy(src, dst)
    print(f'Copied {src} to {dst}')

exercise_3()

"""Exercise 4: Delete File
Delete the `hr_data_backup.csv` file from the `output/` directory."""
def exercise_4():
    file_path = '../data/output/hr_data_backup.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Deleted {file_path}')
    else:
        print(f'File {file_path} does not exist')

exercise_4()

"""Exercise 5: Read and Summarize CSV
Read `sales.csv` and compute the average sales_amount."""
def exercise_5():
    df = pd.read_csv('../data/sales.csv')
    avg_sales = df['sales_amount'].mean()
    print(f'Average Sales Amount: {avg_sales:.2f}')

exercise_5()

"""Exercise 6: Save Processed Data
Group `hr_data.csv` by department and save the average salary to `output/hr_avg_salary.csv`."""
def exercise_6():
    df = pd.read_csv('../data/hr_data.csv')
    avg_salary = df.groupby('department')['salary'].mean().reset_index()
    output_path = '../data/output/hr_avg_salary.csv'
    avg_salary.to_csv(output_path, index=False)
    print(f'Saved average salary to {output_path}')

exercise_6()

"""Exercise 7: Run System Command
Use subprocess to count the number of files in the `data/` directory."""
def exercise_7():
    result = subprocess.run(['ls', '../data', '|', 'wc', '-l'], capture_output=True, text=True, shell=True)
    print('Number of files in data directory:', result.stdout.strip())

exercise_7()

"""Exercise 8: Create Backup
Create a zip backup of the `output/` directory."""
def exercise_8():
    backup_file = os.path.join('../data/output', f'output_backup_{time.strftime("%Y%m%d_%H%M%S")}.zip')
    shutil.make_archive(backup_file[:-4], 'zip', '../data/output')
    print(f'Created backup: {backup_file}')

exercise_8()

"""Exercise 9: Schedule Task
Schedule a task to print the current time every 5 seconds for 20 seconds."""
def exercise_9():
    def print_time():
        print(f'Current time: {time.strftime("%H:%M:%S")}')
    
    schedule.every(5).seconds.do(print_time)
    end_time = time.time() + 20
    while time.time() < end_time:
        schedule.run_pending()
        time.sleep(1)
    schedule.clear()

exercise_9()

"""Exercise 10: Web Scraping Title
Scrape the title of a webpage (e.g., https://example.com) using Selenium."""
def exercise_10():
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://example.com')
    title = driver.title
    print(f'Webpage Title: {title}')
    driver.quit()

exercise_10()

"""Exercise 11: Web Scraping Links
Scrape all hyperlinks from a webpage and print their URLs."""
def exercise_11():
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://example.com')
    links = driver.find_elements(By.TAG_NAME, 'a')
    urls = [link.get_attribute('href') for link in links]
    print('Hyperlinks:', urls)
    driver.quit()

exercise_11()

"""Exercise 12: File Renaming
Rename all CSV files in the `output/` directory to append `_processed` to their names."""
def exercise_12():
    for file in os.listdir('../data/output'):
        if file.endswith('.csv'):
            old_path = os.path.join('../data/output', file)
            new_path = os.path.join('../data/output', file.replace('.csv', '_processed.csv'))
            os.rename(old_path, new_path)
            print(f'Renamed {old_path} to {new_path}')

exercise_12()

"""Exercise 13: Monitor Directory
Check if a new file is added to the `output/` directory and print a message."""
def exercise_13():
    initial_files = set(os.listdir('../data/output'))
    time.sleep(5)  # Wait for potential changes
    current_files = set(os.listdir('../data/output'))
    new_files = current_files - initial_files
    if new_files:
        print('New files detected:', new_files)
    else:
        print('No new files detected')

exercise_13()

"""Exercise 14: Automate CSV Merge
Merge `sales.csv` and `hr_data.csv` into a single CSV in the `output/` directory."""
def exercise_14():
    sales = pd.read_csv('../data/sales.csv')
    hr = pd.read_csv('../data/hr_data.csv')
    # Mock a common column for merging
    sales['department_id'] = (sales['sales_amount'] % 3 + 1).astype(str)
    hr['department_id'] = hr['department'].astype(str)
    merged_df = pd.merge(sales, hr, on='department_id', how='inner')
    output_path = '../data/output/merged_data.csv'
    merged_df.to_csv(output_path, index=False)
    print(f'Merged data saved to {output_path}')

exercise_14()

"""Exercise 15: Automate Report Generation
Generate a summary report of `sales.csv` (e.g., total sales by store_size) and save as a text file."""
def exercise_15():
    df = pd.read_csv('../data/sales.csv')
    report = df.groupby('store_size')['sales_amount'].sum().to_string()
    output_path = '../data/output/sales_report.txt'
    with open(output_path, 'w') as f:
        f.write('Sales Report by Store Size\n')
        f.write(report)
    print(f'Report saved to {output_path}')

exercise_15()