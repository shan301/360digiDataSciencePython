# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:19:12 2025
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""4. File Handling and Exception Handling
File handling enables reading, writing, and managing files, while exception handling ensures robust code by managing errors. These are critical for data analysis tasks like processing datasets.

4.1. Working Directory
Manage the current working directory to organize file operations."""
import os
import pathlib

print(f"Current Directory: {os.getcwd()}")
# os.chdir('D:\\Data')  # Commented to avoid changing directory in demo
print(f"Using pathlib: {pathlib.Path.cwd()}")  # Modern alternative

"""4.2. Opening and Reading Files
Open files in various modes and read their contents."""
with open("demo1.txt", "w") as f:
    f.write("Sample data for demo1.txt")
with open("demo1.txt", "r") as file:
    content = file.read()
print(f"File Content: {content}")  # Output: File Content: Sample data for demo1.txt

"""4.3. Writing and Appending Files
Write new content or append to existing files."""
with open("demo1.txt", "w") as f:
    f.write("Overwritten content")
with open("demo1.txt", "a") as f:
    f.write("\nAppended content")
with open("demo1.txt", "r") as f:
    print(f"Updated Content: {f.read()}")  # Output: Overwritten content\nAppended content

"""4.4. Creating Files
Create new files using 'x' mode, which fails if the file exists."""
try:
    with open("demo2.txt", "x") as f:
        f.write("Newly created file")
    print("File created successfully")
except FileExistsError:
    print("File already exists")

"""4.5. Deleting Files
Safely delete files after checking existence."""
if os.path.exists("demo2.txt"):
    os.remove("demo2.txt")
    print("File deleted")
else:
    print("File does not exist")

"""4.6. Directory Handling
Create, list, and remove directories for organizing data files."""
os.makedirs("data_folder", exist_ok=True)
print(f"Files in data_folder: {os.listdir('data_folder')}")
# os.rmdir("data_folder")  # Commented to keep folder for exercises
path = pathlib.Path("data_folder")
print(f"Is directory: {path.is_dir()}")  # Output: Is directory: True

"""4.7. Context Managers for Multiple Files
Handle multiple files simultaneously using contextlib."""
from contextlib import ExitStack

with ExitStack() as stack:
    file1 = stack.enter_context(open("demo1.txt", "r"))
    file2 = stack.enter_context(open("demo3.txt", "w"))
    file2.write(file1.read())
print("Content copied from demo1.txt to demo3.txt")

"""4.8. Basic Exception Handling
Use try-except to handle runtime errors."""
try:
    x = int(input("Enter a number: "))
    print(f"Number: {x}")
except ValueError:
    print("Please enter a valid integer")

"""4.9. Specific Exception Handling
Handle different types of exceptions separately."""
try:
    x = int(input("Enter numerator: "))
    y = int(input("Enter denominator: "))
    result = x / y
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Cannot divide by zero")
except ValueError:
    print("Please enter valid integers")

"""4.10. Else and Finally Blocks
Use else for code that runs if no exception occurs, and finally for cleanup."""
try:
    x = int(input("Enter a number: "))
    y = 10 / x
except ZeroDivisionError:
    print("Cannot divide by zero")
else:
    print(f"Result: {y}")
finally:
    print("Operation completed")

"""4.11. Raising Custom Exceptions
Trigger user-defined errors for specific conditions."""
age = int(input("Enter your age: "))
if age < 18:
    raise ValueError("You must be 18 or older")
print("You are eligible")

"""4.12. Custom Exception Classes
Define custom exceptions for specific error scenarios."""
class DataValidationError(Exception):
    pass

def validate_data(value):
    if not isinstance(value, (int, float)) or value < 0:
        raise DataValidationError("Data must be a non-negative number")
    return value

try:
    validate_data(-5)
except DataValidationError as e:
    print(f"Error: {e}")

"""4. File Handling and Exception Handling Exercises
Exercise 1: Create and Write to a File
Write a program to create 'data.txt' and write 'This is a data file' to it."""
with open("data.txt", "w") as f:
    f.write("This is a data file")
print("File created and written")

"""Exercise 2: Read Entire File
Read and print the contents of 'data.txt'."""
try:
    with open("data.txt", "r") as f:
        content = f.read()
    print(f"Content: {content}")
except FileNotFoundError:
    print("Error: File not found")

"""Exercise 3: Append to File
Append 'New data added' to 'data.txt'."""
with open("data.txt", "a") as f:
    f.write("\nNew data added")
print("Text appended")

"""Exercise 4: Read Line by Line
Read 'data.txt' line by line and print each line."""
try:
    with open("data.txt", "r") as f:
        for line in f:
            print(line.strip())
except FileNotFoundError:
    print("Error: File not found")

"""Exercise 5: Count Words in File
Count the number of words in 'data.txt'."""
try:
    with open("data.txt", "r") as f:
        words = f.read().split()
    print(f"Word count: {len(words)}")
except FileNotFoundError:
    print("Error: File not found")

"""Exercise 6: Create Directory
Create a directory 'datasets' if it doesn't exist."""
os.makedirs("datasets", exist_ok=True)
print("Directory 'datasets' created or already exists")

"""Exercise 7: List Directory Contents
List all files in the 'datasets' directory."""
print(f"Files in 'datasets': {os.listdir('datasets')}")

"""Exercise 8: Copy File Content
Copy the content of 'data.txt' to a new file 'data_copy.txt'."""
try:
    with open("data.txt", "r") as src, open("data_copy.txt", "w") as dst:
        dst.write(src.read())
    print("Content copied to data_copy.txt")
except FileNotFoundError:
    print("Error: Source file not found")

"""Exercise 9: Handle Division Errors
Ask for two numbers and divide them, handling zero division and invalid inputs."""
try:
    a = int(input("Enter numerator: "))
    b = int(input("Enter denominator: "))
    result = a / b
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
except ValueError:
    print("Error: Please enter valid numbers")

"""Exercise 10: Handle File Not Found
Try to read a non-existent file and handle the FileNotFoundError."""
try:
    with open("nonexistent.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print("Error: File does not exist")

"""Exercise 11: Use Finally Block
Modify Exercise 9 to print 'Calculation done' using finally."""
try:
    a = int(input("Enter numerator: "))
    b = int(input("Enter denominator: "))
    result = a / b
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
except ValueError:
    print("Error: Please enter valid numbers")
finally:
    print("Calculation done")

"""Exercise 12: Raise Custom Exception
Ask for a dataset size; raise an exception if it's less than 10."""
size = int(input("Enter dataset size: "))
if size < 10:
    raise ValueError("Dataset size must be at least 10")
print("Dataset size is valid")

"""Exercise 13: Custom Exception for Data Validation
Define a custom exception and use it to validate a positive number."""
class InvalidDataError(Exception):
    pass

def check_positive(value):
    if value <= 0:
        raise InvalidDataError("Value must be positive")
    return value

try:
    value = float(input("Enter a positive number: "))
    check_positive(value)
    print(f"Valid value: {value}")
except InvalidDataError as e:
    print(f"Error: {e}")
except ValueError:
    print("Error: Please enter a valid number")

"""Exercise 14: Write CSV File
Write a list of dictionaries to a CSV file 'records.csv'."""
import csv

data = [{"name": "Alice", "score": 85}, {"name": "Bob", "score": 90}]
with open("records.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "score"])
    writer.writeheader()
    writer.writerows(data)
print("CSV file written")

"""Exercise 15: Read CSV File
Read 'records.csv' and print its contents."""
try:
    with open("records.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row)
except FileNotFoundError:
    print("Error: CSV file not found")

