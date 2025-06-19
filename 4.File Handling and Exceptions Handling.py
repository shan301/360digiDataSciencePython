# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:19:12 2025

@author: Shantanu
"""

"""File Handling in Python
Python provides built-in functions to work with files. The script demonstrates:

1. Getting & Changing the Working Directory"""
import os
os.getcwd()  # Get the current working directory
os.chdir('D:\\Data')  # Change directory to 'D:\\Data'

"""2. Opening Files"""
f = open("demo1.txt")  # Default mode is "r" (read)
"""The open() function opens a file.
Modes:
"r" - Read (default mode).
"w" - Write (overwrites existing content).
"a" - Append (adds content without deleting previous content).
"x" - Create a new file."""

"""3. Reading Files"""
with open("demo1.txt") as file:
    xy = file.read()
print(xy)
"""Using with open(), the file is automatically closed after reading.
file.read() reads the entire file.
file.readline() reads one line at a time.
file.readlines() returns a list of lines."""

"""4. Writing to Files"""
f = open("demo1.txt", "w")
f.write("Adding new lines")
f.close()
""" "w" mode deletes existing content before writing.
To append data: """
f = open("demo1.txt", "a")
f.write("\nNow the file has more content!")
f.close()
""" "a" mode adds content without deleting existing data."""

"""5. Creating a File"""
f = open("demo2.txt", "x")  # Creates a new file
f.write("New file")
f.close()
""" "x" creates a file; errors occur if it already exists."""

"""6. Deleting a File"""
import os
os.remove("demo2.txt")  # Deletes "demo2.txt"
"""os.remove(filename) deletes a file."""


"""Exception Handling in Python
1. Handling Exceptions with try-except"""
try:
    x = eval(input("Enter the 1st value: "))
    y = eval(input("Enter the 2nd value: "))
    results = x + y
    print(results)
except:
    print("Please enter a valid number")
"""try: contains code that may cause an error.
except: executes if an error occurs."""

"""2. Handling Specific Exceptions"""
try:
    x = eval(input("Enter the 1st value: "))
    y = eval(input("Enter the 2nd value: "))
    results = x / y
    print("Final results =", results)
except ZeroDivisionError:
    print("Please enter a non-zero divisor")
except NameError:
    print("Please enter valid numbers")
except TypeError:
    print("Please enter numbers of the same type")
"""Handles different exceptions separately."""

"""3. Using else and finally"""
try:
    x = int(input("Enter a number: "))
    y = int(input("Enter another number: "))
    z = x / y
except ZeroDivisionError:
    print("Division by zero is not allowed")
else:
    print("Division =", z)
finally:
    print("Finally block executed")
"""else: runs if no exception occurs.
finally: always executes, used for cleanup."""

"""4. Raising Custom Exceptions"""
x = int(input("Enter the number: "))
if x < 18:
    raise Exception("You are not eligible for voting")
"""raise is used to trigger user-defined exceptions."""

"""Summary
File Handling

Open, read, write, append, and delete files.
Use "x" mode to create a file and "a" mode to append.
Use with open() to handle files safely.
Exception Handling

Use try-except to catch errors.
Use specific exception types for better error handling.
else runs when no exception occurs.
finally always runs, useful for cleanup.
Use raise for custom exceptions."""


"""File Handling Exercises
Exercise 1: Create and Write to a File
Task: Write a Python program to create a file called testfile.txt and write "Hello, this is a test file" into it."""
# Creating and writing to a file
with open("testfile.txt", "w") as f:
    f.write("Hello, this is a test file")

print("File created and written successfully!")

"""Exercise 2: Read a File
Task: Open testfile.txt, read the contents, and print them."""
# Reading the file
with open("testfile.txt", "r") as f:
    content = f.read()

print("File Content:", content)

"""Exercise 3: Append Text to a File
Task: Append "This is an additional line" to testfile.txt without deleting its previous content."""
# Appending to a file
with open("testfile.txt", "a") as f:
    f.write("\nThis is an additional line")

print("Text appended successfully!")

"""Exercise 4: Read a File Line by Line
Task: Read testfile.txt line by line and print each line separately."""
# Reading line by line
with open("testfile.txt", "r") as f:
    for line in f:
        print(line.strip())  # strip() removes extra newlines

"""Exercise 5: Count Number of Words in a File
Task: Count the number of words in testfile.txt."""
# Counting words
with open("testfile.txt", "r") as f:
    content = f.read()
word_count = len(content.split())
print("Total number of words:", word_count)

"""Exercise 6: Delete a File
Task: Delete testfile.txt using Python."""
import os
# Check if file exists before deleting
if os.path.exists("testfile.txt"):
    os.remove("testfile.txt")
    print("File deleted successfully!")
else:
    print("File does not exist!")

"""Exception Handling Exercises
Exercise 7: Handle Division by Zero
Task: Write a Python program that asks the user for two numbers and divides them. 
Handle the case when the user tries to divide by zero."""
try:
    a = int(input("Enter numerator: "))
    b = int(input("Enter denominator: "))
    result = a / b
    print("Result:", result)
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
except ValueError:
    print("Error: Please enter valid numbers.")

"""Exercise 8: Handle File Not Found Error
Task: Try to open a file that does not exist and handle the FileNotFoundError."""
try:
    with open("missingfile.txt", "r") as f:
        content = f.read()
        print(content)
except FileNotFoundError:
    print("Error: The file does not exist.")

"""Exercise 9: Handle Multiple Exceptions
Task: Ask the user to enter two numbers. Handle errors if the user enters a non-numeric value or tries to divide by zero."""
try:
    x = int(input("Enter first number: "))
    y = int(input("Enter second number: "))
    print("Result:", x / y)
except ZeroDivisionError:
    print("Error: You cannot divide by zero.")
except ValueError:
    print("Error: Please enter only numbers.")

"""Exercise 10: Use finally Block
Task: Modify Exercise 9 to always print "Program execution completed" using the finally block."""
try:
    x = int(input("Enter first number: "))
    y = int(input("Enter second number: "))
    print("Result:", x / y)
except ZeroDivisionError:
    print("Error: You cannot divide by zero.")
except ValueError:
    print("Error: Please enter only numbers.")
finally:
    print("Program execution completed.")

"""Exercise 11: Raise a Custom Exception
Task: Write a program that asks the user for their age. 
If the age is less than 18, raise an exception with the message "You must be 18 or older". """
age = int(input("Enter your age: "))

if age < 18:
    raise Exception("You must be 18 or older to proceed.")
else:
    print("You are eligible!")


"""Summary of Concepts Covered
✅ File Handling
Opening, reading, writing, and appending files
Reading files line by line
Counting words in a file
Deleting a file
✅ Exception Handling
Handling ZeroDivisionError
Handling FileNotFoundError
Handling multiple exceptions
Using finally for cleanup
Raising custom exceptions"""

