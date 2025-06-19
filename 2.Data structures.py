# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:16:14 2025

@author: Shantanu
"""

"""2. Data Structures
Data structures are ways to store and organize data efficiently in Python. Common data structures include lists, tuples, dictionaries, and sets, which are widely used in data analysis and science.

2.1. Lists
Lists are mutable, ordered collections of items, enclosed in square brackets []."""
numbers = [1, 2, 3, 4, 5]
print(f"List: {numbers}")  # Output: List: [1, 2, 3, 4, 5]
numbers.append(6)  # Adds 6 to the end
print(f"After append: {numbers}")  # Output: After append: [1, 2, 3, 4, 5, 6]

"""2.2. Tuples
Tuples are immutable, ordered collections of items, enclosed in parentheses ()."""
coordinates = (10, 20)
print(f"Tuple: {coordinates}")  # Output: Tuple: (10, 20)
# coordinates[0] = 15  # This would raise an error as tuples are immutable

"""2.3. Dictionaries
Dictionaries are mutable, unordered collections of key-value pairs, enclosed in curly braces {}."""
student = {"name": "Alice", "age": 25, "grade": "A"}
print(f"Dictionary: {student}")  # Output: Dictionary: {'name': 'Alice', 'age': 25, 'grade': 'A'}
print(f"Student name: {student['name']}")  # Output: Student name: Alice
student["age"] = 26  # Update value
print(f"Updated dictionary: {student}")  # Output: Updated dictionary: {'name': 'Alice', 'age': 26, 'grade': 'A'}

"""2.4. Sets
Sets are mutable, unordered collections of unique items, enclosed in curly braces {}."""
fruits = {"apple", "banana", "apple", "orange"}
print(f"Set: {fruits}")  # Output: Set: {'apple', 'banana', 'orange'} (duplicates removed)
fruits.add("mango")
print(f"After adding mango: {fruits}")  # Output: After adding mango: {'apple', 'banana', 'orange', 'mango'}

"""2.5. List Comprehensions
List comprehensions provide a concise way to create lists."""
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")  # Output: Squares: [1, 4, 9, 16, 25]






"""2. Data Structures Exercises
Exercise 1: Create a List
Write a program that creates a list of 5 favorite data analysis tools and prints it."""
tools = ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn"]
print(f"Favorite data analysis tools: {tools}")

"""Exercise 2: Sum of List Elements
Write a program that takes a list of numbers as input and calculates their sum."""
numbers = list(map(int, input("Enter numbers separated by spaces: ").split()))
total = sum(numbers)
print(f"Sum of numbers: {total}")

"""Exercise 3: Filter Even Numbers
Write a program that takes a list of numbers and creates a new list with only even numbers using a list comprehension."""
numbers = list(map(int, input("Enter numbers separated by spaces: ").split()))
even_numbers = [num for num in numbers if num % 2 == 0]
print(f"Even numbers: {even_numbers}")

"""Exercise 4: Tuple of Coordinates
Write a program that asks for x and y coordinates, stores them in a tuple, and prints the tuple."""
x = float(input("Enter x coordinate: "))
y = float(input("Enter y coordinate: "))
point = (x, y)
print(f"Coordinate tuple: {point}")

"""Exercise 5: Dictionary of Student Grades
Write a program that creates a dictionary with 3 students' names and their grades, then prints the dictionary."""
grades = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78
}
print(f"Student grades: {grades}")

"""Exercise 6: Update Dictionary
Write a program that asks for a student's name and new grade, then updates the dictionary from Exercise 5."""
name = input("Enter student name: ")
grade = int(input("Enter new grade: "))
grades[name] = grade
print(f"Updated grades: {grades}")

"""Exercise 7: Find Common Elements
Write a program that takes two lists as input and finds common elements using sets."""
list1 = input("Enter first list (space-separated): ").split()
list2 = input("Enter second list (space-separated): ").split()
common = set(list1) & set(list2)
print(f"Common elements: {common}")

"""Exercise 8: Remove Duplicates
Write a program that takes a list of numbers as input and removes duplicates using a set."""
numbers = list(map(int, input("Enter numbers separated by spaces: ").split()))
unique_numbers = list(set(numbers))
print(f"List without duplicates: {unique_numbers}")

"""Exercise 9: Dictionary from Lists
Write a program that takes two lists (keys and values) and creates a dictionary."""
keys = input("Enter keys (space-separated): ").split()
values = input("Enter values (space-separated): ").split()
if len(keys) == len(values):
    dictionary = dict(zip(keys, values))
    print(f"Created dictionary: {dictionary}")
else:
    print("Lists must have equal length")

"""Exercise 10: Count Word Frequency
Write a program that takes a sentence as input and counts the frequency of each word using a dictionary."""
sentence = input("Enter a sentence: ")
words = sentence.lower().split()
word_freq = {}
for word in words:
    word_freq[word] = word_freq.get(word, 0) + 1
print(f"Word frequency: {word_freq}")

"""Exercise 11: Sort List of Dictionaries
Write a program that creates a list of dictionaries with student names and scores, then sorts by score."""
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78}
]
sorted_students = sorted(students, key=lambda x: x["score"], reverse=True)
print(f"Sorted students by score: {sorted_students}")

"""Exercise 12: Nested Dictionary
Write a program that creates a nested dictionary for a dataset with two categories (e.g., 'sales' and 'expenses') and prints it."""
dataset = {
    "sales": {
        "Q1": 50000,
        "Q2": 60000
    },
    "expenses": {
        "Q1": 30000,
        "Q2": 35000
    }
}
print(f"Dataset: {dataset}")