# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:17:50 2025

@author: Shantanu
"""

"""5. Object-Oriented Programming (OOP)
OOP is a programming paradigm that uses objects and classes to structure code. It is widely used in data analysis for organizing and managing data processing tasks.

5.1. Classes and Objects
A class is a blueprint for objects, and an object is an instance of a class."""
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def display(self):
        return f"Student: {self.name}, Age: {self.age}"

student1 = Student("Alice", 25)
print(student1.display())  # Output: Student: Alice, Age: 25

"""5.2. Methods
Methods are functions defined inside a class that operate on objects."""
class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def calculate_mean(self):
        return sum(self.data) / len(self.data)

analyzer = DataAnalyzer([10, 20, 30, 40])
print(f"Mean: {analyzer.calculate_mean()}")  # Output: Mean: 25.0

"""5.3. Inheritance
Inheritance allows a class to inherit attributes and methods from another class."""
class Person:
    def __init__(self, name):
        self.name = name
    
    def introduce(self):
        return f"Hi, I'm {self.name}"

class Employee(Person):
    def __init__(self, name, job_title):
        super().__init__(name)
        self.job_title = job_title
    
    def introduce(self):
        return f"{super().introduce()}, Job Title: {self.job_title}"

employee = Employee("Bob", "Data Scientist")
print(employee.introduce())  # Output: Hi, I'm Bob, Job Title: Data Scientist

"""5.4. Encapsulation
Encapsulation restricts access to an object's data, using private attributes (with _ or __)."""
class Dataset:
    def __init__(self):
        self.__data = []
    
    def add_data(self, value):
        self.__data.append(value)
    
    def get_data(self):
        return self.__data

dataset = Dataset()
dataset.add_data(42)
print(f"Dataset: {dataset.get_data()}")  # Output: Dataset: [42]

"""5.5. Polymorphism
Polymorphism allows different classes to define methods with the same name but different behaviors."""
class Visualization:
    def plot(self):
        pass

class BarChart(Visualization):
    def plot(self):
        return "Plotting a bar chart"

class LineChart(Visualization):
    def plot(self):
        return "Plotting a line chart"

bar = BarChart()
line = LineChart()
print(bar.plot())  # Output: Plotting a bar chart
print(line.plot())  # Output: Plotting a line chart

"""5. OOP Exercises
Exercise 1: Create a Class
Write a program that creates a class 'Book' with attributes title and author, and a method to display book details."""
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    
    def display(self):
        return f"Book: {self.title} by {self.author}"

book = Book("Python for Data Analysis", "Wes McKinney")
print(book.display())

"""Exercise 2: Class for Data Statistics
Write a program that creates a class 'DataStats' to calculate the sum and average of a list of numbers."""
class DataStats:
    def __init__(self, numbers):
        self.numbers = numbers
    
    def get_sum(self):
        return sum(self.numbers)
    
    def get_average(self):
        return sum(self.numbers) / len(self.numbers) if self.numbers else 0

stats = DataStats([10, 20, 30, 40, 50])
print(f"Sum: {stats.get_sum()}")
print(f"Average: {stats.get_average()}")

"""Exercise 3: Inheritance with Employee
Write a program that creates a 'Person' class and an 'Employee' class that inherits from it, adding a salary attribute."""
class Person:
    def __init__(self, name):
        self.name = name
    
    def info(self):
        return f"Name: {self.name}"

class Employee(Person):
    def __init__(self, name, salary):
        super().__init__(name)
        self.salary = salary
    
    def info(self):
        return f"{super().info()}, Salary: {self.salary}"

emp = Employee("Alice", 60000)
print(emp.info())

"""Exercise 4: Encapsulated Bank Account
Write a program that creates a 'BankAccount' class with a private balance attribute and methods to deposit and check balance."""
class BankAccount:
    def __init__(self):
        self.__balance = 0
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited {amount}. New balance: {self.__balance}"
        return "Invalid deposit amount"
    
    def get_balance(self):
        return f"Balance: {self.__balance}"

account = BankAccount()
print(account.deposit(1000))
print(account.get_balance())

"""Exercise 5: Polymorphic Data Visualizers
Write a program with a base class 'Visualizer' and two derived classes 'Histogram' and 'ScatterPlot' with a 'draw' method."""
class Visualizer:
    def draw(self):
        pass

class Histogram(Visualizer):
    def draw(self):
        return "Drawing a histogram"

class ScatterPlot(Visualizer):
    def draw(self):
        return "Drawing a scatter plot"

hist = Histogram()
scat = ScatterPlot()
print(hist.draw())
print(scat.draw())

"""Exercise 6: Class for Data Cleaning
Write a program that creates a 'DataCleaner' class to remove None values from a list."""
class DataCleaner:
    def __init__(self, data):
        self.data = data
    
    def remove_none(self):
        self.data = [x for x in self.data if x is not None]
        return self.data

cleaner = DataCleaner([1, None, 2, None, 3])
print(f"Cleaned data: {cleaner.remove_none()}")

"""Exercise 7: Class for Student Grades
Write a program that creates a 'Student' class to store a student's name and grades, with a method to calculate GPA."""
class Student:
    def __init__(self, name, grades):
        self.name = name
        self.grades = grades
    
    def calculate_gpa(self):
        return sum(self.grades) / len(self.grades) if self.grades else 0

student = Student("Bob", [85, 90, 88])
print(f"{student.name}'s GPA: {student.calculate_gpa()}")

"""Exercise 8: Inheritance with Data Source
Write a program with a 'DataSource' class and a 'CSVDataSource' class that inherits from it to read a CSV file's columns."""
class DataSource:
    def __init__(self, source_name):
        self.source_name = source_name
    
    def get_data(self):
        return "Fetching data"

class CSVDataSource(DataSource):
    def __init__(self, source_name, columns):
        super().__init__(source_name)
        self.columns = columns
    
    def get_data(self):
        return f"Fetching CSV data with columns: {self.columns}"

csv_source = CSVDataSource("data.csv", ["id", "name", "value"])
print(csv_source.get_data())

"""Exercise 9: Encapsulated Dataset
Write a program that creates a 'Dataset' class with a private attribute for data and methods to add and retrieve data."""
class Dataset:
    def __init__(self):
        self.__data = []
    
    def add_data(self, item):
        self.__data.append(item)
    
    def get_data(self):
        return self.__data

dataset = Dataset()
dataset.add_data(42)
dataset.add_data(100)
print(f"Dataset: {dataset.get_data()}")

"""Exercise 10: Class for Data Transformation
Write a program that creates a 'DataTransformer' class to square all numbers in a list."""
class DataTransformer:
    def __init__(self, numbers):
        self.numbers = numbers
    
    def square_numbers(self):
        self.numbers = [x**2 for x in self.numbers]
        return self.numbers

transformer = DataTransformer([1, 2, 3, 4])
print(f"Squared numbers: {transformer.square_numbers()}")

"""Exercise 11: Polymorphic Data Processors
Write a program with a 'Processor' class and two derived classes 'Normalizer' and 'Standardizer' with a 'process' method."""
class Processor:
    def process(self):
        pass

class Normalizer(Processor):
    def process(self):
        return "Normalizing data"

class Standardizer(Processor):
    def process(self):
        return "Standardizing data"

norm = Normalizer()
std = Standardizer()
print(norm.process())
print(std.process())

"""Exercise 12: Class for Data Analysis Pipeline
Write a program that creates a 'DataPipeline' class to combine cleaning and transformation steps."""
class DataPipeline:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        self.data = [x for x in self.data if x is not None]
        return self
    
    def transform_data(self):
        self.data = [x**2 for x in self.data]
        return self
    
    def get_result(self):
        return self.data

pipeline = DataPipeline([1, None, 2, 3])
result = pipeline.clean_data().transform_data().get_result()
print(f"Processed data: {result}")