"""3. Functions
Functions are reusable blocks of code that perform specific tasks, defined using the `def` keyword. They are essential for organizing code and automating repetitive tasks in data analysis.

3.1. Basic Functions
Functions encapsulate code for reuse, taking no arguments or returning no values if needed."""
def say_hello():
    print("Hello World!")

say_hello()  # Output: Hello World!

"""3.2. Function Parameters and Arguments
Functions can accept inputs (parameters) and process them. Arguments are values passed during function calls."""
def employee_info(name, age, salary):
    print(f"Hi {name}, Age: {age}, Salary: {salary}")

employee_info("Sharat", 25, 50000)  # Output: Hi Sharat, Age: 25, Salary: 50000

"""3.3. Default Arguments
Default arguments provide optional parameters with preset values."""
def greet(message, times=1):
    print(message * times)

greet("Hello ")       # Output: Hello 
greet("World ", 3)    # Output: World World World 

"""3.4. Keyword Arguments
Keyword arguments allow passing arguments by parameter name, enabling flexible order."""
def describe_person(name, age, city):
    print(f"{name} is {age} years old, lives in {city}")

describe_person(age=30, city="Mumbai", name="Anil")  # Output: Anil is 30 years old, lives in Mumbai

"""3.5. Variable-Length Arguments
*args accepts multiple positional arguments as a tuple, **kwargs accepts keyword arguments as a dictionary."""
def print_args(*args):
    for arg in args:
        print(arg)

print_args("Data", "Science", 2025)  # Output: Data, Science, 2025

def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_kwargs(name="Alice", role="Analyst")  # Output: name: Alice, role: Analyst

"""3.6. Variable Scope
Local variables exist within a function; global variables are accessible throughout."""
global_var = 50
def modify_var(x):
    local_var = 2
    print(f"Inside function, x: {x}, local_var: {local_var}")

modify_var(global_var)  # Output: Inside function, x: 50, local_var: 2
print(f"Outside function, global_var: {global_var}")  # Output: Outside function, global_var: 50

"""3.7. Recursion
Recursive functions call themselves to solve smaller instances, requiring a base case."""
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(f"Factorial of 5: {factorial(5)}")  # Output: Factorial of 5: 120

"""3.8. Lambda Functions
Lambda functions are anonymous, concise functions for simple operations."""
square = lambda x: x * x
print(f"Square of 4: {square(4)}")  # Output: Square of 4: 16

"""3.9. Map, Filter, and Reduce
Map applies a function to all items in an iterable; filter selects items; reduce aggregates items."""
from functools import reduce

numbers = [1, 2, 3, 4]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
product = reduce(lambda x, y: x * y, numbers)
print(f"Squares: {squares}")  # Output: Squares: [1, 4, 9, 16]
print(f"Evens: {evens}")      # Output: Evens: [2, 4]
print(f"Product: {product}")   # Output: Product: 24

"""3.10. Iterators
Iterators traverse collections using iter() and next()."""
data = ["Mean", "Median", "Mode"]
iterator = iter(data)
print(next(iterator))  # Output: Mean
print(next(iterator))  # Output: Median

"""3.11. Generators
Generators yield values one at a time, saving memory."""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("Fibonacci sequence:", list(fibonacci(5)))  # Output: Fibonacci sequence: [0, 1, 1, 2, 3]

"""3.12. List and Dictionary Comprehension
Comprehensions provide concise ways to create lists or dictionaries."""
squares_list = [x**2 for x in range(1, 5)]
cubes_dict = {x: x**3 for x in range(1, 4)}
print(f"Squares list: {squares_list}")  # Output: Squares list: [1, 4, 9, 16]
print(f"Cubes dict: {cubes_dict}")      # Output: Cubes dict: {1: 1, 2: 8, 3: 27}

"""3.13. Zip and Unzip
Zip combines iterables into tuples; unzip separates them."""
names = ["Alice", "Bob"]
ages = [25, 30]
zipped = dict(zip(names, ages))
print(f"Zipped: {zipped}")  # Output: Zipped: {'Alice': 25, 'Bob': 30}

"""3.14. Function Decorators
Decorators modify or extend function behavior, useful for logging or validation."""
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b

print(add(2, 3))  # Output: Calling add with args: (2, 3), kwargs: {}, 5

"""3.15. Partial Functions
Partial functions fix some arguments of a function, simplifying calls."""
from functools import partial

def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(f"Double of 5: {double(5)}")  # Output: Double of 5: 10

"""3.16. Error Handling in Functions
Functions can handle exceptions to make code robust."""
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero"

print(divide(10, 2))  # Output: 5.0
print(divide(10, 0))  # Output: Error: Division by zero

"""3. Functions Exercises
Exercise 1: Greet Function
Write a function that takes a name and prints a greeting."""
def greet(name):
    print(f"Hello, {name}!")

greet("Rahul")  # Output: Hello, Rahul!

"""Exercise 2: Sum of Numbers
Write a function that takes two numbers and returns their sum."""
def add_numbers(a, b):
    return a + b

print(f"Sum: {add_numbers(5, 10)}")  # Output: Sum: 15

"""Exercise 3: Default Greeting
Write a function that greets a user with a default name 'Guest' if no name is provided."""
def greet_default(name="Guest"):
    print(f"Hello, {name}!")

greet_default()        # Output: Hello, Guest!
greet_default("Amit")  # Output: Hello, Amit!

"""Exercise 4: Keyword Arguments
Write a function that describes a dataset using keyword arguments."""
def describe_dataset(name, rows, columns):
    print(f"Dataset {name} has {rows} rows and {columns} columns.")

describe_dataset(name="Sales", columns=5, rows=100)  # Output: Dataset Sales has 100 rows and 5 columns.

"""Exercise 5: Variable-Length Sum
Write a function that sums any number of numerical arguments."""
def sum_all(*numbers):
    return sum(numbers)

print(f"Sum: {sum_all(1, 2, 3, 4)}")  # Output: Sum: 10

"""Exercise 6: Recursive Power
Write a recursive function to calculate x raised to power n."""
def power(x, n):
    if n == 0:
        return 1
    return x * power(x, n - 1)

print(f"2^3: {power(2, 3)}")  # Output: 2^3: 8

"""Exercise 7: Lambda for Scaling
Write a lambda function to scale a number by a factor."""
scale = lambda x, factor: x * factor
print(f"Scale 10 by 3: {scale(10, 3)}")  # Output: Scale 10 by 3: 30

"""Exercise 8: Map for Data Transformation
Use map() to convert a list of temperatures from Celsius to Fahrenheit."""
celsius = [0, 10, 20, 30]
fahrenheit = list(map(lambda x: (x * 9/5) + 32, celsius))
print(f"Fahrenheit: {fahrenheit}")  # Output: Fahrenheit: [32.0, 50.0, 68.0, 86.0]

"""Exercise 9: Filter for Positive Numbers
Use filter() to extract positive numbers from a list."""
numbers = [-2, -1, 0, 1, 2]
positives = list(filter(lambda x: x > 0, numbers))
print(f"Positives: {positives}")  # Output: Positives: [1, 2]

"""Exercise 10: Iterator for Column Names
Convert a list of column names into an iterator and print the first two."""
columns = ["ID", "Name", "Salary", "Dept"]
iterator = iter(columns)
print(f"First: {next(iterator)}")  # Output: First: ID
print(f"Second: {next(iterator)}")  # Output: Second: Name

"""Exercise 11: Generator for Even Numbers
Write a generator function to yield even numbers up to n."""
def even_numbers(n):
    for i in range(2, n + 1, 2):
        yield i

print("Evens:", list(even_numbers(10)))  # Output: Evens: [2, 4, 6, 8, 10]

"""Exercise 12: Decorator for Timing
Write a decorator to measure the execution time of a function."""
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_sum(n):
    return sum(range(n))

print(slow_sum(1000000))  # Output: slow_sum took <time> seconds, 499999500000

"""Exercise 13: Partial Function for Data Scaling
Use partial to create a function that scales data by a fixed factor."""
from functools import partial

def scale_data(value, factor):
    return value * factor

scale_by_10 = partial(scale_data, factor=10)
print(f"Scale 5 by 10: {scale_by_10(5)}")  # Output: Scale 5 by 10: 50

"""Exercise 14: Error Handling in Division
Write a function that divides two numbers and handles division by zero."""
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"

print(safe_divide(10, 2))  # Output: 5.0
print(safe_divide(10, 0))  # Output: Error: Cannot divide by zero

"""Exercise 15: Zip for Dataset Creation
Use zip() to combine lists of feature names and values into a dictionary."""
features = ["Age", "Salary", "Experience"]
values = [30, 60000, 5]
dataset = dict(zip(features, values))
print(f"Dataset: {dataset}")  # Output: Dataset: {'Age': 30, 'Salary': 60000, 'Experience': 5}

