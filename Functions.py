"""1. Basic Functions
Functions are reusable blocks of code that perform specific tasks. 
They are defined using the def keyword and can be called multiple times to avoid repetition."""
def sayHello():
    print('Hello World!')

sayHello()

"""2. Function Parameters & Arguments
Functions can take inputs called parameters to process data. 
Arguments are actual values passed to these parameters when calling a function.
Positional arguments are values passed to a function in the same order as the parameters are defined. 
The position of each argument is crucial and matched accordingly."""
def hello(name, age, sal):
    print("Hi", name, "Your age:", age, "Your salary:", sal)

hello("Sharat", 25, 50000)
"""Default Arguments-Default arguments allow a function's parameter to have a default value if no argument 
is provided during the function call. 
This makes some parameters optional.
"""
def say(message, times=1):
    print(message * times)

say('Hello')       # Default `times=1`
say('World ', 5)   # Overriding default
"""Keyword arguments are passed using the parameter name (e.g., name='John'), making it clear which parameter is being set. 
They can be used in any order when calling the function."""
def func(a, b=5, c=10):
    print(f'a is {a}, b is {b}, c is {c}')

func(3, 7)
func(25, c=24)
func(c=50, a=100)

"""3.Variable scope defines where a variable is accessible: Local variables exist within a function, while 
Global variables are defined outside and accessible throughout the program."""
x = 50
def func(x):
    print('x is', x)
    x = 2  # Local variable
    print('Changed local x to', x)

func(x)
print('x is still', x)

"""4.Variable-Length Arguments-Variable-length arguments allow a function to accept any number of arguments. 
They are defined using *args for positional arguments and **kwargs for keyword arguments.
*args-*args is used to pass a variable number of positional arguments to a function. 
Inside the function, args is treated as a tuple of all additional positional arguments.
**kwargs-kwargs allows passing any number of keyword arguments, which are accessible inside the function as a dictionary. 
This is useful when you don't know beforehand how many keyword arguments will be passed."""
def myFun(*argv):  
    for arg in argv:  
        print(arg)

myFun('Hello', 'Welcome', 'to', '360digitmg')

def intro(**data):
    for key, value in data.items():
        print(f"{key} is {value}")

intro(Firstname="Sita", Age=22, Phone=1234567890)

"""5. Recursion-Recursion is a technique where a function calls itself to solve smaller instances of a problem.
It requires a base condition to stop the recursive calls and avoid infinite loops."""
def fact(N):
    if N == 0:
        return 1
    else:
        return N * fact(N-1)

print(fact(5))

"""6.List Comprehension-List comprehension is a concise way to create lists using a single line of code. 
It can include conditions and loops to generate lists efficiently."""
symbols = '$¢£¥€'
codes = [ord(symbol) for symbol in symbols]
print(codes)

"""7. Dictionary Comprehension-Dictionary comprehension allows creating dictionaries in a single line by defining 
key-value pairs with loops and conditions, making the code more readable and compact."""
numbers = range(10)
new_dict_comp = {n: n**2 for n in numbers if n % 2 == 0}
print(new_dict_comp)

"""8. Zip and Unzip-zip() combines multiple iterables (like lists) into tuples, pairing elements by index. 
Unzip is done using zip(*zipped_data) to separate zipped tuples back into individual lists."""
name = ["Manjeet", "Nikhil", "Shambhavi"]
roll_no = [4, 1, 3]

mapped = zip(name, roll_no)
print(set(mapped))

"""9. Built-in Functions: Sorting-Python's sorted() function sorts items in a list or iterable in ascending or descending order. 
You can also customize sorting using the key and reverse parameters."""
animals = ['cat', 'dog', 'elephant']
print(sorted(animals, key=len))

"""10.Lambda Functions-Lambda functions are small anonymous functions defined with the lambda keyword. 
They are often used for short, simple operations without formally defining a function."""
s = lambda x: x * x
print(s(12))
"""Using map()"""
val = [1, 2, 3, 4, 5, 6]
doubled = list(map(lambda x: x * 2, val))
print(doubled)
"""Using filter()"""
filtered_odd = list(filter(lambda x: x % 2, val))
print(filtered_odd)
"""Using reduce()"""
from functools import reduce
product = reduce(lambda x, y: x * y, val)
print(product)

"""11.Iterators-Iterators are objects that allow you to traverse through all elements of a collection one at a time using next(). 
They are created from iterables (like lists) using iter()."""
names = ["Rishu", 'Aayush', 'Shubh', 'Ram']
new_list = iter(names)
print(next(new_list))
print(next(new_list))

"""12. Generators (yield)-Generators are special functions that yield items one by one using the yield keyword. 
They are memory-efficient because they produce items on the fly without storing them all at once."""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

x = fib(5)
for i in x:
    print(i)

"""13. String Slicing-String slicing is a way to access substrings by specifying a range of indices. 
It uses the syntax string[start:end:step] to extract parts of the string efficiently."""
String = 'DATA SCIENTIST'
print(String[:6])
print(String[-1:-14:-3])

"""Questions"""
"""Write a function greet that takes a name as input and prints "Hello, [name]!"""
def greet(name):
    print(f"Hello, {name}!")

greet("Rahul")

"""Write a function add_numbers that takes two numbers as arguments and returns their sum."""
def add_numbers(a, b):
    return a + b

print(add_numbers(5, 10))

"""Modify the greet function so that if no name is provided, it defaults to "Guest"."""
def greet(name="Guest"):
    print(f"Hello, {name}!")

greet()       # Should print "Hello, Guest!"
greet("Amit") # Should print "Hello, Amit!"

"""Keyword Arguments
Write a function describe_person that takes name, age, and city as keyword arguments and prints them."""
def describe_person(name, age, city):
    print(f"{name} is {age} years old and lives in {city}.")

describe_person(age=30, city="Mumbai", name="Anil")

""" Variable-Length Arguments (*args)
Exercise:
Write a function sum_all that takes any number of arguments and returns their sum."""
def sum_all(*numbers):
    return sum(numbers)

print(sum_all(1, 2, 3, 4, 5))  # 15
print(sum_all(10, 20))         # 30

"""Recursion
Exercise:
Write a recursive function factorial that calculates the factorial of a number."""
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120

"""List Comprehension
Exercise:
Use list comprehension to create a list of squares from 1 to 10."""
squares = [x**2 for x in range(1, 11)]
print(squares)

"""Dictionary Comprehension
Create a dictionary where keys are numbers from 1 to 5 and values are their cubes."""
cubes = {x: x**3 for x in range(1, 6)}
print(cubes)

"""Lambda Functions
Exercise:
Use a lambda function to multiply two numbers."""
multiply = lambda x, y: x * y
print(multiply(5, 3))

"""map() Function
Use map() to convert a list of numbers into their squares."""
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)

"""filter() Function
Use filter() to extract even numbers from a list."""
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

"""Iterators
Convert a list into an iterator and print its elements using next()."""
numbers = [10, 20, 30, 40]
iterator = iter(numbers)

print(next(iterator))  # 10
print(next(iterator))  # 20
print(next(iterator))  # 30
print(next(iterator))  # 40

"""Generators (yield)
Write a generator function countdown that counts down from a given number"""
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)

"""Zip and Unzip
Use zip() to combine two lists into a dictionary."""
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

combined = dict(zip(names, ages))
print(combined)

"""String Slicing
Extract "SCI" from the string "DATA SCIENTIST" using slicing."""
text = "DATA SCIENTIST"
print(text[5:8])

"""Fibonacci Generator
Write a generator function fibonacci that generates the first n Fibonacci numbers."""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(7):
    print(num)

