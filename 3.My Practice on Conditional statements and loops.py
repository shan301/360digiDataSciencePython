"""1. Conditional Statements (Control Flow)
Conditional statements allow your program to make decisions based on certain conditions.

1.1. If Statement
The if statement executes a block of code if a condition is True."""
a = 23  
if a >= 22:
   print("if")  # This will print "if" because 23 >= 22

"""1.2.If-Else Statement
If a condition is True, the if block runs; otherwise, the else block executes."""
is_male = False  

if is_male:
    print("You are male")
else:
    print("You are female")  # This will print "You are female" because is_male is False.

"""1.3. If-Elif-Else Statement
Used when there are multiple conditions to check."""
a = 23  

if a >= 22:
   print("if")
   print("one more statement")
elif a >= 21:
   print("elif")
else:
   print("else")
"""Here, if executes because a = 23 satisfies a >= 22."""

"""1.4.Nested If Statements
An if statement inside another if.
Check if a number is positive and even."""
num = int(input("Enter a number: "))

if num > 0:
    print("The number is positive")
    if num % 2 == 0:
        print("The number is even")
    else:
        print("The number is odd")
else:
    print("The number is not positive")

    

"""1. Conditional Statements Questions
Exercise 1: Check Even or Odd-Write a program that takes an integer input from the user and checks whether it is even or odd."""
num = int(input("Enter a number: "))  

if num % 2 == 0:
    print("Even number")
else:
    print("Odd number")

"""Exercise 2: Age Category-Write a program that asks the user for their age and prints whether they are:
Child (0-12 years)
Teenager (13-19 years)
Adult (20-59 years)
Senior (60+ years)"""
age = int(input("Enter your age: "))  

if age >= 0 and age <= 12:
    print("Child")
elif age >= 13 and age <= 19:
    print("Teenager")
elif age >= 20 and age <= 59:
    print("Adult")
elif age >= 60:
    print("Senior")
else:
    print("Invalid age entered")

"""Exercise 3: Find the Largest Number
Write a program that takes three numbers as input and prints the largest one."""
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

if a >= b and a >= c:
    print(f"Largest number is: {a}")
elif b >= a and b >= c:
    print(f"Largest number is: {b}")
else:
    print(f"Largest number is: {c}")

"""Exercise 4: Grading System
Write a program that asks the user for their exam score (0-100) and prints their grade:

A (90-100)
B (80-89)
C (70-79)
D (60-69)
Fail (below 60)"""
score = int(input("Enter your exam score (0-100): "))

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
elif score >= 60:
    print("Grade: D")
else:
    print("Fail")

"""Exercise 5: Print Numbers 1 to 10
Use a for loop to print numbers from 1 to 10."""
for i in range(1, 11):
    print(i)

"""Exercise 6: Print Even Numbers (0-20)
Write a program that prints even numbers from 0 to 20."""
for i in range(0, 21, 2):  
    print(i)

"""Exercise 7: Sum of First 10 Natural Numbers
Write a program that calculates and prints the sum of the first 10 natural numbers (1 to 10)."""
sum = 0
for i in range(1, 11):
    sum += i

print(f"Sum of first 10 natural numbers: {sum}")

"""Exercise 8: Factorial of a Number
Write a program that calculates the factorial of a given number n."""
num = int(input("Enter a number: "))
factorial = 1

if num < 0:
    print("Factorial is not defined for negative numbers.")
elif num == 0:
    print("Factorial of 0 is 1")
else:
    for i in range(1, num + 1):
        factorial *= i
    print(f"Factorial of {num} is {factorial}")

"""Exercise 9: Multiplication Table
Ask the user for a number and print its multiplication table (up to 10)."""
num = int(input("Enter a number for multiplication table: "))

for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")

"""Exercise 10: Reverse a String Using a Loop
Write a program that takes a string as input and prints the reversed string using a loop."""
string = input("Enter a string: ")
reversed_string = ""

for char in string:
    reversed_string = char + reversed_string

print(f"Reversed string: {reversed_string}")

"""Exercise 11: Loan Eligibility Checker
Write a program that checks if a person is eligible for a loan based on these conditions:

Condition 1: Age should be 21 or older.
Condition 2 (Nested inside Condition 1): Salary should be at least ₹30,000.
Condition 3 (Nested inside Condition 2): Credit score should be 700 or more.
Output:

If all conditions are met, print: "You are eligible for a loan."
If age is less than 21, print: "You are too young to apply for a loan."
If salary is insufficient, print: "You need to have a minimum salary of ₹30,000."
If credit score is low, print: "Your credit score is too low to get a loan."""
age = int(input("Enter your age: "))
salary = int(input("Enter your monthly salary: "))
credit_score = int(input("Enter your credit score: "))

# Outer if for age
if age >= 21:
    # First level passed (age ok), now check salary
    if salary >= 30000:
        # Second level passed (salary ok), now check credit score
        if credit_score >= 700:
            print("You are eligible for a loan.")
        else:
            print("Your credit score is too low to get a loan.")
    else:
        print("You need to have a minimum salary of ₹30,000.")
else:
    print("You are too young to apply for a loan.")

"""Exercise 12:Ask the user to enter their age and marks:

If age ≥ 18, check marks.
If marks ≥ 50, print "Eligible for scholarship".
Else, print "Not eligible for scholarship".
If age < 18, print "You are too young"."""
# Taking inputs from user
age = int(input("Enter your age: "))
marks = int(input("Enter your marks: "))

# Outer condition to check age
if age >= 18:
    print("Age requirement met")
    # Inner condition to check marks
    if marks >= 50:
        print("Eligible for scholarship")
    else:
        print("Not eligible for scholarship")
else:
    print("You are too young")
