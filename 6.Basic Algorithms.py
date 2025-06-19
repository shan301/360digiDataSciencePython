# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:25:32 2025

@author: Shantanu
"""

"""6. Basic Algorithms
Algorithms are step-by-step procedures for solving problems. This file covers basic algorithms relevant to data analysis, such as searching, sorting, and simple mathematical computations.

6.1. Linear Search
Linear search checks each element in a list to find a target value."""
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

numbers = [4, 2, 7, 1, 9]
print(f"Index of 7: {linear_search(numbers, 7)}")  # Output: Index of 7: 2
print(f"Index of 5: {linear_search(numbers, 5)}")  # Output: Index of 5: -1

"""6.2. Binary Search
Binary search efficiently finds a target in a sorted list by dividing the search interval in half."""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

sorted_numbers = [1, 3, 5, 7, 9]
print(f"Index of 5: {binary_search(sorted_numbers, 5)}")  # Output: Index of 5: 2

"""6.3. Bubble Sort
Bubble sort repeatedly swaps adjacent elements if they are in the wrong order."""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

unsorted = [64, 34, 25, 12, 22]
print(f"Sorted array: {bubble_sort(unsorted.copy())}")  # Output: Sorted array: [12, 22, 25, 34, 64]

"""6.4. Factorial (Recursive)
A recursive algorithm calculates factorial by calling itself with smaller inputs."""
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(f"Factorial of 5: {factorial(5)}")  # Output: Factorial of 5: 120

"""6.5. Fibonacci Sequence
The Fibonacci sequence generates numbers where each number is the sum of the two preceding ones."""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"Fibonacci of 6: {fibonacci(6)}")  # Output: Fibonacci of 6: 8

"""6. Basic Algorithms Exercises
Exercise 1: Linear Search for Names
Write a program that searches for a name in a list of names and returns its index."""
def search_name(names, target):
    for i in range(len(names)):
        if names[i].lower() == target.lower():
            return i
    return -1

names = ["Alice", "Bob", "Charlie", "David"]
target = input("Enter name to search: ")
print(f"Index of {target}: {search_name(names, target)}")

"""Exercise 2: Binary Search for Scores
Write a program that performs binary search on a sorted list of exam scores."""
def binary_search_scores(scores, target):
    left, right = 0, len(scores) - 1
    while left <= right:
        mid = (left + right) // 2
        if scores[mid] == target:
            return mid
        elif scores[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

scores = sorted(list(map(int, input("Enter sorted scores (space-separated): ").split())))
target = int(input("Enter score to search: "))
print(f"Index of {target}: {binary_search_scores(scores, target)}")

"""Exercise 3: Bubble Sort for Sales Data
Write a program that sorts a list of sales amounts using bubble sort."""
def bubble_sort_sales(sales):
    n = len(sales)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sales[j] > sales[j + 1]:
                sales[j], sales[j + 1] = sales[j + 1], sales[j]
    return sales

sales = list(map(float, input("Enter sales amounts (space-separated): ").split()))
print(f"Sorted sales: {bubble_sort_sales(sales.copy())}")

"""Exercise 4: Recursive Sum
Write a program that calculates the sum of numbers from 1 to n using recursion."""
def recursive_sum(n):
    if n <= 0:
        return 0
    return n + recursive_sum(n - 1)

n = int(input("Enter a number: "))
print(f"Sum from 1 to {n}: {recursive_sum(n)}")

"""Exercise 5: Fibonacci List
Write a program that generates the first n Fibonacci numbers."""
def fibonacci_list(n):
    fib = [0, 1]
    if n <= 2:
        return fib[:n]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

n = int(input("Enter number of Fibonacci numbers: "))
print(f"Fibonacci sequence: {fibonacci_list(n)}")

"""Exercise 6: Find Maximum Value
Write a program that finds the maximum value in a list without using max()."""
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

numbers = list(map(int, input("Enter numbers (space-separated): ").split()))
print(f"Maximum value: {find_max(numbers)}")

"""Exercise 7: Count Occurrences
Write a program that counts the occurrences of a target number in a list."""
def count_occurrences(arr, target):
    count = 0
    for num in arr:
        if num == target:
            count += 1
    return count

numbers = list(map(int, input("Enter numbers (space-separated): ").split()))
target = int(input("Enter number to count: "))
print(f"Occurrences of {target}: {count_occurrences(numbers, target)}")

"""Exercise 8: Reverse List
Write a program that reverses a list without using reverse() or slicing."""
def reverse_list(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr

items = input("Enter items (space-separated): ").split()
print(f"Reversed list: {reverse_list(items.copy())}")

"""Exercise 9: Check Palindrome
Write a program that checks if a string is a palindrome."""
def is_palindrome(s):
    s = s.lower()
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

text = input("Enter a string: ")
print(f"Is palindrome: {is_palindrome(text)}")

"""Exercise 10: Merge Two Sorted Lists
Write a program that merges two sorted lists into a single sorted list."""
def merge_sorted_lists(list1, list2):
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    return merged

list1 = sorted(list(map(int, input("Enter first sorted list (space-separated): ").split())))
list2 = sorted(list(map(int, input("Enter second sorted list (space-separated): ").split())))
print(f"Merged sorted list: {merge_sorted_lists(list1, list2)}")

"""Exercise 11: Prime Number Check
Write a program that checks if a number is prime."""
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

num = int(input("Enter a number: "))
print(f"Is {num} prime? {is_prime(num)}")

"""Exercise 12: GCD (Greatest Common Divisor)
Write a program that calculates the GCD of two numbers using the Euclidean algorithm."""
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print(f"GCD of {a} and {b}: {gcd(a, b)}")