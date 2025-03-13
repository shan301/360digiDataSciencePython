# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:22:38 2025

@author: Shantanu
"""

"""Regular Expressions (RegEx) in Python
Key Concepts in the Script
1. Basic Search (re.search())
re.search(pattern, string) looks for a pattern in a given string.
Example:"""
import re
x = re.search("cat", "A cat and a rat can't be friends.")
print(x)  
# Returns a match object if found, else None

"""2. Using Wildcards (.)
The dot (.) represents any character except a newline.
Example:"""
x = re.search(r" .at ", "A cat and a rat can't be friends.")
print(x)  
# Matches " cat ", " rat ", etc.

"""3. Character Classes ([...])
Square brackets allow matching specific sets of characters.
Example:"""
r"M[ae][iy]er"  
# Matches "Maier", "Mayer", "Meier", "Meyer"

"""4. Finding All Matches (re.findall())
Returns a list of all occurrences of a pattern.
Example:"""
txt = "The rain in Spain"
x = re.findall("ai", txt)
print(x)  
# Output: ['ai', 'ai']

"""5. Special Sequences
Sequence	Meaning
\d	        Any digit (0-9)
\D	        Non-digit
\s	        Any whitespace
\S	        Non-whitespace
\w	        Any word character (alphanumeric + _)
\W	        Non-word character
\b	        Word boundary
\B	        Non-word boundary"""
import re
Nameage = "Sharat is 35 and Pavan is 28"
ages = re.findall(r'\d{1,2}', Nameage)
print(ages)  # Output: ['35', '28']

"""6. Quantifiers
Symbol	    Meaning
*	        0 or more times
+	        1 or more times
?	        0 or 1 time
{n}	        Exactly n times
{n,}	    At least n times
{n,m}	    Between n and m times"""
import re
x = re.findall(r'\d+', "Phone number: 98765")
print(x)  # Output: ['98765']

"""7. re.sub() - Replacing Text
re.sub(pattern, replacement, string) replaces occurrences of a pattern.
Example:"""
import re
phone = "2004-959-559 # This is Phone Number"
num = re.sub('#.*$', "", phone)  # Remove comment
print(num)  # Output: "2004-959-559"

"""8. re.split() - Splitting Strings
re.split(pattern, string) splits a string based on the pattern.
Example:"""
txt = "The rain in Spain"
x = re.split("\s", txt)  # Splits on whitespace
print(x)  # Output: ['The', 'rain', 'in', 'Spain']

"""9. re.compile() - Precompiling Patterns
Improves performance when using the same regex multiple times.
Example:"""
regex = re.compile("\n")
new_str = regex.sub("", "Hello\nWorld")
print(new_str)  # Output: "HelloWorld"

"""10. Real-Life Use Cases
Extracting phone numbers, emails, names, and dates.
Validating PAN numbers, emails, and passwords.
Searching and modifying large text files."""


"""Exercise 1: Find all Email Addresses
Task: Extract all email addresses from the given text."""
import re

text = "Contact us at support@example.com, sales@example.org, and info@company.net."

# Your code here
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

print(emails)  # Expected Output: ['support@example.com', 'sales@example.org', 'info@company.net']


"""Exercise 2: Extract Dates
Task: Find all dates in the format DD/MM/YYYY."""

import re

text = "Today's date is 12/02/2024. Another date is 05/11/2023."

# Your code here
dates = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)

print(dates)  # Expected Output: ['12/02/2024', '05/11/2023']


"""Exercise 3: Validate Phone Numbers
Task: Check if the given phone number is valid (format: XXX-XXX-XXXX)."""

import re

phone = "987-654-3210"

# Your code here
if re.fullmatch(r'\d{3}-\d{3}-\d{4}', phone):
    print("Valid phone number")
else:
    print("Invalid phone number")
# Expected Output: Valid phone number

"""Exercise 4: Extract Hashtags from a Tweet
Task: Find all hashtags (#hashtag) in the tweet."""

import re

tweet = "Learning #Python is fun! #coding #100DaysOfCode"

# Your code here
hashtags = re.findall(r'#\w+', tweet)

print(hashtags)  # Expected Output: ['#Python', '#coding', '#100DaysOfCode']


"""Exercise 5: Extract Numbers from a String
Task: Extract all numbers (including decimal numbers)."""

import re

text = "There are 12 apples, 3.5 bananas, and 99 oranges."

# Your code here
numbers = re.findall(r'\d+\.?\d*', text)

print(numbers)  # Expected Output: ['12', '3.5', '99']


"""Exercise 6: Find Words Starting with "S"
Task: Find all words that start with the letter "S" or "s". """

import re

text = "Sam and Sally went to the store on Sunday."

# Your code here
words = re.findall(r'\b[Ss]\w+', text)

print(words)  # Expected Output: ['Sam', 'Sally', 'store', 'Sunday']


"""Exercise 7: Censor Bad Words
Task: Replace the bad words with ****. """

import re

text = "This is a damn good example. What the hell are you doing?"

# Your code here
censored_text = re.sub(r'\b(damn|hell)\b', '****', text, flags=re.IGNORECASE)

print(censored_text)  
# Expected Output: "This is a **** good example. What the **** are you doing?"


"""Exercise 8: Check Password Strength
Task: Validate a password with the following rules:

At least 8 characters long.
Contains at least one uppercase letter.
Contains at least one lowercase letter.
Contains at least one number.
Contains at least one special character (@, #, $, etc.). """

import re

password = "Secure@123"

# Your code here
if re.fullmatch(r'(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@#$%^&+=]).{8,}', password):
    print("Strong password")
else:
    print("Weak password")
# Expected Output: Strong password


"""Exercise 9: Remove Extra Spaces
Task: Remove multiple spaces and replace them with a single space."""

import re

text = "This   is   a    test   sentence."

# Your code here
clean_text = re.sub(r'\s+', ' ', text).strip()

print(clean_text)  
# Expected Output: "This is a test sentence."


"""Exercise 10: Extract Usernames from URLs
Task: Extract the username from a social media URL. """

import re

url = "https://www.instagram.com/john_doe123"

# Your code here
match = re.search(r'instagram\.com/(\w+)', url)

if match:
    print("Username:", match.group(1))
else:
    print("No username found")

# Expected Output: "Username: john_doe123" 


""" Exercise 11:Extract IP Addresses
Task: Extract all valid IP addresses (IPv4) from a given text.
A valid IPv4 address consists of four numbers (0-255) separated by dots (.). """

import re

text = "Valid IPs: 192.168.1.1, 255.255.255.255, 0.0.0.0. Invalid IPs: 999.999.999.999, 300.200.100.50"

# Regular expression for IPv4 address
pattern = r'\b(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\b'

# Find all valid IP addresses
ips = re.findall(pattern, text)

print("Extracted IPs:", ips)


"""Exercise 12:Validate Email Formats Strictly
Task: Validate email addresses with stricter conditions:

Starts with letters, numbers, . or _
Contains @
Domain contains letters and can have dots (.) but not consecutive
Ends with a valid extension (e.g., .com, .net, .org)"""
import re

emails = [
    "valid.email@example.com",
    "username_123@sub.domain.net",
    "invalid-email@.com",
    "@missingusername.com",
    "missing@domain",
    "double..dots@email.com"
]

# Regular expression for strict email validation
pattern = r'^[a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

for email in emails:
    if re.fullmatch(pattern, email):
        print(f"Valid: {email}")
    else:
        print(f"Invalid: {email}")
        

"""Exercise:12 Extract Words with 5 or More Letters
Task: Extract only words that have at least 5 characters."""

import re

text = "Python is amazing! I love programming and automation."

# Regular expression for words with 5+ letters
pattern = r'\b[a-zA-Z]{5,}\b'

# Find all matching words
words = re.findall(pattern, text)

print("Words with 5+ letters:", words)