# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:22:38 2025
Updated for repository consistency and additional concepts.

@author: Shantanu
"""

"""7. Regular Expressions (RegEx)
Regular expressions are powerful tools for pattern matching and text processing, widely used in data analysis for tasks like data validation and extraction.

7.1. Basic Search
re.search() finds the first occurrence of a pattern in a string."""
import re

text = "A cat and a rat can't be friends."
match = re.search(r"cat", text)
print(f"Match found: {match.group() if match else 'None'}")  # Output: Match found: cat

"""7.2. Wildcards
The dot (.) matches any single character except newline."""
match = re.search(r" .at ", text)
print(f"Wildcard match: {match.group() if match else 'None'}")  # Output: Wildcard match:  cat 

"""7.3. Character Classes
Square brackets [] define a set of characters to match."""
pattern = r"M[ae][iy]er"
text = "Names: Maier, Mayer, Meier, Meyer"
matches = re.findall(pattern, text)
print(f"Character class matches: {matches}")  # Output: Character class matches: ['Maier', 'Mayer', 'Meier', 'Meyer']

"""7.4. Finding All Matches
re.findall() returns all non-overlapping matches as a list."""
text = "The rain in Spain"
matches = re.findall(r"ai", text)
print(f"All matches: {matches}")  # Output: All matches: ['ai', 'ai']

"""7.5. Special Sequences
Common sequences include \d (digits), \w (word characters), \s (whitespace)."""
text = "Sharat is 35 and Pavan is 28"
ages = re.findall(r'\d{1,2}', text)
print(f"Ages: {ages}")  # Output: Ages: ['35', '28']

"""7.6. Quantifiers
Quantifiers specify how many times a pattern should match."""
text = "Phone: 98765"
numbers = re.findall(r'\d+', text)
print(f"Numbers: {numbers}")  # Output: Numbers: ['98765']

"""7.7. Substituting Text
re.sub() replaces pattern matches with a specified string."""
phone = "2004-959-559 # This is Phone Number"
cleaned = re.sub(r'#.*$', "", phone).strip()
print(f"Cleaned phone: {cleaned}")  # Output: Cleaned phone: 2004-959-559

"""7.8. Splitting Strings
re.split() splits a string based on a pattern."""
text = "The rain in Spain"
words = re.split(r"\s+", text)
print(f"Split words: {words}")  # Output: Split words: ['The', 'rain', 'in', 'Spain']

"""7.9. Precompiling Patterns
re.compile() improves performance for repeated regex use."""
regex = re.compile(r"\n")
text = "Hello\nWorld"
cleaned = regex.sub("", text)
print(f"Cleaned text: {cleaned}")  # Output: Cleaned text: HelloWorld

"""7.10. Named Capture Groups
Use (?P<name>...) to name captured groups for clarity."""
text = "Email: user@domain.com"
match = re.search(r"(?P<user>\w+)@(?P<domain>\w+\.\w+)", text)
if match:
    print(f"User: {match.group('user')}, Domain: {match.group('domain')}")  # Output: User: user, Domain: domain.com

"""7.11. Lookaheads and Lookbehinds
Lookaheads (?=...) and lookbehinds (?<=...) match patterns based on context."""
text = "price: $100, discount: $20"
matches = re.findall(r'(?<=price: \$)\d+', text)
print(f"Price values: {matches}")  # Output: Price values: ['100']

"""7.12. Regex Flags
Flags like re.IGNORECASE and re.MULTILINE modify regex behavior."""
text = "CAT and cat are similar"
matches = re.findall(r"cat", text, re.IGNORECASE)
print(f"Case-insensitive matches: {matches}")  # Output: Case-insensitive matches: ['CAT', 'cat']

"""7. Regular Expression Exercises
Exercise 1: Extract Email Addresses
Extract all email addresses from a text."""
text = "Contact: support@example.com, sales@example.org, info@company.net"
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
print(f"Emails: {emails}")  # Output: Emails: ['support@example.com', 'sales@example.org', 'info@company.net']

"""Exercise 2: Extract Dates
Find all dates in the format DD/MM/YYYY."""
text = "Events on 12/02/2024 and 05/11/2023."
dates = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
print(f"Dates: {dates}")  # Output: Dates: ['12/02/2024', '05/11/2023']

"""Exercise 3: Validate Phone Numbers
Check if a phone number matches the format XXX-XXX-XXXX."""
phone = input("Enter phone number (e.g., 123-456-7890): ")
if re.fullmatch(r'\d{3}-\d{3}-\d{4}', phone):
    print("Valid phone number")
else:
    print("Invalid phone number")

"""Exercise 4: Extract Hashtags
Find all hashtags in a tweet."""
tweet = "Learning #Python is fun! #coding #100DaysOfCode"
hashtags = re.findall(r'#\w+', tweet)
print(f"Hashtags: {hashtags}")  # Output: Hashtags: ['#Python', '#coding', '#100DaysOfCode']

"""Exercise 5: Validate Email Strictly
Validate an email with strict rules: letters/numbers/._ before @, valid domain, and extension."""
email = input("Enter email: ")
pattern = r'^[a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
if re.fullmatch(pattern, email):
    print("Valid email")
else:
    print("Invalid email")

"""Exercise 6: Extract Words Starting with 'S'
Find all words starting with 'S' or 's'."""
text = "Sam and Sally went to the store on Sunday."
words = re.findall(r'\b[Ss]\w+', text)
print(f"S-words: {words}")  # Output: S-words: ['Sam', 'Sally', 'store', 'Sunday']

"""Exercise 7: Censor Bad Words
Replace bad words with '****'."""
text = "This is a damn good example. What the hell are you doing?"
censored = re.sub(r'\b(damn|hell)\b', '****', text, flags=re.IGNORECASE)
print(f"Censored: {censored}")  # Output: Censored: This is a **** good example. What the **** are you doing?

"""Exercise 8: Check Password Strength
Validate a password with rules: 8+ chars, 1 uppercase, 1 lowercase, 1 number, 1 special char."""
password = input("Enter password: ")
pattern = r'(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@#$%^&+=]).{8,}'
if re.fullmatch(pattern, password):
    print("Strong password")
else:
    print("Weak password")

"""Exercise 9: Remove Extra Spaces
Replace multiple spaces with a single space."""
text = "This   is   a    test   sentence."
cleaned = re.sub(r'\s+', ' ', text).strip()
print(f"Cleaned: {cleaned}")  # Output: Cleaned: This is a test sentence.

"""Exercise 10: Extract IP Addresses
Extract valid IPv4 addresses from a text."""
text = "IPs: 192.168.1.1, 255.255.255.255, 999.999.999.999"
pattern = r'\b(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\b'
ips = re.findall(pattern, text)
print(f"Valid IPs: {ips}")  # Output: Valid IPs: ['192.168.1.1', '255.255.255.255']

"""Exercise 11: Extract Named Groups
Extract username and domain from an email using named capture groups."""
text = "Contact: user123@domain.com"
match = re.search(r'(?P<user>\w+)@(?P<domain>\w+\.\w+)', text)
if match:
    print(f"Username: {match.group('user')}, Domain: {match.group('domain')}")  # Output: Username: user123, Domain: domain.com

"""Exercise 12: Extract Prices with Lookbehind
Extract numbers following a dollar sign using lookbehind."""
text = "Items: $50, $100, discount: $20"
prices = re.findall(r'(?<=\$)\d+', text)
print(f"Prices: {prices}")  # Output: Prices: ['50', '100', '20']

"""Exercise 13: Case-Insensitive Search
Find all occurrences of 'data' (case-insensitive) in a text."""
text = "Data, DATA, and dAtA are common."
matches = re.findall(r'data', text, re.IGNORECASE)
print(f"Matches: {matches}")  # Output: Matches: ['Data', 'DATA', 'dAtA']

"""Exercise 14: Extract Long Words
Extract words with 5 or more letters."""
text = "Python is amazing! I love programming and automation."
words = re.findall(r'\b[a-zA-Z]{5,}\b', text)
print(f"Long words: {words}")  # Output: Long words: ['Python', 'amazing', 'programming', 'automation']

"""Exercise 15: Split Log Entries
Split a log file string into entries based on timestamps."""
log = "2024-01-01 10:00: Error 2024-01-01 10:01: Success"
entries = re.split(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}: ', log)
print(f"Log entries: {entries[1:]}")  # Output: Log entries: ['Error ', 'Success']
