# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:30:11 2025

@author: Shantanu
"""

"""10. Web Scraping
Web scraping is the process of extracting data from websites. Python libraries like requests and BeautifulSoup are commonly used for this purpose in data analysis.

10.1. Fetching a Web Page
The requests library is used to send HTTP requests and retrieve web page content."""
import requests

url = "https://example.com"
response = requests.get(url)
print(f"Status code: {response.status_code}")  # Output: Status code: 200 (if successful)
print(f"First 100 characters: {response.text[:100]}")

"""10.2. Parsing HTML with BeautifulSoup
BeautifulSoup is used to parse HTML and extract specific elements."""
from bs4 import BeautifulSoup

html = "<html><body><h1>Welcome</h1><p>This is a test.</p></body></html>"
soup = BeautifulSoup(html, "html.parser")
print(f"Title: {soup.h1.text}")  # Output: Title: Welcome
print(f"Paragraph: {soup.p.text}")  # Output: Paragraph: This is a test.

"""10.3. Extracting Multiple Elements
You can find all instances of a tag, like all links or paragraphs."""
html = "<html><body><a href='link1'>Link1</a><a href='link2'>Link2</a></body></html>"
soup = BeautifulSoup(html, "html.parser")
links = soup.find_all("a")
for link in links:
    print(f"Link: {link['href']}")  # Output: Link: link1, Link: link2

"""10.4. Handling Tables
Web scraping often involves extracting data from tables."""
html = "<table><tr><td>Item1</td><td>10</td></tr><tr><td>Item2</td><td>20</td></tr></table>"
soup = BeautifulSoup(html, "html.parser")
table_data = [[cell.text for cell in row.find_all("td")] for row in soup.find_all("tr")]
print(f"Table data: {table_data}")  # Output: Table data: [['Item1', '10'], ['Item2', '20']]

"""10.5. Error Handling
Handle exceptions like network errors or missing elements."""
try:
    response = requests.get("https://nonexistent-url.com")
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""10. Web Scraping Exercises
Exercise 1: Fetch Web Page Title
Write a program that fetches the title of a web page given a URL."""
import requests
from bs4 import BeautifulSoup

url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.text if soup.title else "No title found"
    print(f"Page title: {title}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 2: Extract All Paragraphs
Write a program that extracts and prints all paragraph texts from a given URL."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.text.strip() for p in soup.find_all("p")]
    print("Paragraphs:")
    for i, para in enumerate(paragraphs, 1):
        print(f"{i}. {para}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 3: Get All Links
Write a program that fetches all hyperlinks (href) from a web page."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    print("Links:")
    for i, link in enumerate(links, 1):
        print(f"{i}. {link}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 4: Scrape Table Data
Write a program that extracts data from the first table on a web page."""
url = input("Enter URL with a table: ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table:
        table_data = [[cell.text.strip() for cell in row.find_all(["td", "th"])] for row in table.find_all("tr")]
        print("Table data:")
        for row in table_data:
            print(row)
    else:
        print("No table found")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 5: Count Headings
Write a program that counts the number of h1, h2, and h3 headings on a web page."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    h1_count = len(soup.find_all("h1"))
    h2_count = len(soup.find_all("h2"))
    h3_count = len(soup.find_all("h3"))
    print(f"H1 headings: {h1_count}")
    print(f"H2 headings: {h2_count}")
    print(f"H3 headings: {h3_count}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 6: Extract Image URLs
Write a program that fetches all image URLs (src) from a web page."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    images = [img["src"] for img in soup.find_all("img", src=True)]
    print("Image URLs:")
    for i, img in enumerate(images, 1):
        print(f"{i}. {img}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 7: Scrape Specific Class
Write a program that extracts text from all elements with a specific class name."""
url = input("Enter URL (e.g., https://example.com): ")
class_name = input("Enter class name to scrape: ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    elements = [elem.text.strip() for elem in soup.find_all(class_=class_name)]
    print(f"Elements with class '{class_name}':")
    for i, elem in enumerate(elements, 1):
        print(f"{i}. {elem}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 8: Scrape List Items
Write a program that extracts all list items (<li>) from a web page."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    items = [li.text.strip() for li in soup.find_all("li")]
    print("List items:")
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 9: Extract Meta Tags
Write a program that fetches all meta tag descriptions from a web page."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    metas = [meta["content"] for meta in soup.find_all("meta", content=True)]
    print("Meta tag contents:")
    for i, meta in enumerate(metas, 1):
        print(f"{i}. {meta}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 10: Scrape Product Prices
Write a program that extracts product names and prices from a web page (assuming elements with class 'product' and 'price')."""
url = input("Enter URL with products: ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = soup.find_all(class_="product")
    print("Products and prices:")
    for i, prod in enumerate(products, 1):
        name = prod.find(class_="product-name")
        price = prod.find(class_="price")
        name_text = name.text.strip() if name else "No name"
        price_text = price.text.strip() if price else "No price"
        print(f"{i}. {name_text}: {price_text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 11: Scrape and Save to List
Write a program that scrapes all paragraph texts and stores them in a list."""
url = input("Enter URL (e.g., https://example.com): ")
paragraphs_list = []
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs_list = [p.text.strip() for p in soup.find_all("p")]
    print(f"Stored {len(paragraphs_list)} paragraphs:")
    for i, para in enumerate(paragraphs_list, 1):
        print(f"{i}. {para}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

"""Exercise 12: Scrape and Count Words
Write a program that scrapes all text from a web page and counts the frequency of each word."""
url = input("Enter URL (e.g., https://example.com): ")
try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text().lower()
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    print("Word frequencies:")
    for word, freq in sorted(word_freq.items()):
        if freq > 1:  # Show only words appearing more than once
            print(f"{word}: {freq}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")