# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 21:00:11 2025

@author: Shantanu
"""

"""21. API Integration
This script introduces fundamental API integration concepts using Python. It covers making HTTP requests, handling responses, authenticating with APIs, and processing data, with practical examples using a mock API and the `sales.csv` dataset from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: requests, json, pandas, time
- Dataset: `sales.csv` from the `data/` directory
- Mock API: jsonplaceholder.typicode.com (public testing API)
"""

# 21.1. Setup
import requests
import json
import pandas as pd
import time
import os

# Create output directory
output_dir = '../data/output'
os.makedirs(output_dir, exist_ok=True)

# Load sample dataset
sales_df = pd.read_csv('../data/sales.csv')
print('Sales Dataset Head:')
print(sales_df.head())

"""21.2. Introduction to API Integration
API integration enables applications to communicate and exchange data. Key concepts:
- HTTP Requests: GET, POST, PUT, DELETE for interacting with APIs.
- JSON Handling: Parsing and formatting API responses.
- Authentication: Using API keys or tokens.
- Rate Limiting: Managing request frequency.
This script demonstrates API interactions using the jsonplaceholder mock API."""

"""21.3. Making a GET Request
Fetch data from an API endpoint using a GET request."""
# Fetch posts from jsonplaceholder
response = requests.get('https://jsonplaceholder.typicode.com/posts')
posts = response.json()
print('First Post Data:')
print(posts[0])

# Save response to file
output_path = os.path.join(output_dir, 'posts.json')
with open(output_path, 'w') as f:
    json.dump(posts[:5], f, indent=4)
print(f'Saved first 5 posts to {output_path}')

"""21.4. Handling API Responses
Parse and process JSON responses."""
# Convert posts to DataFrame
posts_df = pd.DataFrame(posts)
print('Posts DataFrame Head:')
print(posts_df.head())

# Filter posts by userId
user_posts = posts_df[posts_df['userId'] == 1]
print('Posts by User ID 1:')
print(user_posts.head())

"""21.5. Making a POST Request
Send data to an API endpoint."""
# Create a new post
new_post = {
    'title': 'Sample Post',
    'body': 'This is a sample post created via API.',
    'userId': 1
}
post_response = requests.post('https://jsonplaceholder.typicode.com/posts', json=new_post)
print('POST Response Status:', post_response.status_code)
print('POST Response Data:', post_response.json())

"""21.6. Authentication
Use API keys or tokens for authenticated requests (mock example)."""
# Mock API key header
headers = {'Authorization': 'Bearer mock_api_key'}
auth_response = requests.get('https://jsonplaceholder.typicode.com/users', headers=headers)
users = auth_response.json()
print('First User Data (Authenticated):')
print(users[0])

"""21.7. Rate Limiting
Implement delays to respect API rate limits."""
# Fetch users with delay
user_data = []
for user_id in range(1, 4):
    response = requests.get(f'https://jsonplaceholder.typicode.com/users/{user_id}')
    user_data.append(response.json())
    time.sleep(1)  # 1-second delay
print('Fetched Users with Rate Limiting:', [user['name'] for user in user_data])

"""21.8. Integrating API with Local Data
Combine API data with local dataset."""
# Merge sales data with mock user data
users_df = pd.DataFrame(user_data)
sales_df['user_id'] = (sales_df.index % len(users_df) + 1)  # Mock user_id
merged_df = sales_df.merge(users_df[['id', 'name']], left_on='user_id', right_on='id')
output_merged = os.path.join(output_dir, 'sales_with_users.csv')
merged_df.to_csv(output_merged, index=False)
print(f'Saved merged data to {output_merged}')

"""21.9. Error Handling
Handle API errors gracefully."""
def safe_get_request(url):
    """Safely make a GET request with error handling."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {url}: {e}')
        return None

# Test error handling with invalid URL
data = safe_get_request('https://jsonplaceholder.typicode.com/invalid')
print('Invalid Endpoint Response:', data)

"""21.10. API Integration Exercises"""

"""Exercise 1: Fetch Posts
Fetch posts from jsonplaceholder and print the first post."""
def exercise_1():
    response = requests.get('https://jsonplaceholder.typicode.com/posts')
    posts = response.json()
    print('First Post:', posts[0])

exercise_1()

"""Exercise 2: Save JSON
Fetch comments and save the first 5 to `output/comments.json`."""
def exercise_2():
    response = requests.get('https://jsonplaceholder.typicode.com/comments')
    comments = response.json()
    output_path = os.path.join('../data/output', 'comments.json')
    with open(output_path, 'w') as f:
        json.dump(comments[:5], f, indent=4)
    print(f'Saved comments to {output_path}')

exercise_2()

"""Exercise 3: Convert to DataFrame
Fetch users and convert to a pandas DataFrame."""
def exercise_3():
    response = requests.get('https://jsonplaceholder.typicode.com/users')
    users = response.json()
    users_df = pd.DataFrame(users)
    print('Users DataFrame Head:')
    print(users_df.head())

exercise_3()

"""Exercise 4: Filter Data
Fetch posts and filter for posts by userId 2."""
def exercise_4():
    response = requests.get('https://jsonplaceholder.typicode.com/posts')
    posts = response.json()
    posts_df = pd.DataFrame(posts)
    user2_posts = posts_df[posts_df['userId'] == 2]
    print('Posts by User ID 2:')
    print(user2_posts.head())

exercise_4()

"""Exercise 5: POST Request
Create a new comment via POST and print the response."""
def exercise_5():
    new_comment = {
        'postId': 1,
        'name': 'Sample User',
        'email': 'sample@example.com',
        'body': 'This is a test comment.'
    }
    response = requests.post('https://jsonplaceholder.typicode.com/comments', json=new_comment)
    print('POST Comment Response:', response.json())

exercise_5()

"""Exercise 6: Authenticated Request
Make a GET request to fetch todos with a mock API key header."""
def exercise_6():
    headers = {'Authorization': 'Bearer mock_api_key'}
    response = requests.get('https://jsonplaceholder.typicode.com/todos', headers=headers)
    todos = response.json()
    print('First Todo (Authenticated):', todos[0])

exercise_6()

"""Exercise 7: Rate Limiting
Fetch the first 3 albums with a 1-second delay between requests."""
def exercise_7():
    albums = []
    for album_id in range(1, 4):
        response = requests.get(f'https://jsonplaceholder.typicode.com/albums/{album_id}')
        albums.append(response.json())
        time.sleep(1)
    print('Fetched Albums:', [album['title'] for album in albums])

exercise_7()

"""Exercise 8: Merge API and Local Data
Merge `sales.csv` with user names from the API."""
def exercise_8():
    response = requests.get('https://jsonplaceholder.typicode.com/users')
    users = response.json()
    users_df = pd.DataFrame(users)[['id', 'name']]
    sales = pd.read_csv('../data/sales.csv')
    sales['user_id'] = (sales.index % len(users_df) + 1)
    merged = sales.merge(users_df, left_on='user_id', right_on='id')
    output_path = os.path.join('../data/output', 'sales_users_merged.csv')
    merged.to_csv(output_path, index=False)
    print(f'Saved merged data to {output_path}')

exercise_8()

"""Exercise 9: Error Handling
Make a safe GET request to an invalid endpoint."""
def exercise_9():
    data = safe_get_request('https://jsonplaceholder.typicode.com/invalid')
    print('Invalid Endpoint Response:', data)

exercise_9()

"""Exercise 10: Fetch Specific Post
Fetch post with ID 10 and print its title."""
def exercise_10():
    response = requests.get('https://jsonplaceholder.typicode.com/posts/10')
    post = response.json()
    print('Post Title:', post['title'])

exercise_10()

"""Exercise 11: Count API Data
Fetch todos and count completed tasks."""
def exercise_11():
    response = requests.get('https://jsonplaceholder.typicode.com/todos')
    todos = response.json()
    completed_count = sum(1 for todo in todos if todo['completed'])
    print('Completed Todos:', completed_count)

exercise_11()

"""Exercise 12: Save API Data
Fetch photos and save the first 10 to `output/photos.json`."""
def exercise_12():
    response = requests.get('https://jsonplaceholder.typicode.com/photos')
    photos = response.json()
    output_path = os.path.join('../data/output', 'photos.json')
    with open(output_path, 'w') as f:
        json.dump(photos[:10], f, indent=4)
    print(f'Saved photos to {output_path}')

exercise_12()

"""Exercise 13: POST Multiple Data
Send 2 new posts via POST and print their IDs."""
def exercise_13():
    posts = [
        {'title': 'Post 1', 'body': 'Body 1', 'userId': 1},
        {'title': 'Post 2', 'body': 'Body 2', 'userId': 1}
    ]
    for post in posts:
        response = requests.post('https://jsonplaceholder.typicode.com/posts', json=post)
        print('Created Post ID:', response.json()['id'])

exercise_13()

"""Exercise 14: Filter and Save
Fetch comments for postId 1 and save to `output/post1_comments.csv`."""
def exercise_14():
    response = requests.get('https://jsonplaceholder.typicode.com/comments?postId=1')
    comments = response.json()
    comments_df = pd.DataFrame(comments)
    output_path = os.path.join('../data/output', 'post1_comments.csv')
    comments_df.to_csv(output_path, index=False)
    print(f'Saved comments to {output_path}')

exercise_14()

"""Exercise 15: API Data Summary
Fetch users and save a summary (name, email) to `output/users_summary.csv`."""
def exercise_15():
    response = requests.get('https://jsonplaceholder.typicode.com/users')
    users = response.json()
    users_df = pd.DataFrame(users)[['name', 'email']]
    output_path = os.path.join('../data/output', 'users_summary.csv')
    users_df.to_csv(output_path, index=False)
    print(f'Saved user summary to {output_path}')

exercise_15()