# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:01:54 2025

@author: Shantanu
"""

"""18. NLP Basics
This script introduces fundamental natural language processing (NLP) concepts using NLTK and spaCy. It covers text preprocessing, tokenization, sentiment analysis, and named entity recognition, using the `text_data.csv` dataset from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, nltk, spacy, textblob, scikit-learn, matplotlib, seaborn
- Dataset: `text_data.csv` from the `data/` directory
- NLTK data: punkt, stopwords, vader_lexicon
- spaCy model: en_core_web_sm
"""

# 18.1. Setup
import pandas as pd
import nltk
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load sample dataset
text_df = pd.read_csv('../data/text_data.csv')
print('Text Dataset Head:')
print(text_df.head())

"""18.2. Introduction to NLP
Natural Language Processing (NLP) enables computers to understand and process human language. Key tasks:
- Text Preprocessing: Cleaning and normalizing text.
- Tokenization: Splitting text into words or sentences.
- Sentiment Analysis: Determining emotional tone.
- Named Entity Recognition (NER): Identifying entities like names or places.
This script uses NLTK and spaCy for basic NLP tasks."""

"""18.3. Text Preprocessing
Clean and normalize text by removing special characters, converting to lowercase, and removing stopwords."""
def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stop_words.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to text column (assuming column named 'text')
text_df['cleaned_text'] = text_df['text'].apply(preprocess_text)
print('Preprocessed Text Data Head:')
print(text_df.head())

"""18.4. Tokenization
Extract tokens (words or sentences) from text using NLTK."""
# Word tokenization
sample_text = nltk.word_tokenize(text_df['text'].iloc[0])
print('Sample Word Tokens:\n', sample_text[:10])

# Sentence tokenization
sample_sentences = nltk.sent_tokenize(text_df['text'].iloc[0])
print('\nSample Sentences:\n', sample_sentences)

"""18.5. Part-of-Speech Tagging
Assign POS tags-of-speech to tokens using spaCy."""
# POS tagging with spaCy
doc = nlp(text_df['text'].iloc[0])
pos_tags = [(token.text, token.pos_) for token in doc]
print('\nSample POS Tags:\n', pos_tags[:10])

"""18.6. Lemmatization
Reduce words to their base form using spaCy."""
def lemmatize(text):
    """Lemmatize text using spaCy."""
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Apply lemmatization
text_df['lemmatized_text'] = text_df['cleaned_text'].apply(lemmatize)
print('\nLemmatized Text Data Head:')
print(text_df.head())

"""18.7. Sentiment Analysis
Determine the sentiment of text using TextBlob."""
def get_sentiment(text):
    """Get sentiment polarity using TextBlob."""
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
text_df['sentiment'] = text_df['cleaned_text'].apply(get_sentiment)
print('\nText Data with Sentiment Head:')
print(text_df[['text', 'sentiment']].head())

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(text_df['sentiment'], bins=20)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.show()

"""18.8. Named Entity Recognition
Identify entities like names and organizations using spaCy."""
# Enable spaCy's NER component
nlp_ner = spacy.load('en_core_web_sm')
doc_ner = nlp_ner(text_df['text'].iloc[0])
entities = [(ent.text, ent.label_) for ent in doc_ner.ents]
print('\nNamed Entities:\n', entities)

"""18.9. Text Vectorization
Convert text to numerical features using TF-IDF for machine learning."""
# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(text_df['cleaned_text'])

# Assuming a 'label' column for classification (e.g., positive/negative)
y = text_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""18.10. Text Classification
Train a Naive Bayes classifier for text classification."""
# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'\nNaive Bayes Classifier Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Naive Bayes Classifier: Confusion Matrix')
plt.show()

"""18.11. Saving and Loading Models
Save the vectorizer and classifier for reuse."""
# Save models
joblib.dump(vectorizer, '../scripts/nlp_utils_vectorizer.pkl')
joblib.dump(nb_model, '../scripts/nlp_utils_nb_model.pkl')

# Load models
loaded_vectorizer = joblib.load('../scripts/nlp_utils_vectorizer.pkl')
loaded_nb = joblib.load('../scripts/nlp_utils_nb_model.pkl')
print('\nLoaded Naive Bayes Model Parameters:', loaded_nb.get_params())

"""18. NLP Basics Exercises"""

"""Exercise 1: Load and Inspect Dataset
Load `text_data.csv` and print its first 5 rows and data types."""
def exercise_1():
    df = pd.read_csv('../data/text_data.csv')
    print('First 5 rows:')
    print(df.head())
    print('\nData Types:')
    print(df.dtypes)

exercise_1()

"""Exercise 2: Preprocess Text
Apply the preprocess_text function to the `text` column and print the first 5 cleaned texts."""
def exercise_2():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print('First 5 Cleaned Texts:')
    print(df['cleaned_text'].head())

exercise_2()

"""Exercise 3: Word Tokenization
Tokenize the first text in `text_data.csv` into words and print the first 10 tokens."""
def exercise_3():
    df = pd.read_csv('../data/text_data.csv')
    tokens = nltk.word_tokenize(df['text'].iloc[0])
    print('First 10 Word Tokens:', tokens[:10])

exercise_3()

"""Exercise 4: Sentence Tokenization
Tokenize the first text in `text_data.csv` into sentences and print them."""
def exercise_4():
    df = pd.read_csv('../data/text_data.csv')
    sentences = nltk.sent_tokenize(df['text'].iloc[0])
    print('Sentences:', sentences)

exercise_4()

"""Exercise 5: POS Tagging
Perform POS tagging on the first cleaned text using spaCy and print the first 10 tags."""
def exercise_5():
    df = pd.read_csv('../data/text_data.csv')
    cleaned_text = preprocess_text(df['text'].iloc[0])
    doc = nlp(cleaned_text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    print('First 10 POS Tags:', pos_tags[:10])

exercise_5()

"""Exercise 6: Lemmatization
Lemmatize the first cleaned text and print the result."""
def exercise_6():
    df = pd.read_csv('../data/text_data.csv')
    cleaned_text = preprocess_text(df['text'].iloc[0])
    lemmatized = lemmatize(cleaned_text)
    print('Lemmatized Text:', lemmatized)

exercise_6()

"""Exercise 7: Sentiment Analysis
Compute sentiment for the first 5 texts and print the results."""
def exercise_7():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    sentiments = df['cleaned_text'].head().apply(get_sentiment)
    print('Sentiments for First 5 Texts:')
    print(sentiments)

exercise_7()

"""Exercise 8: Named Entity Recognition
Perform NER on the first text and print the entities."""
def exercise_8():
    df = pd.read_csv('../data/text_data.csv')
    doc = nlp_ner(df['text'].iloc[0])
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print('Named Entities:', entities)

exercise_8()

"""Exercise 9: TF-IDF Vectorization
Vectorize the cleaned text using TF-IDF and print the shape of the resulting matrix."""
def exercise_9():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    print('TF-IDF Matrix Shape:', X.shape)

exercise_9()

"""Exercise 10: Train Naive Bayes
Train a Naive Bayes classifier on the TF-IDF vectors and compute accuracy."""
def exercise_10():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

exercise_10()

"""Exercise 11: Confusion Matrix
Plot the confusion matrix for the Naive Bayes classifier."""
def exercise_11():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

exercise_11()

"""Exercise 12: Save Models
Save the TF-IDF vectorizer and Naive Bayes model to `nlp_utils` files."""
def exercise_12():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    joblib.dump(vectorizer, '../scripts/nlp_utils_ex12_vectorizer.pkl')
    joblib.dump(nb_model, '../scripts/nlp_utils_ex12_nb_model.pkl')
    print('Models saved successfully')

exercise_12()

"""Exercise 13: Predict New Text
Use the Naive Bayes model to predict the label of a new text sample."""
def exercise_13():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    new_text = preprocess_text("This is a great product!")
    new_vector = vectorizer.transform([new_text])
    prediction = nb_model.predict(new_vector)
    print('Prediction for new text:', prediction)

exercise_13()

"""Exercise 14: Stopwords Customization
Add custom stopwords to the preprocessing function and reprocess the first text."""
def exercise_14():
    df = pd.read_csv('../data/text_data.csv')
    custom_stopwords = set(nltk.corpus.stopwords.words('english')).union({'product', 'item'})
    def custom_preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in custom_stopwords]
        return ' '.join(tokens)
    cleaned_text = custom_preprocess(df['text'].iloc[0])
    print('Custom Preprocessed Text:', cleaned_text)

exercise_14()

"""Exercise 15: Sentiment Distribution
Plot the sentiment distribution for texts with positive vs. negative labels."""
def exercise_15():
    df = pd.read_csv('../data/text_data.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentiment', hue='label', bins=20)
    plt.title('Sentiment Distribution by Label')
    plt.xlabel('Sentiment Polarity')
    plt.show()

exercise_15()