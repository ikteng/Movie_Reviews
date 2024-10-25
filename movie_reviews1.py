import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import string
import tensorflow as tf

# Load the IMDb dataset
def load_imdb_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    # Decode reviews back to text format
    word_index = tf.keras.datasets.imdb.get_word_index()
    index_word = {index + 3: word for word, index in word_index.items()}
    # Adjusting the index to account for padding and unknown tokens
    index_word[0], index_word[1], index_word[2] = '<PAD>', '<START>', '<UNK>'
    X_train = [' '.join([index_word[i] for i in review]) for review in X_train]
    X_test = [' '.join([index_word[i] for i in review]) for review in X_test]
    return X_train, X_test, y_train, y_test

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation]).lower()
    return text

# Function to train the model
def train_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Main method
if __name__ == "__main__":
    # Load IMDb data
    X_train, X_test, y_train, y_test = load_imdb_data()

    # Preprocess the text data
    X_train = [preprocess_text(review) for review in X_train]
    X_test = [preprocess_text(review) for review in X_test]
    
    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Print accuracy
    print(f'Accuracy: {accuracy}')
    # Print report
    print('Classification Report:\n', report)

    # Predict sentiment for a new review
    new_review = input("Enter movie review: ")
    new_review = preprocess_text(new_review)
    prediction = model.predict([new_review])[0]

    sentiment = "positive" if prediction == 1 else "negative"
    print(f'Predicted sentiment for the new review: {sentiment}')
