import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
import string

# Load the IMDB dataset using Keras
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Create a word index mapping
word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k: v + 3 for k, v in word_index.items()}  # Offset the index to reserve 0, 1, 2
word_index['<PAD>'] = 0  # Padding
word_index['<START>'] = 1  # Start token
word_index['<UNK>'] = 2  # Unknown token
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Preprocess the text data
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize
    words = text.split()
    return words  # return a list of words with no punctuation

# Preprocess the training and testing reviews
train_reviews = [decode_review(review) for review in train_data]
test_reviews = [decode_review(review) for review in test_data]

# Apply the preprocessing function
train_reviews = [preprocess_text(review) for review in train_reviews]
test_reviews = [preprocess_text(review) for review in test_reviews]

# Create a DataFrame for training and testing data
train_df = pd.DataFrame({'review': train_reviews, 'sentiment': train_labels})
test_df = pd.DataFrame({'review': test_reviews, 'sentiment': test_labels})

# Build a simple Naive Bayes classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities = {}

    def train(self, data):
        total_docs = len(data)

        # Calculate class probabilities
        sentiment_counts = Counter(data['sentiment'])
        for sentiment, count in sentiment_counts.items():
            self.class_probabilities[sentiment] = count / total_docs

        # Calculate word probabilities
        for sentiment, group in data.groupby('sentiment'):
            word_counts = Counter(word for review in group['review'] for word in review)
            total_words = sum(word_counts.values())
            self.word_probabilities[sentiment] = {word: count / total_words for word, count in word_counts.items()}

    def predict(self, review):
        scores = {sentiment: self.class_probabilities[sentiment] for sentiment in self.class_probabilities}

        for word in review:
            for sentiment in self.class_probabilities:
                if word in self.word_probabilities[sentiment]:
                    scores[sentiment] *= self.word_probabilities[sentiment][word]

        return max(scores, key=scores.get)

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier()
classifier.train(train_df)

# Make predictions
predictions = [classifier.predict(review) for review in test_df['review']]

# Evaluate the model
correct_predictions = sum(predictions[i] == test_df.iloc[i]['sentiment'] for i in range(len(test_df)))
accuracy = correct_predictions / len(test_df)

print(f'Accuracy: {accuracy:.4f}')

# Predict sentiment for a new review
new_review = input("Enter movie review: ")
new_review = preprocess_text(new_review)
prediction = classifier.predict(new_review)

# Map predictions to human-readable labels
sentiment_label = "Positive" if prediction == 1 else "Negative"
print(f'Predicted sentiment for the new review: {sentiment_label}')
