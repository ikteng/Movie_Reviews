import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Set vocabulary size and maximum sequence length
vocab_size = 10000  # Only consider the top 10,000 words
max_length = 200  # Maximum length of reviews

# Load the IMDB dataset using Keras
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Function to preprocess the data by padding sequences
def preprocess_data(X):
    return pad_sequences(X, maxlen=max_length, padding='post', truncating='post')

# Preprocess the training and testing data
X_train_padded = preprocess_data(X_train)
X_test_padded = preprocess_data(X_test)

# Function to train the deep learning model
def train_model(X_train, y_train):
    # Build model
    model = Sequential([
        Embedding(vocab_size, 32, input_length=max_length),  # Embedding layer
        Flatten(),  # Flatten layer
        Dense(1, activation='sigmoid')  # Dense layer, sigmoid for binary classification
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, validation_split=0.2)  # Validation split of 20%
    
    return model

# Train the deep learning model
model = train_model(X_train_padded, y_train)

# Evaluate the deep learning model
def evaluate_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy

# Evaluate the model's accuracy
accuracy = evaluate_model(model, X_test_padded, y_test)
print(f'Accuracy: {accuracy}')

# Predict sentiment for a new review
def predict_sentiment(model, review):
    # The IMDB dataset returns integers instead of words, so we need to encode the new review accordingly.
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    # Fit the tokenizer on the IMDB dataset (here's a workaround to demonstrate prediction)
    tokenizer.fit_on_texts([' '.join(imdb.get_word_index().keys())])  # Dummy fit to initialize tokenizer

    # Preprocess the new review
    new_review_sequence = tokenizer.texts_to_sequences([review])
    new_review_padded = pad_sequences(new_review_sequence, maxlen=max_length, padding='post', truncating='post')

    # Make the prediction
    predicted_prob = model.predict(new_review_padded)[0][0]
    predicted_sentiment = 'positive' if predicted_prob >= 0.5 else 'negative'
    return predicted_sentiment

# Get user input for a new review
new_review = input("Enter movie review: ")
# Print the predicted sentiment for the new review
predicted_sentiment = predict_sentiment(model, new_review)
print(f'Predicted sentiment for the new review: {predicted_sentiment}')
