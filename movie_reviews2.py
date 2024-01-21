import pandas as pd
import string
from collections import Counter

# Load the dataset
df = pd.read_csv('movie_reviews\IMDB Dataset.csv')

# Preprocess the text data
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize
    words = text.split()
    return words # return a list of words with no punctuation

df['review'] = df['review'].apply(preprocess_text)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(df)) # find out the size of 80% of data
train_data = df.iloc[:train_size] # 80% of data used for training
test_data = df.iloc[train_size:] # remaining 20% used for testing

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
classifier.train(train_data)

# Make predictions
predictions = [classifier.predict(review) for review in test_data['review']]

# Evaluate the model
correct_predictions = sum(predictions[i] == test_data.iloc[i]['sentiment'] for i in range(len(test_data)))
accuracy = correct_predictions / len(test_data)

print(f'Accuracy: {accuracy:.4f}')

# Predict sentiment for a new review
new_review = input("Enter movie review:")
new_review = preprocess_text(new_review)
prediction = classifier.predict(new_review)

print(f'Predicted sentiment for the new review: {prediction}')
