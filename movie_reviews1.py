# imported libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string

# function to load the data from the file_path as DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# function to preprocess text
def preprocess_text(text, stop_words):
    # Remove punctuation and convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation]).lower()
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# function to train model
def train_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# function to evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# main method
if __name__ == "__main__":
    # get dataframe of csv file
    file_path = 'movie_reviews\IMDB Dataset.csv'
    df = load_data(file_path)

    # Preprocess the text data
    stop_words = set(ENGLISH_STOP_WORDS) # define the stopwords
    # preprocesses each text review in the 'review' column of the DataFrame by applying the preprocess_text function, and the preprocessed reviews are then stored back in the 'review' column of the DataFrame. 
    df['review'] = df['review'].apply(lambda x: preprocess_text(x, stop_words)) 

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # print accuracy
    print(f'Accuracy: {accuracy}')
    # print report
    print('Classification Report:\n', report)

    # Predict sentiment for a new review
    new_review = input("Enter movie review: ")
    new_review = preprocess_text(new_review, stop_words)
    prediction = model.predict([new_review])[0]

    print(f'Predicted sentiment for the new review: {prediction}')
