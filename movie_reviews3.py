# data manipulation
import pandas as pd
# split dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# build and train model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# Set vocabulary size and maximum sequence length
vocab_size = 10000
max_length = 200

# initialize tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>') # oov = out of vocabulary

# initialize label encoder
label_encoder = LabelEncoder()

# Function to load the data from the file_path as DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to Remove punctuation and convert to lowercase
def preprocess_text(text):
    # Initialize an empty string to store the processed text
    processed_text = ""

    # Loop through each character in the input text
    for char in text:
        # Check if the character is not in string.punctuation
        if char not in string.punctuation:
            # Append the lowercase version of the character to the processed text
            processed_text += char.lower()

    return processed_text

# Function to train deep learning model
def train_model(X_train, y_train):
    """
    X: reviews
    y: positive or negative
    """
    # update the internal vocabulary based on word frequency in the provided texts
    tokenizer.fit_on_texts(X_train)

    # convert reviews to integers which corresponds to the index of the word in the tokenizer's word index
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    # padded/truncated to ensure all sequences are the same length
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')

    # transform y_train into numerical format for training
    y_train_encoded = label_encoder.fit_transform(y_train)

    # build model
    model = Sequential([
        Embedding(vocab_size, 32, input_length=max_length), # embedding layer
        Flatten(), # flatten layer
        Dense(1, activation='sigmoid') # dense layer, sigmoid for output values between 0 & 1
    ])

    # compile model, specifies optimizer, loss function and metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # train the model
    model.fit(X_train_padded, y_train_encoded, epochs=5, validation_split=0.2) # validation split of 20% is used for monitoring model's performance on a validation set

    return model, tokenizer, label_encoder

# Function to evaluate deep learning model
def evaluate_model(model, label_encoder, X_test, y_test):
    # convert review into integers based on vocabulary learned during training
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    # padded/truncated to ensure all sequences are the same length
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')
    # converts label into numerical values
    y_test_encoded = label_encoder.transform(y_test)
    # compute accuracy
    _, accuracy = model.evaluate(X_test_padded, y_test_encoded)
    return accuracy

# Main method
if __name__ == "__main__":
    # load data from csv file
    file_path = 'movie_reviews\IMDB Dataset.csv'
    df = load_data(file_path)

    # Preprocess the text data
    df['review'] = df['review'].apply(preprocess_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Train the deep learning model
    model, tokenizer, label_encoder = train_model(X_train, y_train)

    # Evaluate the deep learning model
    accuracy = evaluate_model(model, label_encoder, X_test, y_test)
    # Print accuracy
    print(f'Accuracy: {accuracy}')

    # Predict sentiment for a new review
    new_review = input("Enter movie review: ")
    # preprocess the new review to remove punctuation and convert to lowercase
    new_review = preprocess_text(new_review)

    # Tokenize the new review
    new_review_sequence = tokenizer.texts_to_sequences([new_review])
    # padded/truncated the new reivew
    new_review_padded = pad_sequences(new_review_sequence, maxlen=max_length, padding='post', truncating='post')

    # Make the prediction
    predicted_prob = model.predict(new_review_padded)[0][0]

    # Convert the predicted probability to sentiment label
    predicted_sentiment = 'positive' if predicted_prob >= 0.5 else 'negative'
    # print the prediction
    print(f'Predicted sentiment for the new review: {predicted_sentiment}')

