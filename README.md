# Movie_Reviews

Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

## Movie Reviews 1
This Python Code using a Naive Bayes classification on a dataset of movie reviews to perform sentiment analysis. 

### Imported Libraries
* pandas
* sklearn
* string

### Load Data
Reads a CSV file containing movie reviews and their sentiments (positive or negative) into a Pandas DataFrame

### Preprocess Text
* defines a function to clearn and preprocess the text data
* remove punctuations, converts text to lowercase, tokenizes the text, and remove English stopwords

### Train Model
* use the CountVectorizer from sklearn.feature_extracttion.text to convert the text data into a bag-of-words representation
* create a pipelines that includes CountVectorizer and a Multinomial Naive Bayes classifier
* trains the model on the training set ('X_train' and 'y_train')

### Evaluate Model
* use the trained model to make predictions on the test set ('X_test')
* calculate and print the accuracy and classification report of the model

### Main Method
* loads the movie reviews dataset
* preprocesses the text data in the 'review' column of the DataFrame
* splits the dataset into the training and testing sets
* trains the model and evaluates its performance
* asks the user to input a new movie review and predicts its sentiment using the trained model

## Movie Reviews 2
This Python code implements a simple Naive Bayes classifications for sentiment analysis.

### Imported Libraries
* pandas
* string
* Counter from collections

### Load Data
Reads a CSV file containing movie reviews and their sentiments (positive or negative) into a Pandas DataDrame

### Preprocess Text
* defines a function to remove punctuation and tokenize the text
* applies this function to the 'review' column of the DataFrame, storing the result as a list of words

### Split dataset
Splits the dataset into training and testing sets, using 80% for training and 20% for testing

### Naive Bayes Classification
* defines a simple Naive Bayes classifier class
* the 'train' method calculates class probabilities and word probabilites based on the training data
* the 'predict' method predicts the sentiment of a given review based on the trained probabilities

### Train the Classifier
* creates an instance of the Naive Bayes classifier class
* trains the classifier using the training data

### Make Predictions
uses the trained classifier to make predictions on the test data

### Evaluate the Model
* compares the predicted sentiments with the actual sentiments in the test data to calculate accuracy
* prints the accuracy of the model on the test data

### Predict New Review
* take a new movie review as input from the user
* preprocesses the input review
* uses the trained classifier to predict the sentiment of the new review
* prints the predicted sentiment

## Movie Reviews 3
This Python code performs sentiment analysis on movie reviews using a deep learning model

### Imported Libraries
* pandas
* sklearn
* tensorflow
* string

### Load Data
Reads a CSV file containing movie reviews and their sentiments (positive or negative) into a Pandas DataFrame

### Preprocess Text
* defines a function to remove punctuation and convert text to lowercase
* applies this function to the 'review' column of the DataFrame, preprocessing the text data

### Split Dataset
splits the dataset into training and testing sets, using 80% for training and 20% for testing

### Initialize Tokenizer and Label Encoder
* initializes a tokenizer to convert text data to sequences of integers
* initializes a label encoder to encode sentiment labels into numerical format

### Train Deep Learning Model
* defines a function that takes training data and labels and returns a trained deep learning model, tokenizer, and label encoder
* tokenizes and pads the input reviews
* builds a sequential model with an embedding layer, a flatten layer, and a dense layer
* compiles and trains the model using binary cross-entropy loss and the Adam optimizer

### Evaluate Model
defines a function that takes a trained model, label encoder, test data, and labels, and returns the accuracy of the model on the test set

### Main Method
* loads the data, preprocesses the text, and splits it into training and testing sets
* calls the 'train_model" function to train the deep learning model
* calls the 'evaluate_model' function to evaluate the model on the testing set
* takes a new movie review as input, preprocesses it, tokenizes and pads it, and uses the trained model to predict the sentiment
* prints the accuracy on the test set and the predicted sentiment for the new review
