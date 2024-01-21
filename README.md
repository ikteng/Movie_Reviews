# Movie_Reviews

Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

This code uses pandas, sklearn, tensorflow.keras and string.
This code performs a sentiment analysis on movie reviews using a simple neural network and evaluates its accuracy on a testing set. The trained model is then used to predict sentiment for a new user-input review.

## Load and Preprocess Data
The data is loaded as a Dataframe from the dataset which is a csv file.
The reviews are preprocessed by removing punctuation and converted to lowercase.

## Split Dataset
The dataset is split intro training and testing sets using 'train_test_split'.
X is reviews and y is sentiment (positive/negative).

## Train Model
The 'train_model' function does:
  - Updates the internal vocabulary using the training data.
  - Converts reviews to sequences of integers.
  - Pads/truncates sequences to a fixed length.
  - Transforms categorical sentiment labels into numerical format.
  - Builds and trains a simple neural network using TensorFlow's Sequential API.

## Evalute Model
It computes the accuracy of the model using the testing set

## Predict Sentiment for new review
User can input a new movie review which is preprocessed, tokenized and padded/truncated to match the input requirement of the model. The model will predict the sentiment probability. If the sentiment probability is more than 0.5, the review is considered as positive. Else, it will be considered as negative.
