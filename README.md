# IMDb Movie Review Sentiment Analysis
This project aims to classify the sentiment (positive or negative) of movie reviews using the IMDb dataset. The model is trained using a Naive Bayes classifier, with text data preprocessed through tokenization and vectorization. The dataset used in this project is sourced from Stanford AI Lab's IMDb dataset.

## Dataset
The dataset contains 50,000 movie reviews labeled as either positive or negative. The reviews have been preprocessed by tokenizing each review into sequences of integers representing words, where each integer corresponds to a word index. For more details, visit the official IMDb dataset page.

<br/>

## movie_reviews1.py
### Features
* Text Preprocessing:
    * Tokenization of reviews into sequences of word indices.
    * Punctuation removal and conversion to lowercase.
* Model: Naive Bayes classifier (MultinomialNB) using term frequency counts.
* Evaluation: Classification report and accuracy score on the test dataset.

### Installation
To run the project, make sure to have the following dependencies installed: pip install numpy pandas scikit-learn tensorflow

### Usage
1. Load the IMDb dataset: The dataset is loaded from tf.keras.datasets.imdb.load_data().
2. Preprocess the text: The text data is preprocessed by removing punctuation and converting the reviews to lowercase.
3. Train the model: A Naive Bayes classifier is trained on the tokenized reviews.
4. Evaluate the model: The model's accuracy and classification report are generated to assess performance on the test data.
5. Sentiment Prediction: The script also allows the user to input a new movie review and predicts whether the review is positive or negative.

### Example
```bash
$ python movie_reviews1.py
Accuracy: 85.23%
Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.86      0.85     12500
           1       0.85      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000
Enter movie review: "This movie was fantastic! I loved it."
Predicted sentiment for the new review: positive
```

<br/>

## movie_reviews2.py
### Features
* Text Preprocessing:
    * Decoding reviews into readable text.
    * Punctuation removal and word tokenization.
* Model: Custom-built Naive Bayes classifier using word probabilities.
* Evaluation: Accuracy score on the test dataset and sentiment prediction for new reviews.

### Installation
To run the project, make sure to have the following dependencies installed:
pip install numpy pandas tensorflow

### Usage
1. Load the IMDb dataset: The dataset is loaded from tf.keras.datasets.imdb.load_data().
2. Preprocess the text: The text data is decoded from integer sequences to readable reviews, followed by punctuation removal and word tokenization.
3. Train the model: A Naive Bayes classifier is custom-built to compute word probabilities and is trained on the preprocessed reviews.
4. Evaluate the model: The model's accuracy is calculated based on the test data.
5. Sentiment Prediction: The script also allows the user to input a new movie review and predicts whether the review is positive or negative.

### Example
```bash
$ python movie_reviews2.py
Accuracy: 0.8425

Enter movie review: "This movie was fantastic! I loved it."
Predicted sentiment for the new review: Positive
```

<br/>

## movie_reviews3.py
### Features
* Text Preprocessing:
    * Padding of sequences to ensure all reviews have the same length.
    * Tokenization to map words to integers.
* Model: Deep learning model with an embedding layer, followed by a flattening layer and a dense output layer using a sigmoid activation function for binary classification.
* Evaluation: Model accuracy on the test dataset.
* Sentiment Prediction: Allows user input for new movie reviews and predicts whether the sentiment is positive or negative.

### Installation
To run the project, make sure to have the following dependencies installed:
pip install numpy pandas scikit-learn tensorflow matplotlib

### Usage
1. Load the IMDb dataset: The dataset is loaded from tf.keras.datasets.imdb.load_data() and padded to ensure all reviews are the same length.
2. Build the model: A deep learning model is created using an embedding layer for text representation, followed by a flattening layer and a dense output layer for binary classification.
3. Train the model: The model is trained on the padded sequences with a validation split of 20%.
4. Evaluate the model: The model's accuracy is evaluated using the test dataset.
5. Sentiment Prediction: The script allows the user to input a new movie review and predicts whether the review is positive or negative.

### Example
```bash
$ python movie_reviews3.py
Epoch 1/5
625/625 [==============================] - 7s 10ms/step - loss: 0.4968 - accuracy: 0.7444 - val_loss: 0.3297 - val_accuracy: 0.8604
Epoch 2/5
625/625 [==============================] - 6s 10ms/step - loss: 0.2414 - accuracy: 0.9089 - val_loss: 0.3196 - val_accuracy: 0.8676
Epoch 3/5
625/625 [==============================] - 6s 10ms/step - loss: 0.1351 - accuracy: 0.9613 - val_loss: 0.3530 - val_accuracy: 0.8640
Epoch 4/5
625/625 [==============================] - 6s 10ms/step - loss: 0.0626 - accuracy: 0.9888 - val_loss: 0.3835 - val_accuracy: 0.8600
Epoch 5/5
625/625 [==============================] - 6s 10ms/step - loss: 0.0285 - accuracy: 0.9971 - val_loss: 0.4209 - val_accuracy: 0.8582
782/782 [==============================] - 3s 4ms/step - loss: 0.4375 - accuracy: 0.8492
Accuracy: 0.8492000102996826

Enter movie review: "This movie was terrible. The plot made no sense, and the characters were awful."
Predicted sentiment for the new review: negative
```

<br/>

# Conclusion
This project demonstrates the application of various machine learning techniques to classify the sentiment of movie reviews using the IMDb dataset. By exploring different models, including Naive Bayes classifiers and deep learning architectures, we were able to achieve a satisfactory level of accuracy in sentiment prediction.
* Model Variations: The transition from a traditional Naive Bayes approach to a deep learning model highlights the effectiveness of leveraging neural networks for natural language processing tasks. The deep learning model, in particular, benefits from its ability to learn complex representations of text through embeddings.
* Performance Evaluation: The evaluation metrics, including accuracy and classification reports, provide a comprehensive understanding of each model's performance, allowing for informed comparisons between them.
* Future Work: There are several avenues for improvement and exploration, such as:
    * Hyperparameter Tuning: Fine-tuning model parameters to optimize performance.
    * Model Ensemble: Combining predictions from multiple models to enhance accuracy.
    * Data Augmentation: Expanding the dataset through techniques like synonym replacement to increase robustness.

This project serves as a foundational exploration into sentiment analysis, showcasing the potential of machine learning in understanding and processing human language. Further enhancements could lead to even more accurate and insightful sentiment predictions.