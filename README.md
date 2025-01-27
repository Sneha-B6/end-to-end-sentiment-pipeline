# end-to-end-sentiment-pipeline

This project provides a sentiment analysis API using the IMDB dataset for classifying movie reviews as positive or negative.

>Setup
 Clone the repository and navigate to the project folder.
 Install required dependencies from requirements.txt.
 Set up the SQLite database with movie reviews using data_setup.py.

>Training
 Train the sentiment analysis model using train_model.py.

>API Usage
 Start the Flask API by running app.py.
 Predict sentiment by sending a GET request to /predict with a review_text parameter.
 The model returns whether the review sentiment is positive or negative.

>Model Information
 Model used: Logistic Regression with TF-IDF
 Performance: Achieved an accuracy of X% on the test set.

>Future Improvements
 Experiment with transformer-based models like BERT.
 Deploy the API to a cloud platform like Heroku.
