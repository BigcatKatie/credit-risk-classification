# credit-risk-classification project
Objective: Use various techniques to train and evaluate a model based on loan risk. Use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Part 1: Split the Data into Training and Testing Sets

# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

Step 1: Read the lending_data.csv data from the Creit_Risk folder into a Pandas DataFrame.
# Read the CSV file from the Resources folder into a Pandas DataFrame
# Review the DataFrame

Step 2: Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
# Separate the data into labels and features
# Separate the y variable, the labels
# Separate the X variable, the features
# Review the y variable Series
# Review the X variable DataFrame

Step 3: Split the data into training and testing datasets by using train_test_split
# Import the train_test_learn module
# Split the data using train_test_split
# Assign a random_state of 1 to the function
# Review the shape of the resulting datasets

Part 2: Create a Logistic Regression Model with the Original Data
Step 1: Fit a logistic regression model by using the training data (X_train and y_train).
# Import the LogisticRegression module from SKLearn
# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
# Fit the model using training data

Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
# Make a prediction using the testing data
# Review the predictions

Step 3: Evaluate the model’s performance by doing the following:
# Generate a confusion matrix for the model
# Print the classification report for the model

Conclusion: The model is very accurate for predicting healthy loans (label 0), with near perfect precision, recall, and F1-score. The model performas well for high-risk loans (label 1), with relatively high precision and recall, there is room for improvement, particularly in reducing the number of false positives. The overall accuracy of 0.99 and strong macro average scores indicate that the model is highly effective across both categories, but slightly better at predicting healthy loans.
