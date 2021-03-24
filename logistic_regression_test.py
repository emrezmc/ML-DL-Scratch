# Importing libraries and modules
import pandas as pd
from utils import *
from logistic_regression import LogisticRegressor

# Reading data into variables.
X = pd.read_csv("breast_data.csv", header=None).to_numpy()
y = pd.read_csv("breast_truth.csv", header=None).to_numpy()

# Splitting data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Creating model
model = LogisticRegressor(learning_rate=0.0001, n_iterations=100)
model.fit(X_train, y_train)

# Predict test data for evaluate model
predict_test = model.predict(X_test)

print("Accuracy of model on test data: %2.2f" % accuracy(y_test, predict_test))
