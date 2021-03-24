import numpy as np


class LogisticRegressor:

    def __init__(self, learning_rate=0.001, n_iterations=500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize parameters
        num_sample, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.weights = self.weights.reshape(-1,1)
        # Gradient Descent
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predicted = self.sigmoidfunc(model)

            dw = (1 / num_sample) * np.dot(X.T, (predicted - y))
            db = (1 / num_sample) * np.sum(predicted - y)
            self.weights = self.weights - self.learning_rate * dw

            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predicted = self.sigmoidfunc(model)
        predicted_classes = [1 if i > 0.5 else 0 for i in predicted]
        return predicted_classes

    def sigmoidfunc(self, x):
        return 1 / (1 + np.exp(-x))
