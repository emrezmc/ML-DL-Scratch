import numpy as np


def train_test_split(X, y, test_size=None, random_state=1234):
    # Creating indices for random train/test split
    np.random.seed(random_state)
    indices = list(range(X.shape[0]))
    num_train = int((1 - test_size) * X.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # Splitting data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


# Simple evaluate accuracy function
def accuracy(y_true, y_pred):
    acc = np.sum((y_pred == y_true) / len(y_true))
    return acc


# Calculation of euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
