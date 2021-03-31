from kmeans import KMeans
import pandas as pd
from utils import accuracy

X = pd.read_csv("breast_data.csv", header=None).to_numpy()
y = pd.read_csv("breast_truth.csv", header=None).to_numpy()

kmeans = KMeans(K=2, max_iterations=500)
y_pred = kmeans.predict(X)

accuracy = accuracy(y, y_pred)
print("Accuracy: %.2f" % accuracy)