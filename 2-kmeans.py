import numpy as np
from utils import euclidean_distance


class KMeans:

    def __init__(self, K=2, max_iterations=500, random_seed=1234):
        self.K = K
        self.max_iterations = max_iterations
        self.random_seed = random_seed

    def initialize_random_centroids(self, X):
        # Initialize centroids as k random samples
        np.random.seed(self.random_seed)
        n_samples, n_features = X.shape
        centroids = np.zeros((self.K, n_features))

        for i in range(self.K):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self, sample, centroids):
        # Return the index of the closest centroid to sample
        closest_index = 0
        closest_dist = float('inf')

        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_index = i
                closest_dist = distance
        return closest_index

    def create_clusters(self, centroids, X):
        # Assign samples to closest centroids to create clusters.
        clusters = [[] for _ in range(self.K)]
        for sample_index, sample in enumerate(X):
            centroid_index = self.closest_centroid(sample, centroids)
            clusters[centroid_index].append(sample_index)
        return clusters

    def calculate_centroids(self, clusters, X):
        # Calculate new centroids as the mean of sample in each cluster
        n_features = X.shape[1]
        centroids = np.zeros((self.K, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_label(self, clusters, X):
        # Classify sample as index of their index
        y_prediction = np.zeros(X.shape[0])
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                y_prediction[sample_index] = cluster_index
        return y_prediction

    def predict(self, X):
        # Apply K-Means clustering and return clustered indices
        # Initialize centroids randomly

        centroids = self.initialize_random_centroids(X)

        # Optimization until converged for max iterations
        for _ in range(self.max_iterations):
            # Create clusters and assign samples to closest cendroids
            clusters = self.create_clusters(centroids, X)

            # Save centroids for convergence check
            prev_centroids = centroids

            # Calculate new centroids from clusters
            centroids = self.calculate_centroids(clusters, X)

            # Check if it converged
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self.get_cluster_label(clusters, X).reshape(-1, 1)