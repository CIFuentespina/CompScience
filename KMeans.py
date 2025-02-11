import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)
            
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

    def predict(self, X):
        return self._assign_clusters(X)

def generate_data(n_samples=300, centers=3):
    np.random.seed(42)
    X = np.vstack([np.random.randn(n_samples // centers, 2) + np.random.rand(2) * 5 for _ in range(centers)])
    return X

def plot_clusters(X, labels, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", edgecolors="k", alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="yellow", s=200, marker="x", label="Centroids")
    plt.legend()
    plt.title("K-Means Clustering")
    plt.show()

X = generate_data()
kmeans = KMeans(k=3)
kmeans.fit(X)

plot_clusters(X, kmeans.labels_, kmeans.centroids)
