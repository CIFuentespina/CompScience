import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, regression=False):
        return np.array([self._predict(x, regression) for x in X])

    def _predict(self, x, regression):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        if regression:
            return np.mean(k_nearest_labels)
        return np.bincount(k_nearest_labels).argmax()

    def _compute_distances(self, x):
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == "minkowski":
            p = 3  # You can change the Minkowski power
            return np.sum(np.abs(self.X_train - x) ** p, axis=1) ** (1 / p)
        else:
            raise ValueError("Unsupported distance metric")

def generate_data(n_samples=100, regression=False):
    np.random.seed(42)
    X = np.random.rand(n_samples, 2)
    if regression:
        y = np.sin(2 * np.pi * X[:, 0]) + np.random.randn(n_samples) * 0.1
    else:
        y = np.random.randint(0, 2, n_samples)
    return X, y

def plot_decision_boundary(knn, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(f"KNN Decision Boundary (k={knn.k}, {knn.distance_metric})")
    plt.show()

X_train, y_train = generate_data(100)
X_test, y_test = generate_data(10)

knn = KNN(k=5, distance_metric="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Predicted labels:", y_pred)

plot_decision_boundary(knn, X_train, y_train)
