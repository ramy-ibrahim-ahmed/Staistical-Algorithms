import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, alpha=0.01, num_iters=1000):
        m = X.shape[0]
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for _ in range(num_iters):
            z = X @ self.w + self.b
            g = self.sigmoid(z)
            error = g - y
            self.w -= alpha * (X.T @ error) / m
            self.b -= alpha * np.sum(error) / m

    def probability(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.probability(X)
        return (proba >= threshold).astype(int)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y):
        m = X.shape[0]
        z = X @ self.w + self.b
        g = self.sigmoid(z)
        loss = np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
        return -loss / m