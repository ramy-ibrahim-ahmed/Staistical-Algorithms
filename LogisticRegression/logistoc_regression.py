import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, alpha=0.01, num_iters=1000, lambda_=0.1, verbose=False):
        """
        regularization hyperparameter default = 0.1
        """
        m = X.shape[0]
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.verbose = verbose

        for _ in range(num_iters):
            z = X @ self.w + self.b
            g = self.sigmoid(z)
            error = g - y
            self.w -= alpha * ((X.T @ error) + lambda_ * self.w) / m
            self.b -= alpha * np.sum(error) / m

            cost_iter = self.cost(X, y)
            if self.verbose and _ % (num_iters / 10) == 0:
                print(f"Iteration {_}: Cost = {cost_iter}")

    def probability(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.probability(X)
        return (proba >= threshold).astype(int)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y, lambda_=0.1):
        m = X.shape[0]
        z = X @ self.w + self.b
        g = self.sigmoid(z)
        loss = np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
        reg_term = (lambda_ / (2 * m)) * np.sum(self.w**2)
        return -loss / m + reg_term