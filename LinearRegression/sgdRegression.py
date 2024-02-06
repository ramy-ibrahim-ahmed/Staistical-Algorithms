import numpy as np


class SGDRegression:
    def __init__(self, alpha=0.01,annealing_rate=0.9, num_iterations=50, print_loss=False):
        self.alpha = alpha
        self.annealing_rate = annealing_rate
        self.num_iterations = num_iterations
        self.print_loss = print_loss

    def __initialize_weights(self, X):
        n = X.shape[1]
        self.m = X.shape[0]
        self.W = np.random.randn(1, n) * 0.01
        self.b = 0

    def __hypothesis(self, X):
        yhat = np.matmul(self.W, X.T) + self.b
        return yhat

    def __mean_squared_error(self, y, yhat):
        loss = (1 / (2 * self.m)) * np.sum((yhat - y)**2)
        return loss

    def gradient_descent(self, X, y):
        t0, t1 = 5, 50
        learning_rate = self.alpha  # Initialize learning rate

        def learning_schedule(t):
            return t0 / (t + t1)

        for i in range(self.num_iterations):
            for j in range(len(y)):
                random_index = np.random.randint(len(y))
                xi = X[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                yhat = self.__hypothesis(xi)

                dW = np.matmul((yhat - yi).T, xi)
                db = np.sum(yhat - yi)

                self.W = self.W - learning_rate * dW
                self.b = self.b - learning_rate * db

                eta = learning_schedule(i * len(y) + j)
                learning_rate = max(eta, 1e-10)  # Ensure the learning rate is always positive

            # Anneal the learning rate after each epoch
            learning_rate *= self.annealing_rate

            if self.print_loss:
               loss = self.__mean_squared_error(yhat, y)
               print(loss)
    def fit(self, X, y):
        self.__initialize_weights(X)
        self.gradient_descent(X, y)

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise ValueError("Input must be a NumPy array or a list of numbers")

        return np.sum(np.matmul(self.W, X.T)) + self.b



