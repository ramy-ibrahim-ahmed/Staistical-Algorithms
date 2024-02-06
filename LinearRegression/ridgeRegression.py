import numpy as np
class RidgeRegression:
    def __init__(self,alpha=0.01,lmbda=0.1,num_iterations=50,print_loss=False):
        self.alpha = alpha
        self.lmbda=0.1
        self.num_iterations=num_iterations
        self.print_loss = print_loss
    def __initialize_weights(self,X):
        self.m = X.shape[0]
        n = X.shape[1]
        self.W = np.random.randn(1,n)*0.01
        self.b = 0
    def __hypothesis(self,X):
        yhat = np.matmul(self.W,X.T)+self.b
        return yhat
    def __mean_squared_error(self,y,yhat):
        loss = (1/(2*self.m)) * np.sum((yhat-y)**2)
        regularization = (self.lmbda/(2*self.m)) * np.sum(self.W**2)
        return loss + regularization

    def gradient_decent(self,X, y,):
        for i in range(self.num_iterations):
            yhat = self.__hypothesis(X)

            dW  = (1/self.m) * np.matmul(yhat-y,X)
            db = (1/self.m) * np.sum(yhat-y)
            W_regularization = (self.lmbda/self.m) * self.W
            self.W = self.W - (self.alpha * dW + self.alpha * W_regularization)
            self.b = self.b - self.alpha * db

            if self.print_loss:
                loss = self.__mean_squared_error(yhat,y)
                print(loss)

    def fit(self,X,y):
        self.__initialize_weights(X)
        self.gradient_decent(X,y)
    def predict(self,X):
        if isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise ValueError("Input must be a NumPy array or a list of numbers")
        return np.sum(np.matmul(self.W,X.T))+self.b

