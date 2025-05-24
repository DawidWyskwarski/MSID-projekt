import numpy as np
from sklearn.exceptions import NotFittedError

class MyLinearRegression:

    def __init__(self):
        self.theta_ = None

    def fit(self, X, y):
        ### Adding the ones for bias
        X = np.c_[np.ones((X.shape[0], 1)), X]
        pars = np.linalg.pinv(X.T @ X) @ X.T @ y

        self.theta_ = pars

    def predict(self, X):
        if self.theta_ is None:
            raise NotFittedError("Model is not fitted yet.")

        return np.c_[np.ones((X.shape[0], 1)), X] @ self.theta_