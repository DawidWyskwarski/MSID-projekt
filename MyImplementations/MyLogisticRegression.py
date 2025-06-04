import numpy as np

class MyLogisticRegression:

    def __init__(self, learning_rate = 0.01, batch_size = 32, epoches = 1000, threshold = 0.5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoches = epoches
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def __loss(self, y_true, y_pred):
        ### We add this to avoid log(0) which doesn't exist
        epsilon = 1e-6

        return -np.mean(y_true*np.log(y_pred + epsilon) + (1-y_true)*np.log(1-y_pred))

    def fit(self, X, y):
        n_samples = X.shape[0]

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epoches):

            for i in range(0,n_samples, self.batch_size):
                X_sub = X[i:i+self.batch_size]
                y_sub = y[i:i+self.batch_size]

                z = np.dot(X_sub,self.weights) + self.bias
                A = self.__sigmoid(z)

                dw = (1 / n_samples) * np.dot( X_sub.T,(A - y_sub) )
                db = (1 / n_samples) * np.sum(A - y_sub)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        ys = np.dot(X, self.weights) + self.bias
        y_pred = self.__sigmoid(ys)

        y_pred_cls = [1 if i >= self.threshold else 0 for i in y_pred]

        return np.array(y_pred_cls)