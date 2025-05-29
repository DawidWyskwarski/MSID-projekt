import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder


class SoftmaxRegression:

    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32, epoches: int = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoches = epoches
        self.labels = None
        ### Why does it suddenly work when I added the '_' after the name
        ### like what the hell ?!?!?
        self.weights_ = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def encode_target(self, y):
        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(y)

        self.labels = label_encoder.classes_

        return y_int

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        X_biased = np.hstack( ( X, np.ones( (n_samples, 1) ) ) )

        y_one_hot = np.eye(n_classes)[ self.encode_target(y) ]

        self.weights_ = np.zeros((n_features + 1, n_classes))

        for _ in range(self.epoches):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_biased[i:i+self.batch_size]
                y_batch = y_one_hot[i:i+self.batch_size]

                logits = X_batch @ self.weights_
                probabilities = self.softmax(logits)

                gradient = X_batch.T @ (probabilities - y_batch) / X_batch.shape[0]

                self.weights_ -= self.learning_rate * gradient

        return self

    def predict(self, X):
        if self.weights_ is None:
            raise NotFittedError

        X_biased = np.hstack((X, np.ones((X.shape[0], 1))))
        probabilities = self.softmax(X_biased @ self.weights_)

        pred = np.argmax(probabilities, axis=1)

        return [self.labels[x] for x in pred]