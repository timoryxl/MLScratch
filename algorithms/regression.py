import numpy as np


class L1Regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2Regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return 0.5 * self.alpha * w.T.dot(w)

    def grad(self, w):
        return self.alpha * w


class L1L2Regularization:
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1 = self.l1_ratio * np.linalg.norm(w)
        l2 = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1 + l2)

    def grad(self, w):
        l1_grad = self.l1_ratio * np.sign(w)
        l2_grad = (1 - self.l1_ratio) * w
        return self.alpha * (l1_grad + l2_grad)


class Regression:
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.w = None
        self.training_errors = []

    def initialize_weights(self, n_features):
        # basic: initiate with zeros
        limit = 1/np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])

        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5*(y_pred-y)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X)/X.shape[0] + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)

