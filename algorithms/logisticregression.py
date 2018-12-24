import numpy as np


class LogisticRegression:
    def __init__(self,
                 learning_rate=0.01,
                 num_iter=100000,
                 fit_intercept=True,
                 verbose=True):

        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.theta = None
        self.verbose = verbose

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return ((-y * np.log(h)) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = X.dot(self.theta)
            h = self.sigmoid(z)
            grad = np.dot(X.T, (h - y)) / len(y)
            self.theta = self.theta - self.learning_rate * grad

            if self.verbose:
                if i % 5000 == 0:
                    print('loss: {}'.format(self.loss(h, y)))

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn import linear_model
    # from algorithms.logisticregression import LogisticRegression
    iris = load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    model = LogisticRegression(learning_rate=0.1, num_iter=300000)
    model.fit(X, y)
    preds = model.predict(X)
    print('UDF performance: \n')
    print('Accuracy: ', (preds == y).mean())
    print('Theta: ', model.theta)

    print('SKLEARN performance: \n')
    model2 = linear_model.LogisticRegression(C=1e20)
    model2.fit(X, y)
    preds2 = model2.predict(X)
    print('Accuracy: ', (preds2 == y).mean())
    print('Theta: ', model2.intercept_, model2.coef_)
