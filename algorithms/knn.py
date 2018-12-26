import numpy as np


class KnnBase:
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights
        self.train_features = None
        self.train_labels = None

    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def get_neighbours(self, test_point, k):
        if k:
            self.k = k
        distance = np.sqrt(np.sum((self.train_features - test_point)**2, axis=1))
        return np.argsort(distance)[0:self.k]

    def _get_key_max_value(self, d):
        new_dict = {k: v for v, k in d.items()}
        return new_dict[max(new_dict)]


class KnnClassifier(KnnBase):
    def predict(self, test_point):
        nearest_indices = self.get_neighbours(test_point, self.k)
        voter = dict()
        for index in nearest_indices:
            label = self.train_labels[index]
            voter[label] = voter.get(label, 0) + 1
        return self._get_key_max_value(voter)


class KnnRegressor(KnnBase):
    def predict(self, test_point):
        nearest_indices = self.get_neighbours(test_point, self.k)
        return np.mean(self.train_labels[nearest_indices])


def get_accuracy(y, y_pred):
    cnt = (y == y_pred).sum()
    return round(cnt/len(y), 2)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.preprocessing import MinMaxScaler

    # load the iris data set
    iris = datasets.load_iris()
    knn_iris_acc = []
    X = iris.data
    y = iris.target

    scale = MinMaxScaler()
    X = scale.fit_transform(X)
    for k in range(2, len(iris.data)):
        clf = KnnClassifier(k)
        clf.fit(X, y)
        iris_pred = []
        for x in X:
            pred = clf.predict(x)
            iris_pred.append(pred)
        iris_target_pred = np.array(iris_pred)
        knn_iris_acc.append(get_accuracy(iris_target_pred, iris.target))
    print(knn_iris_acc)
