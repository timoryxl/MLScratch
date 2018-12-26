import numpy as np


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = dict()
        self.classifications = dict()

    def fit(self, data):

        # Initialization
        for i in range(self.k):
            self.centroids[i] = data[i]

        for iter_ in range(self.max_iter):

            for i in range(self.k):
                self.classifications[i] = []

            for point in data:
                distances = [np.sqrt(np.sum((point - self.centroids[centroid]) ** 2)) for centroid in self.centroids]
                label = distances.index(min(distances))
                self.classifications[label].append(point)

            prev_centroid = dict(self.centroids)
            for label in self.classifications:
                self.centroids[label] = np.mean(self.classifications[label], axis=0)

            optimized = True
            for c in self.centroids:
                error = np.sum((self.centroids[c]-prev_centroid[c])/prev_centroid[c])
                if error > self.tol:
                    optimized = False
                    break
            if optimized:
                break

    def predict(self, data):
        distances = [np.sqrt(np.sum((data - self.centroids[centroid]) ** 2)) for centroid in self.centroids]
        return distances.index(min(distances))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])

    colors = 10 * ["g", "r", "c", "b", "k"]
    clf = KMeans()
    clf.fit(X)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    plt.show()
