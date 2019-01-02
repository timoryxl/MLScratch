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

