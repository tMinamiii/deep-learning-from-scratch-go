import numpy as np

from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # self.W = np.random.rand(2, 3)
        self.W = np.array([
            [0.47355232, 0.9977393, 0.84668094],
            [0.85557411, 0.03563661, 0.69422093]
        ])

    def predict(self, x):
        # print(x)
        # print(self.W)
        # print(np.dot(x, self.W))
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)
t = np.array([0, 0, 1])
loss = net.loss(x, t)


def f(W):
    # print(x, t)
    # print(net.W)
    # print(net.loss(x, t))
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
