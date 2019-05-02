import time

import numpy as np

from common.functions import cross_entropy_error, sigmoid, softmax
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        start = time.time()
        y = self.predict(x)
        cee = cross_entropy_error(y, t)
        end = time.time()
        print((end-start) * 1000)
        return cee

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=2)

        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {}
        print("calc W1")
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        print("calc b1")
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        print("calc W2")
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        print("calc b2")
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


def runTLN():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.numerical_gradient(x_batch, t_batch)

        for key in ['W1', 'b1', 'W2', 'b2']:
            net.params[key] -= learning_rate * grad[key]

        loss = net.loss(x_batch, t_batch)
        print(loss)
        train_loss_list.append(loss)


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


def run_simple_net():
    net = simpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    np.argmax(p)
    t = np.array([0, 0, 1])
    net.loss(x, t)

    def f(W):
        # print(x, t)
        # print(net.W)
        # print(net.loss(x, t))
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)


def main():
    runTLN()


if __name__ == '__main__':
    main()
