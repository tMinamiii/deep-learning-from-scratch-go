import numpy as np

from common.functions import cross_entropy_error, sigmoid, softmax

x = np.array([[3, 1, 0], [1, 4, 0]])
y = np.array([[1, 0], [0, 1]])
sm = softmax(x)
cee = cross_entropy_error(y, y)
print(cee)
