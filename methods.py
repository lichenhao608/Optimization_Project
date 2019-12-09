# ------------------------------------------------------------- #
# This file includes some useful methods for this project
# ------------------------------------------------------------- #
import numpy as np


def gradient(f, x, h=1e-10):
    '''
    Calculate the gradient of f at x by using forward difference

    Args:
        f: function that want to find the gradient
        x: the position where want to find the gradient

    Returns:
        grad: an array of gradients for each dimension
    '''
    temp = []
    N = len(x)

    for i in range(N):
        xx = np.copy(x)
        xx[i] += h
        temp.append((f(xx) - f(x)) / h)

    return np.array(temp)
