import numpy as np


def gradient(f, x, c=0, rou=0, lam=0, h=1e-10):
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
        if (rou==0 and lam==0):
            temp.append((f(xx) - f(x)) / h)
        else:
            pass
    return np.array(temp)

def test_function(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + (x[6]+1) ** 2 + (x[7]-5) ** 2 + (x[8]-5) ** 2 + (x[9]+6) ** 2


def Rosenbrock(x, b=5, a=1):
    return (a-x[0]) ** 2 + b* (x[1]-x[0]**2) ** 2 + (a-x[1]) ** 2 + b * (x[2]-x[1]**2) ** 2 + (a-x[2]) ** 2 + b * (x[3]-x[2]**2) **2 + (a-x[3]) ** 2 + b * (x[4]-x[3]**2) ** 2 + (a-x[4]) ** 2 + b * (x[5]-x[4]**2) **2 + (a-x[5]) ** 2 + b * (x[6]-x[5]**2) **2 + (a-x[6]) ** 2 + b * (x[7]-x[6]**2)**2 + (a-x[7]) ** 2 + b * (x[8]-x[7]**2) ** 2 + (a-x[8]) ** 2 + b * (x[9]-x[8]**2) ** 2

def penalty_function(f, h, x, rou):
    return f(x) + rou/2 * sum((h(x))**2) - (lam * h(x))

def constraint_function(x):
    return 3*x[0]**3 + x[1]**2 - x[2]**2 + x[3] ** 2 + 4 * x[4] + 5 * x[5] + 7 * x[6] + x[7] ** 2 - x[8]  + x[9] - 2


x = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 3.0, 2.0, 1.0]


