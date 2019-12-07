import numpy as np
from methods import gradient


class AdamOpt():

    def __init__(self, n, alpha=0.001, gamma_v=0.9, gamma_s=0.999, eps=1e-8):
        self._alpha = alpha
        self._gamma_v = gamma_v
        self._gamma_s = gamma_s
        self._eps = eps
        self._n = n
        self._k = 0
        self._s = np.zeros(self._n)
        self._v = np.zeros(self._n)

    def init(self):
        self._k = 0
        self._s = np.zeros(self._n)
        self._v = np.zeros(self._n)

    def step(self, f, x):
        g = gradient(f, x)
        self._v = self._gamma_v * self._v + (1 - self._gamma_v) * g
        self._s = self._gamma_s * self._s + (1 - self._gamma_s) * (g ** 2)
        self._k += 1
        v_hat = self._v / (1 - self._gamma_v ** self._k)
        s_hat = self._s / (1 - self._gamma_s ** self._k)
        return x - self._alpha * v_hat / (np.sqrt(s_hat) + self._eps)

    def optimize(self, f, x, tol=1e-10):
        diff = np.Inf

        while diff > tol:
            nx = self.step(f, x)
            diff = np.abs(f(nx) - f(x))
            x = nx

        return x


class MomemtunOpt():

    def __init__(self, n, alpha=1e-4, beta=1e-4):
        self._alpha = alpha
        self._beta = beta
        self._n = n
        self._test = 0
        self._v = np.zeros(n)

    def init(self):
        self._v = np.zeros(self._n)

    def step(self, f, x):
        g = gradient(f, x)
        self._v = self._beta * self._v - self._alpha * g
        return x + self._v

    def optimize(self, f, x, tol=1e-10):
        diff = np.Inf

        while diff > tol:
            nx = self.step(f, x)
            diff = np.abs(f(nx) - f(x))
            x = nx

        return x

class gradient_descent_methods():
    
    def __init__(self, alpha = 0.001):
        self._alpha = alpha
    
    def step(self, f, x):
        x = x - self._alpha * gradient(f, x)
        return x
    
    def optimize(self, f, x, tol=1e-10):
        diff = np.inf
        
        while diff > tol:
            nx = self.step(f, x)
            diff = np.abs(f(nx) - f(x))
            x = nx
        
        return x
