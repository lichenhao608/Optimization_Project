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

    def __init__(self, alpha=-10):
        self.alpha = alpha

    def step(self, f, x):
        g = gradient(f, x)
        alpha = self.backtracking_line_search(f, x, g, self.alpha)
        return x + alpha * g

    def backtracking_line_search(self, f, x, d, alpha, p=0.5, beta=1e-4):
        y, g = f(x), gradient(f, x)
        while f(x + alpha * d) > y + beta * alpha * np.matmul(g, d):
            alpha *= p
        return alpha

    def optimize(self, f, x, tol=1e-10):
        diff = np.inf

        while diff > tol:
            nx = self.step(f, x)
            diff = np.abs(f(nx) - f(x))
            x = nx

        return x


class conjugate_gradient_methods():

    def __init__(self, f, x,  x_tol=0.0005, f_tol=0.01):
        self._x_tol = x_tol
        self._f_tol = f_tol
        self._g = gradient(f, x)
        self._d = -1 * self._g

    def step(self, f, x):
        gprime = gradient(f, x)
        beta = max(0, np.dot(gprime, gprime - self._g) /
                   (np.dot(self._g, self._g)))
        dprime = -1 * gprime + beta * self._d
        x = self.backtracking_line_search(f, x, dprime)
        self._g = gprime
        self._d = dprime
        return x

    def backtracking_line_search(self, f, x, d, alpha=0.001, p=0.5, beta=1e-4):
        y = f(x)
        g = gradient(f, x)

        while f(x + alpha * d) > y + beta * alpha * (np.dot(g, d)):
            alpha = alpha * 0.5

        return x + alpha * d


class ArgumentedLagrange():

    def __init__(self, h, X, rho=1, gamma=2):
        '''
        Args:
            h (callable): contraint functions, should return an array with shape
                (X,)
            X (int): number of constraint functions
        '''
        self.h = h
        self.rho = rho
        self.gamma = gamma
        self.lam = np.zeros(X)

    def step(self, f, x):
        def p(a):
            return f(a) + self.rho / 2 * np.sum(self.h(a)**2) - np.matmul(self.lam, self.h(a))
        x = self.minimize(lambda a: f(a) + p(a), x)
        r = self.gamma * self.rho
        self.lam -= r * self.h(x)
        return x

    def minimize(self, f, x):
        m = gradient_descent_methods()
        x = m.optimize(f, x)
        return x
