import numpy as np
from methods import gradient


class LookAheadOpt():

    def __init__(self, opt, k=5, alpha=0.5):
        self._opt = opt
        self._k = k
        self._alpha = alpha

    def step(self, f, x):
        nx = np.copy(x)
        # TODO: should the la opt each step using the initial state of the called
        # opt or continuing update the called opt
        self._opt.init()

        for _ in range(self._k):
            nx = self._opt.step(f, nx)

        return x + self._alpha * (nx - x)

    def optimize(self, f, x, tol=1e-10):
        diff = np.Inf

        while diff > tol:
            nx = self.step(f, x)
            diff = np.abs(f(nx) - f(x))
            x = nx

        return x


class LookAheadOptNN():
    pass
