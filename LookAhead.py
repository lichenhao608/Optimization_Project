import numpy as np
from methods import gradient
from torch.optim import Optimizer
from collections import defaultdict


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


class LookAheadOptNN(Optimizer):
    '''Implements lookahead optimizer by Michael R. Zhang
    Paper: zhang2019lookahead,
            title={Lookahead Optimizer: k steps forward, 1 step back},
            author={Michael R. Zhang and James Lucas and Geoffrey Hinton and Jimmy Ba},
            year={2019},
            eprint={1907.08610},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
    Link: https://arxiv.org/abs/1907.08610

    Args:
        opt (Optimizer): Another optimizer as the based optimizer to run.
        k (int, optional): Synchronization period.
        alpha (float, optional): slow weight step size.
    '''

    def __init__(self, opt, k=5, alpha=0.5):
        assert isinstance(opt, Optimizer), 'Should have a optimizer method'

        self.opt = opt
        self.k = k
        self.alpha = alpha
        self.param_groups = self.opt.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.opt.state

    def step(self, closure=None):
        '''Performs a single optimization step

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        '''
        for _ in range(k):
            loss = self.opt.step(closure)

        for group in self.opt.param_groups:
            for fast in group['params']:
                param_state = self.state[fast]
                if "slow_param" not in param_state:
                    param_state['slow_param'] = torch.zeros_like(fast.data)
                slow = param_state['slow_param']
                slow += (fast.data - slow) * self.alpha
                fast.data.copy(slow)

        return loss

    def state_dict(self):
        fast_state_dict = self.opt.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(LookAheadOptNN, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
