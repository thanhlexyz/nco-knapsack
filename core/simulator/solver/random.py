import numpy as np
import simulator

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        n_sub, n_var, n_feature = observation.shape
        active = observation[:, :, -1]
        action = []
        for i in range(n_sub):
            indices = np.where(active[i] == 1)[0]
            j = np.random.choice(indices)
            action.append(j)
        return action
