import numpy as np
import simulator
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, weight, value):
        args   = self.args
        C      = args.capacity
        n_item = weight.shape[0]
        indices = np.random.permutation(n_item)
        action = torch.zeros_like(weight)
        for j in range(n_item):
            action[indices[j]] = 1
            if torch.sum(weight * action) > C:
                action[indices[j]] = 0
                break
        return action
