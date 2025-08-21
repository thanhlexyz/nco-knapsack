import numpy as np
import simulator
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, batch):
        weight = batch['weight']
        value  = batch['value']
        args   = self.args
        C      = args.capacity
        action = []
        n_instance, n_item = weight.shape
        for i in range(n_instance):
            w, v = weight[i], value[i]
            indices = np.random.permutation(n_item)
            a = torch.zeros_like(w)
            for j in range(n_item):
                a[indices[j]] = 1
                if torch.sum(w * a) > C:
                    a[indices[j]] = 0
                    break
            action.append(a)
        action = torch.stack(action)
        return action
