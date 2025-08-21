import simulator
import torch

from .base import BaseSolver

def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        # generate all solution
        self.x = gen_all_binary_vectors(args.n_item)

    def select_action(self, batch):
        weight = batch['weight']
        value  = batch['value']
        args   = self.args
        C      = args.capacity
        action = []
        n_instance, n_item = weight.shape
        for i in range(n_instance):
            w, v = weight[i], value[i]
            # compute all objective and constraint
            objective  = (self.x @ v[:, None]).squeeze()
            constraint = ((self.x @ w[:, None] - C) <= 0).squeeze()
            reward     = objective * constraint
            a          = self.x[int(torch.argmax(reward).item())]
            action.append(a)
        action = torch.stack(action)
        return action
