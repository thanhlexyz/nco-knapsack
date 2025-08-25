import simulator
import torch

from .base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        # generate all solution
        self.x = simulator.solver.util.gen_all_binary_vectors(args.n_item)

    def select_action(self, weight, value):
        args   = self.args
        C      = args.capacity
        n_item = weight.shape[0]
        # compute all objective and constraint
        objective  = (self.x @ value[:, None]).squeeze()
        constraint = ((self.x @ weight[:, None] - C) <= 0).squeeze()
        reward     = objective * constraint
        action     = self.x[int(torch.argmax(reward).item())]
        return action
