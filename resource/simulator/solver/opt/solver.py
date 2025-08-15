import numpy as np
import simulator
import torch
import time

def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize environment
        self.env = simulator.Knapsack(args)
        # initialize monitor
        self.monitor = simulator.Monitor(args)
        # gen all solution
        self.x = gen_all_binary_vectors(args.knapsack_n_item)
        # sort all solution by number of 1's in binary representation
        self.x   = self.x[torch.tensor(self.env.sort_indices)]

    def select_action(self):
        # extract args
        args     = self.args
        env      = self.env
        value    = torch.tensor(env.values[env.frame], dtype=torch.float32)
        weight   = torch.tensor(env.weights[env.frame], dtype=torch.float32)
        capacity = torch.tensor(args.knapsack_n_item * args.knapsack_capacity, dtype=torch.float32)
        # compute all objective and constraint
        objective  = (self.x @ value[:, None]).squeeze()
        constraint = ((self.x @ weight[:, None] - capacity) <= 0).squeeze()
        reward     = objective * constraint
        action     = int(torch.argmax(reward).item())
        return action

    def test(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # repeat for many episode
        for episode in range(args.n_test_episode):
            # reset
            observation = env.reset()
            # discrete event simulator loop for one episode
            for t in range(args.n_frame):
                # compute action
                tic = time.time()
                action = self.select_action()
                compute_time = time.time() - tic
                # send action to environment
                next_observation, reward, done, info = env.step(action)
                # add info to monitor
                info['compute_time'] = compute_time
                monitor.step(info)
        # save simulation data
        monitor.export_csv()
#         # print optimal action
#         opt_action         = self.select_action()
#         decoded_opt_action = env.decode_action(opt_action)
#         print(f'{decoded_opt_action=} {opt_action=}')

    def train(self):
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass
