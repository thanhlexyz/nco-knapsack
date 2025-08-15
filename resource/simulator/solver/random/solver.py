import numpy as np
import simulator
import time

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize environment
        self.env = simulator.Knapsack(args)
        # initialize monitor
        self.monitor = simulator.Monitor(args)

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
                # action = self.select_action()
                action = env.action_space.sample()
                compute_time = time.time() - tic
                # send action to environment
                next_observation, reward, done, info = env.step(action)
                # add info to monitor
                info['compute_time'] = compute_time
                monitor.step(info)
        # save simulation data
        monitor.export_csv()

    def train(self):
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass
