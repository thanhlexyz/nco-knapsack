from gymnasium import spaces

import gymnasium as gym
import numpy as np
import torch
import os

class Knapsack(gym.Env):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # define search space
        self.observation_space  = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space       = spaces.Discrete(args.n_action)
        self.proto_action_space = spaces.Box(low=0, high=1, shape=(args.n_proto_action,))
        # load data
        self.load_data()
        # intialize
        self.frame = 0
        self.initialize_sorted_action()

    def load_data(self):
        # extract args
        args = self.args
        # get data path
        if args.mode == 'train':
            path = os.path.join(args.train_data_dir, f'knapsack{args.knapsack_n_item}.npz')
        elif args.mode == 'test':
            path = os.path.join(args.train_data_dir, f'knapsack{args.knapsack_n_item}.npz')
        else:
            raise NotImplementedError
        frames       = np.load(path)
        self.weights = frames['weight']
        self.values  = frames['value']

    def initialize_sorted_action(self):
        # extract args
        args = self.args
        # sort by number of 1's in binary representation of the action
        actions             = np.arange(args.n_action)
        counts              = np.array([np.binary_repr(_).count('1') for _ in actions])
        self.sort_indices   = np.argsort(counts)
        self.sorted_actions = actions[self.sort_indices]

    def get_next_observation(self):
        observation = np.concatenate([self.weights[self.frame], self.values[self.frame]])
        return observation

    def decode_action(self, int_action):
        # extract args
        args = self.args
        #
        int_action = self.sorted_actions[int_action]
        str_action = bin(int_action)[2:]
        str_action = str_action.zfill(args.knapsack_n_item)
        np_action  = np.array([int(i) for i in str_action])
        return np_action

    def get_next_reward(self, int_action):
        # extract args
        args     = self.args
        value    = self.values[self.frame]
        weight   = self.weights[self.frame]
        capacity = args.knapsack_n_item * args.knapsack_capacity
        # decode action
        action = self.decode_action(int_action)
        # compute objective value
        objective = np.sum(action * value)
        # compute constraint violation amount
        constraint = np.sum(action * weight) - capacity
        # compute constraint violation flag
        violated = bool(constraint > 0)
        # compute reward
        reward = float(objective if not violated else 0)
        # reward = objective - constraint
        # reward = objective
        # get done
        done = False
        # assemble info
        info = {'objective' : objective,
                'constraint': constraint,
                'violated'  : violated,
                'reward'    : reward}
        return reward, done, info

    def reset(self):
        self.frame = 0
        return self.get_next_observation()

    def step(self, action):
        # get reward
        reward, done, info = self.get_next_reward(action)
        # increate frame counter
        self.frame += 1
        # get next observation
        next_observation = self.get_next_observation()
        return next_observation, reward, done, info
