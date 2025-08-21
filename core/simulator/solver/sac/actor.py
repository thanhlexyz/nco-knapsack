import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1       = nn.Linear(args.n_observation, args.n_actor_hidden)
        self.relu1     = nn.ReLU()
        self.fc2       = nn.Linear(args.n_actor_hidden, args.n_actor_hidden)
        self.relu2     = nn.ReLU()
        self.fc_mean   = nn.Linear(args.n_actor_hidden, args.n_proto_action)
        self.fc_logstd = nn.Linear(args.n_actor_hidden, args.n_proto_action)
        # proto action rescaling
        high = np.ones(args.n_proto_action) # env.proto_action_space.high
        low  = np.zeros(args.n_proto_action) # env.proto_action_space.low
        self.register_buffer('action_scale', torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
