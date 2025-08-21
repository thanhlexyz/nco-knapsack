import torch.optim as optim
import numpy as np
import simulator
import torch

from .soft_q_network import SoftQNetwork
from .replay_buffer import ReplayBuffer
from .mapper import Mapper
from .actor import Actor

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.create_model()

    def select_action(self, batch):
        weight = batch['weight']
        value  = batch['value']
        args   = self.args
        C      = args.capacity
        action = []
        n_instance, n_item = weight.shape
        actor  = self.actor
        qf     = self.qf1
        mapper = self.mapper
        #
        observation = torch.cat([batch['weight'], batch['value']], dim=1)
        proto_action, _, _ = actor.get_action(observation)
        action, _ = mapper.get_best_match(proto_action.detach().cpu().numpy(),
                                          observation, qf)
        return action

    def create_model(self):
        # extract args
        args = self.args
        # initialize actor/critic
        self.actor      = Actor(args)
        self.qf1        = SoftQNetwork(args)
        self.qf2        = SoftQNetwork(args)
        self.qf1_target = SoftQNetwork(args)
        self.qf2_target = SoftQNetwork(args)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # initialize optimizer
        self.critic_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.qf_lr)
        self.actor_optimizer  = optim.Adam(list(self.actor.parameters()), lr=args.actor_lr)
        # initialize mapper
        self.mapper = Mapper(args)
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(args.n_buffer)

    def train_epoch(self):
        # extract args
        replay_buffer = self.replay_buffer
        mapper        = self.mapper
        actor         = self.actor
        args          = self.args
        qf            = self.qf1
        #
        for episode in range(args.n_train_episode):
            # generate random instance
            weight      = torch.from_numpy(np.random.rand(1, args.n_item).astype(dtype=np.float32))
            value       = torch.from_numpy(np.random.rand(1, args.n_item).astype(dtype=np.float32))
            observation = torch.cat([weight, value], dim=1)
            # select action
            proto_action, _, _ = actor.get_action(observation)
            action, discrete_proto_action = mapper.get_best_match(proto_action.detach().cpu().numpy(), observation, qf)
            #
            objective = simulator.solver.util.get_objective(value, action)
            constraint = simulator.solver.util.check_contraint(weight, args.capacity, action)
            reward = objective * constraint
            # store replay buffer
            replay_buffer.push(observation[0], discrete_proto_action, reward)
            #
            if self.global_step > args.n_start_learning:
