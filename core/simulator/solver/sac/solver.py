import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import simulator
import torch

from .replay_buffer import ReplayBuffer, TrainingData
from .soft_q_network import SoftQNetwork
from .mapper import Mapper
from .actor import Actor

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.create_model()

    def select_action(self, weight, value):
        args       = self.args
        C          = args.capacity
        action     = []
        n_instance = weight.shape[0]
        actor      = self.actor
        qf         = self.qf1
        mapper     = self.mapper
        #
        observation = torch.cat([weight[None, :], value[None, :]], dim=1)
        proto_action, _, _ = actor.get_action(observation)
        action, _ = mapper.get_best_match(proto_action.detach().cpu().numpy(),
                                          observation, qf)
        return action[0]

    def create_model(self):
        # extract args
        args = self.args
        # initialize actor/critic
        self.actor = Actor(args)
        self.qf = SoftQNetwork(args)
        # self.qf1        = SoftQNetwork(args)
        # self.qf2        = SoftQNetwork(args)
        # self.qf1_target = SoftQNetwork(args)
        # self.qf2_target = SoftQNetwork(args)
        # self.qf1_target.load_state_dict(self.qf1.state_dict())
        # self.qf2_target.load_state_dict(self.qf2.state_dict())
        # initialize optimizer
        # self.critic_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.qf_lr)
        self.qf_optimizer = optim.Adam(list(self.qf.parameters()), lr=args.qf_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.actor_lr)
        # initialize mapper
        self.mapper = Mapper(args)
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(args.n_buffer)
        # initialize global step
        self.global_step = 0

    def train_epoch(self):
        # extract args
        replay_buffer = self.replay_buffer
        monitor       = self.monitor
        mapper        = self.mapper
        actor         = self.actor
        args          = self.args
        qf            = self.qf
        info          = {'step': 0, 'objective': 0.0, 'constraint': 0.0, 'reward': 0.0, 'actor_loss': 0.0, 'qf_loss': 0.0}
        n_train_step  = 0
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
            replay_buffer.push(observation[0].cpu().numpy(), discrete_proto_action.detach().cpu().numpy(), reward.item())
            # learning
            if self.global_step > args.n_start_learning:
                actor_loss, qf_loss, alpha_loss = self.optimize()
                n_train_step += 1

            # logging
            info['step']       += 1
            info['objective']  += objective.item()
            info['constraint'] += constraint.item()
            info['reward']     += reward.item()
            if self.global_step % args.n_monitor == 0:
                info['objective']  /= info['step']
                info['constraint'] /= info['step']
                info['reward']     /= info['step']
                if n_train_step >= 1:
                    info['actor_loss'] /= n_train_step
                    info['qf_loss']    /= n_train_step
                monitor.step(info)
                monitor.export_csv()
                info = {'step': 0, 'objective': 0.0, 'constraint': 0.0, 'reward': 0.0, 'actor_loss': 0.0, 'qf_loss': 0.0}
                n_train_step  = 0
            self.global_step += 1

    def optimize(self):
        args          = self.args
        replay_buffer = self.replay_buffer
        self.actor.train()
        self.qf.train()
        # extract batch from buffer
        transitions = replay_buffer.sample(args.batch_size)
        data = TrainingData(transitions, args)
        qf_loss = self.optimize_qf(data)
        actor_loss, alpha_loss = self.optimize_actor(data)
        if self.global_step % args.n_target_update == 0:
            self.update_target()
        return actor_loss, qf_loss, alpha_loss

    def optimize_qf(self, data):
        qf, qf_optimizer = self.qf, self.qf_optimizer
        qv_target = data.reward.squeeze()
        qv = qf(data.observation, data.proto_action).squeeze()
        qf_loss = F.mse_loss(qv, qv_target)
        qf_optimizer.zero_grad()
        qf_loss.backward()
        qf_optimizer.step()
        return qf_loss.item()

    def optimize_actor(self, data):
        return 0, 0

    def update_target(self):
        pass
#         # extract args
#         qf1, qf2, qf1_target, qf2_target = self.qf1, self.qf2, self.qf1_target, self.qf2_target
#         args = self.args
#         # update
#         for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
#             target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
#         for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
#             target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
