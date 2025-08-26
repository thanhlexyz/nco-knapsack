import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import simulator
import torch

from .replay_buffer import ReplayBuffer, TrainingData
from .constraint_network import ConstraintNetwork
from .soft_q_network import SoftQNetwork
from .mapper import Mapper
from .actor import Actor

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.create_model()

    def create_model(self):
        # extract args
        args = self.args
        # initialize actor/critic
        self.actor = Actor(args).to(args.device)
        self.qf = SoftQNetwork(args).to(args.device)  # q function
        self.cqf = ConstraintNetwork(args).to(args.device) # constraint q function
        # initialize optimizer
        self.qf_optimizer = optim.Adam(list(self.qf.parameters()), lr=args.qf_lr)
        self.cqf_optimizer = optim.Adam(list(self.cqf.parameters()), lr=args.qf_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.actor_lr)
        # initialize automatic entropy tuning
        if args.autotune_entropy:
            self.target_entropy  = -torch.prod(torch.Tensor(args.n_proto_action).to(args.device)).item()
            self.log_alpha       = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha           = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.qf_lr)
        else:
            self.alpha = args.alpha
        # initialize mapper
        self.mapper = Mapper(args)
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(args.n_buffer)
        # initialize global step
        self.global_step = 0

    def select_action(self, weight, value):
        args       = self.args
        C          = args.capacity
        action     = []
        n_instance = weight.shape[0]
        actor      = self.actor
        qf         = self.qf
        cqf        = self.cqf
        mapper     = self.mapper
        #
        observation = torch.cat([weight[None, :], value[None, :]], dim=1).to(args.device)
        proto_action, _, _ = actor.get_action(observation)
        action, _ = mapper.get_best_match(proto_action.detach().cpu().numpy(),
                                          observation, qf, cqf)
        return action[0]

    def train_epoch(self):
        # extract args
        replay_buffer = self.replay_buffer
        monitor       = self.monitor
        mapper        = self.mapper
        actor         = self.actor
        args          = self.args
        cqf           = self.cqf
        qf            = self.qf
        info          = {'step': 0, 'objective': 0.0, 'constraint': 0.0, 'reward': 0.0, 'actor_loss': 0.0, 'qf_loss': 0.0, 'cqf_loss': 0.0, 'alpha_loss': 0.0}
        n_train_step  = 0
        #
        for episode in range(args.n_train_episode):
            # generate random instance
            weight      = torch.from_numpy(np.random.rand(1, args.n_item).astype(dtype=np.float32))
            value       = torch.from_numpy(np.random.rand(1, args.n_item).astype(dtype=np.float32))
            observation = torch.cat([weight, value], dim=1).to(args.device)
            # select action
            proto_action, _, _ = actor.get_action(observation)
            # print(f'{proto_action.mean().item()} {proto_action.min()} {proto_action.max().item()}')
            action, discrete_proto_action = mapper.get_best_match(proto_action.detach().cpu().numpy(), observation, qf, cqf)
            #
            objective = simulator.solver.util.get_objective(value, action)
            constraint = simulator.solver.util.check_contraint(weight, args.capacity, action)
            reward = objective * constraint
            # store replay buffer
            replay_buffer.push(observation[0].cpu().numpy(), reward.item(), constraint.item())
            # learning
            if self.global_step > args.n_start_learning:
                if self.global_step % args.n_optimize == 0:
                    actor_loss, qf_loss, cqf_loss, alpha_loss = self.optimize()
                    info['actor_loss'] += actor_loss
                    info['qf_loss']    += qf_loss
                    info['cqf_loss']   += cqf_loss
                    info['alpha_loss'] += alpha_loss
                    n_train_step += 1
            # logging
            info['step']        = self.global_step
            info['objective']  += objective.item()
            info['constraint'] += constraint.item()
            info['reward']     += reward.item()
            if self.global_step % args.n_monitor == 0:
                info['objective']  /= args.n_monitor
                info['constraint'] /= args.n_monitor
                info['reward']     /= args.n_monitor
                if n_train_step >= 1:
                    info['actor_loss'] /= n_train_step
                    info['qf_loss']    /= n_train_step
                    info['cqf_loss']   /= n_train_step
                    info['alpha_loss'] /= n_train_step
                monitor.step(info)
                monitor.export_csv()
                info = {'step': self.global_step, 'objective': 0.0, 'constraint': 0.0, 'reward': 0.0, 'actor_loss': 0.0, 'qf_loss': 0.0, 'cqf_loss': 0.0, 'alpha_loss': 0.0}
                n_train_step  = 0
            self.global_step += 1

    def optimize(self):
        args = self.args
        replay_buffer = self.replay_buffer
        self.actor.train()
        self.qf.train()
        self.cqf.train()
        # extract batch from buffer
        transitions = replay_buffer.sample(args.batch_size)
        data = TrainingData(transitions, args)
        qf_loss = self.optimize_qf(data)
        cqf_loss = self.optimize_cqf(data)
        actor_loss = self.optimize_actor(data)
        alpha_loss = self.optimize_alpha(data)
        if self.global_step % args.n_target_update == 0:
            self.update_target()
        return actor_loss, qf_loss, cqf_loss, alpha_loss

    def optimize_qf(self, data):
        args = self.args
        qf, qf_optimizer, actor, alpha = self.qf, self.qf_optimizer, self.actor, self.alpha
        observation = data.observation.to(args.device)
        reward = data.reward.squeeze().to(args.device)
        with torch.no_grad():
            proto_action, log_prob, _ = actor.get_action(observation)
        qv_target = reward - alpha * log_prob.squeeze()
        qv = qf(observation, proto_action).squeeze()
        qf_loss = F.mse_loss(qv, qv_target)
        qf_optimizer.zero_grad()
        qf_loss.backward()
        qf_optimizer.step()
        return qf_loss.item()

    def optimize_cqf(self, data):
        args = self.args
        qf, qf_optimizer, actor, alpha = self.cqf, self.cqf_optimizer, self.actor, self.alpha
        loss_fn = nn.CrossEntropyLoss()
        observation = data.observation.to(args.device)
        constraint = data.constraint.squeeze().to(args.device)
        with torch.no_grad():
            proto_action, log_prob, _ = actor.get_action(observation)
        logits = qf(observation, proto_action).squeeze()
        qf_loss = loss_fn(logits, constraint.long())
        qf_optimizer.zero_grad()
        qf_loss.backward()
        qf_optimizer.step()
        return qf_loss.item()

    def optimize_actor(self, data):
        # extract args
        qf, cqf, actor, actor_optimizer, alpha, args = self.qf, self.cqf, self.actor, self.actor_optimizer, self.alpha, self.args
        observation = data.observation.to(args.device)
        # optimize actor
        proto_action, log_prob, _ = actor.get_action(observation)
        qv = qf(observation, proto_action)
        logits = cqf(observation, proto_action)
        mask = torch.argmax(logits, dim=1)
        actor_loss = (alpha * log_prob - qv * mask).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        return actor_loss.item()

    def optimize_alpha(self, data):
        alpha_loss = 0.0
        args = self.args
        # optimize alpha
        if args.autotune_entropy:
            log_alpha, alpha_optimizer, target_entropy, actor = self.log_alpha, self.alpha_optimizer, self.target_entropy, self.actor
            observation = data.observation.to(args.device)
            with torch.no_grad():
                _, log_prob, _ = actor.get_action(observation)
            alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            self.alpha = log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        return alpha_loss

    def update_target(self):
        pass
