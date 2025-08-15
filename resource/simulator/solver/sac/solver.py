from .buffer import ReplayBuffer, Transition
from .normalizer import RunningMeanStd
from .critic import Critic
from .actor import Actor

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import simulator
import pickle
import torch
import time
import os

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize environment
        self.env  = simulator.Knapsack(args)
        # initialize monitor
        self.monitor = simulator.Monitor(args)
        # initialize algorithm
        self.initialize()

    def initialize(self):
        # extract args
        args = self.args
        env  = self.env
        # initialize actor/critic
        self.actor = Actor(env, args)
        self.critic1 = Critic(args)
        self.critic2 = Critic(args)
        self.critic1_target = Critic(args)
        self.critic2_target = Critic(args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # automatic entropy tuning
        self.target_entropy = -torch.tensor(args.n_proto_action).item()
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha          = self.log_alpha.exp().item()
        # initialize optimizer
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=args.sac_critic_lr)
        self.actor_optimizer  = optim.Adam(list(self.actor.parameters()), lr=args.sac_alpha_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.sac_critic_lr)
        # initialize buffer
        self.buffer = ReplayBuffer(args.sac_buffer_size)
        # initialize observation normalizer
        self.observation_normalizer = RunningMeanStd(shape=[args.n_observation])
        self.reward_normalizer      = RunningMeanStd(shape=[1])
        # initialize proto action to action mapper
        self.mapper = simulator.create_mapper(args)

    def optimize(self):
        # extract args
        args             = self.args
        actor            = self.actor
        alpha            = self.alpha # Tunable entropy
        log_alpha        = self.log_alpha # Tunable entropy
        alpha_optimizer  = self.alpha_optimizer
        target_entropy   = self.target_entropy
        buffer           = self.buffer
        critic1          = self.critic1
        critic2          = self.critic2
        critic1_target   = self.critic1_target
        critic2_target   = self.critic2_target
        actor_optimizer  = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        on               = self.observation_normalizer
        rn               = self.reward_normalizer
        # optimization loop
        actor_losses  = []
        critic_losses = []
        # training when enough data
        if self.n_frame > args.sac_start_learning:
            # sample data
            transitions            = buffer.sample(args.n_minibatch)
            batch                  = Transition(*zip(*transitions))
            batch_next_observation = on.normalize_torch(torch.stack(batch.next_observation))
            batch_observation      = on.normalize_torch(torch.stack(batch.observation))
            batch_proto_action     = torch.stack(batch.proto_action)
            batch_action           = torch.stack(batch.action)
            batch_reward           = rn.normalize_torch(torch.stack(batch.reward))
            # early stop critic
            if not self.stop_training_critic:
                # inference target
                with torch.no_grad():
                    next_state_action, next_state_log_prob, _ = actor.get_action(batch_next_observation)
                    critic1_next_target                       = critic1_target(batch_next_observation, next_state_action)
                    critic2_next_target                       = critic2_target(batch_next_observation, next_state_action)
                    min_critic_next_target                    = torch.min(critic1_next_target, critic2_next_target) - alpha * next_state_log_prob
                    next_q_value                              = batch_reward + args.gamma * min_critic_next_target.flatten()
                # inference estimate
                q_value_1    = critic1(batch_observation, batch_proto_action).flatten()
                q_value_2    = critic2(batch_observation, batch_proto_action).flatten()
                critic1_loss = F.mse_loss(q_value_1, next_q_value)
                critic2_loss = F.mse_loss(q_value_2, next_q_value)
                critic_loss  = critic1_loss + critic2_loss
                # optimize the critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                # report data
                critic_losses.append(float(critic_loss.item()))
            # optimize actor with delayed updpate
            if self.n_frame % args.sac_actor_update_interval == 0:
                for _ in range(args.sac_actor_update_interval * args.sac_actor_iter):
                    # actor inference
                    action, log_prob, _ = actor.get_action(batch_observation)
                    q_value_1    = critic1(batch_observation, action)
                    q_value_2    = critic2(batch_observation, action)
                    min_q_value = torch.min(q_value_1, q_value_2)
                    actor_loss = (alpha * log_prob - min_q_value).mean()
                    # optimize actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    # report data
                    actor_losses.append(float(actor_loss.item()))
                    # autotune the entropy
                    with torch.no_grad():
                        _, log_prob, _ = actor.get_action(batch_observation)
                    alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
            # update the target_nework
            if self.n_frame % args.sac_target_update_interval == 0:
                for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                    target_param.data.copy_(args.sac_tau * param.data + (1 - args.sac_tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                    target_param.data.copy_(args.sac_tau * param.data + (1 - args.sac_tau) * target_param.data)
        # compute average
        avg_actor_loss  = np.mean(actor_losses) if len(actor_losses) > 0 else 0
        avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
        return avg_actor_loss, avg_critic_loss

    def train(self):
        # extract args
        env     = self.env
        args    = self.args
        actor   = self.actor
        critic  = self.critic1
        buffer  = self.buffer
        mapper  = self.mapper
        monitor = self.monitor
        rn      = self.reward_normalizer
        on      = self.observation_normalizer
        # loop
        self.n_frame     = 0
        self.n_episode   = 0
        self.stop_training_actor  = False
        self.stop_training_critic = False
        actor_losses     = []
        critic_losses    = []
        best_critic_loss = np.inf
        best_actor_loss  = np.inf
        patience         = 0
        buffer.reset()
        for episode in range(args.n_train_episode):
            # initialize environment
            raw_observation = env.reset()
            if self.n_frame < args.sac_start_learning:
                on.update(raw_observation)
            observation = on.normalize(raw_observation)
            observation = torch.tensor(observation, dtype=torch.float)
            episode_info = {}
            for frame in range(args.n_frame):
                # agent select action
                with torch.no_grad():
                    if self.n_frame < args.sac_start_learning and args.mode == 'train':
                        proto_action = torch.tensor(env.proto_action_space.sample(), dtype=torch.float32)
                        action, discrete_proto_action = mapper.get_best_match(proto_action.detach().cpu().numpy(), observation, critic)
                    else:
                        proto_action, _, _ = actor.get_action(observation.reshape(1, -1))
                        action, discrete_proto_action = mapper.get_best_match(proto_action.detach().cpu().numpy(), observation, critic)

                # send action to environments
                raw_next_observation, raw_reward, done, info = env.step(action)
                for key in info:
                    if key not in episode_info:
                        episode_info[key] = []
                    episode_info[key].append(info[key])

                # buffer management
                if args.mode == 'train':
                    if self.n_frame < args.sac_start_learning:
                        # update observation normalizer
                        on.update(raw_next_observation)
                    # update normalize reward
                    rn.update(raw_reward)
                    # add information to buffer
                    with torch.no_grad():
#                         buffer.push(torch.tensor(raw_observation, dtype=torch.float),
#                                     proto_action.clone().detach().squeeze(),
#                                     torch.tensor(action, dtype=torch.float),
#                                     torch.tensor(raw_next_observation, dtype=torch.float),
#                                     torch.tensor(raw_reward, dtype=torch.float))
                        buffer.push(torch.tensor(raw_observation, dtype=torch.float),
                                    discrete_proto_action.clone().detach().squeeze(),
                                    torch.tensor(action, dtype=torch.float),
                                    torch.tensor(raw_next_observation, dtype=torch.float),
                                    torch.tensor(raw_reward, dtype=torch.float))

                # assign observation = next observation
                raw_observation = raw_next_observation
                observation     = on.normalize(raw_observation)
                observation     = torch.tensor(observation, dtype=torch.float)

                # optimize network
                if args.mode == 'train':
                    actor_loss, critic_loss = self.optimize()
                else:
                    actor_loss, critic_loss = 0, 0
                if actor_loss != 0:
                    actor_losses.append(actor_loss)
                if critic_loss != 0:
                    critic_losses.append(critic_loss)

                # increase frame counter
                self.n_frame += 1
            # update n_episode
            self.n_episode += 1
            # add information to monitor
            info = {'frame'      : self.n_frame,
                    'episode'    : self.n_episode,
                    'buffer_size': len(self.buffer),
                    'actor_loss' : np.mean(actor_losses) if len(actor_losses) > 0 else 0,
                    'critic_loss': np.mean(critic_losses) if len(critic_losses) > 0 else 0,
                    'patience'   : patience}

            for key in episode_info:
                info[key] = np.mean(episode_info[key])

            info['return'] = np.sum(episode_info['reward'])
            monitor.step(info)
            # save simulation data
            monitor.export_csv()

            # save best model
            if not self.stop_training_critic:
                avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else np.inf
                if avg_critic_loss <= best_critic_loss:
                    self.save()
                    best_critic_loss = avg_critic_loss
                    patience         = 0
#                 else:
#                     patience += 1
#                     if patience >= args.sac_patience:
#                         patience = 0
#                         self.stop_training_critic = True
#             elif not self.stop_training_actor:
#                 avg_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else np.inf
#                 if avg_critic_loss <= best_actor_loss:
#                     self.save()
#                     best_actor_loss = avg_actor_loss
#                     patience        = 0
#                 else:
#                     patience += 1
#                     if patience >= args.sac_patience:
#                         self.stop_training_actor = True
#             else:
#                 break


        # final csv_export
        monitor.export_csv()

    def test(self):
        # extract args
        args    = self.args
        env     = self.env
        actor   = self.actor
        critic  = self.critic1
        mapper  = self.mapper
        monitor = self.monitor
        on      = self.observation_normalizer
        # loop many testing episode
        frame = 0
        for episode in range(args.n_test_episode):
            # reset
            observation = env.reset()
            observation = on.normalize(observation)
            observation = torch.tensor(observation, dtype=torch.float)
            # discrete event simulator loop
            for t in range(args.n_frame):
                # compute action
                tic          = time.time()
                # agent select action
                with torch.no_grad():
                    proto_action, _, _ = actor.get_action(observation.reshape(1, -1))
                    action, discrete_proto_action = mapper.get_best_match(proto_action.detach().cpu().numpy(), observation, critic)
                compute_time = time.time() - tic
                # send action to environment
                next_observation, reward, done, info = env.step(action)
                # extract next observation
                observation = on.normalize(next_observation)
                observation = torch.tensor(observation, dtype=torch.float)
                frame += 1
                # add information to monitor
                info['frame']        = frame
                info['compute_time'] = compute_time
                # add info to monitor
                monitor.step(info)
            # save simulation data
            monitor.export_csv()

    def save(self):
        # extract args
        args = self.args
        # create path to save model
        data = {
            'observation_normalizer': self.observation_normalizer,
            'reward_normalizer'     : self.reward_normalizer,
            'actor_state_dict'      : self.actor.state_dict(),
            'critic1_state_dict'    : self.critic1.state_dict(),
            'critic2_state_dict'    : self.critic2.state_dict(),
        }
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        # save model
        with open(path, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f'[+] saved: {path}')

    def load(self):
        # extract args
        args = self.args
        # create path to save model
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fp:
                data = pickle.load(fp)
            self.observation_normalizer = data['observation_normalizer']
            self.reward_normalizer      = data['reward_normalizer']
            try:
                self.actor.load_state_dict(data['actor_state_dict'])
                self.critic1.load_state_dict(data['critic1_state_dict'])
                self.critic2.load_state_dict(data['critic2_state_dict'])
            except:
                pass
            print(f'[+] loaded: {path}')
