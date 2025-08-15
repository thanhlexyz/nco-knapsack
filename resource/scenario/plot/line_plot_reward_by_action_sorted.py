import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()

def line_plot_reward_by_action_sorted(args):
    # load raw data
    path     = os.path.join(args.train_data_dir, f'knapsack{args.knapsack_n_item}.npz')
    frames   = np.load(path)
    weight   = torch.tensor(frames['weight'][0], dtype=torch.float32)
    value    = torch.tensor(frames['value'][0], dtype=torch.float32)
    capacity = torch.tensor(args.knapsack_n_item * args.knapsack_capacity, dtype=torch.float32)
    # gen all solution
    x = gen_all_binary_vectors(args.knapsack_n_item)
    actions     = np.arange(args.n_action)
    objectives  = (x @ value[:, None]).squeeze()
    constraints = ((x @ weight[:, None] - capacity) <= 0).squeeze()
    rewards     = (objectives * constraints).detach().cpu().numpy()
    # sort metric
    y = torch.sum(x, dim=1)
    indices = torch.argsort(y).detach().cpu().numpy()
    x       = x[indices]
    actions = actions[indices]
    rewards = rewards[indices]
    # plot function landscape
    plt.scatter(np.arange(len(rewards)), rewards, s=1)
    # decorate
    plt.ylabel('reward')
    plt.xlabel('action')
    plt.tight_layout()
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.knapsack_n_item}.pdf')
    plt.savefig(path)
