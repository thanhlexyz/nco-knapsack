import argparse
import torch
import os

def create_folders(args):
    ls = [args.dataset_dir, args.csv_dir, args.figure_dir, args.checkpoint_dir]
    for folder in ls:
        os.makedirs(folder, exist_ok=True)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    # prepare
    parser.add_argument('--n_train_episode', type=int, default=1000000)
    parser.add_argument('--n_test_episode', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='knapsack')
    parser.add_argument('--n_item', type=int, default=10)
    parser.add_argument('--capacity', type=float, default=1.0) # fraction
    # large discrete action space solver
    parser.add_argument('--mapper', type=str, default='knn')
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--n_proto_action', type=int, default=3)
    # solver
    parser.add_argument('--solver', type=str, default='greedy')
    # sac parameter
    parser.add_argument('--n_buffer', type=int, default=500000) # 1e6
    parser.add_argument('--n_start_learning', type=int, default=10000) # 20000
    parser.add_argument('--n_save', type=int, default=100) # (episodes)
    parser.add_argument('--batch_size', type=int, default=10000) # 64 by CleanRL, 12000 to utilize all RTX 4080
    parser.add_argument('--actor_lr', type=float, default=1e-3) # 3e-4
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--qf_lr', type=float, default=1e-3) # 3e-4
    parser.add_argument('--n_optimize', type=int, default=5)
    parser.add_argument('--n_target_update', type=int, default=1) # 8000
    parser.add_argument('--tau', type=float, default=0.001) # 1.0
    parser.add_argument('--autotune_entropy', action='store_true')
    parser.add_argument('--target_entropy_scale', type=float, default=0.89)
    parser.add_argument('--alpha', type=float, default=0.2) # fixed entropy
    # sac FC model parameter
    parser.add_argument('--n_actor_hidden', type=int, default=128)
    parser.add_argument('--n_qf_hidden', type=int, default=256)
    # data directory
    parser.add_argument('--checkpoint_dir', type=str, default='../data/checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='../data/dataset')
    parser.add_argument('--figure_dir', type=str, default='../figure')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    # logging
    parser.add_argument('--n_monitor', type=int, default=100)
    # plotting
    parser.add_argument('--metric', type=str, default='value')
    parser.add_argument('--n_smooth', type=int, default=1)
    # device
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    # other flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # additional args
    args.n_observation = args.n_item * 2
    args.n_action      = 2 ** args.n_item
    print(f'{args.n_observation=} {args.n_proto_action=} {args.n_action=}')
    return args
