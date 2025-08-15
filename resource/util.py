import argparse
import torch
import os

def create_folders(args):
    ls = [args.train_csv_dir, args.test_csv_dir, args.figure_dir,
          args.train_data_dir, args.test_data_dir, args.model_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_default_device(args):
    torch.set_default_device(args.device)

base_folder = os.path.dirname(os.path.dirname(__file__))

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    # duration
    parser.add_argument('--n_train_episode', type=int, default=10000)
    parser.add_argument('--n_test_episode', type=int, default=1)
    parser.add_argument('--n_frame', type=int, default=100)
    # knapsack environment
    parser.add_argument('--knapsack_n_item', type=int, default=10)
    parser.add_argument('--knapsack_capacity', type=float, default=0.25) # fraction
    # solver
    parser.add_argument('--solver', type=str, default='random')
    # mdp
    parser.add_argument('--gamma', type=float, default=0.0)
    # sac
    parser.add_argument('--sac_tau', type=float, default=0.005) # target smoothing coeff
    parser.add_argument('--sac_patience', type=int, default=5) # early stop
    parser.add_argument('--sac_n_actor_hidden', type=int, default=128)
    parser.add_argument('--sac_n_critic_hidden', type=int, default=256)
    parser.add_argument('--sac_alpha_lr', type=float, default=3e-4)
    parser.add_argument('--sac_actor_lr', type=float, default=5e-4)
    parser.add_argument('--sac_critic_lr', type=float, default=5e-4)
    parser.add_argument('--sac_actor_iter', type=int, default=1)
    parser.add_argument('--sac_buffer_size', type=int, default=10000000)
    parser.add_argument('--sac_start_learning', type=int, default=1000) # frames
    parser.add_argument('--sac_actor_update_interval', type=int, default=2)
    parser.add_argument('--sac_target_update_interval', type=int, default=1)
    parser.add_argument('--n_minibatch', type=int, default=1000)
    # large discrete action space solver
    parser.add_argument('--mapper', type=str, default='knn')
    parser.add_argument('--knn_k', type=int, default=25)
    parser.add_argument('--n_proto_action', type=int, default=5)
    # plot
    parser.add_argument('--metric', type=str, default='reward')
    # data directory
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--train_data_dir', type=str, default='../data/train/data')
    parser.add_argument('--test_data_dir', type=str, default='../data/test/data')
    parser.add_argument('--train_csv_dir', type=str, default='../data/train/csv')
    parser.add_argument('--test_csv_dir', type=str, default='../data/test/csv')
    # other
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # set default device cuda
    set_default_device(args)
    # additional args
    args.n_observation = args.knapsack_n_item * 2
    args.n_action      = 2 ** args.knapsack_n_item
    print(f'{args.n_observation=} {args.n_proto_action=} {args.n_action=}')
    return args
