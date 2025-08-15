import numpy as np
import os

def prepare_knapsack(args):
    # generate random weight and value
    weight = np.random.rand(args.n_frame + 1, args.knapsack_n_item)
    value  = np.random.rand(args.n_frame + 1, args.knapsack_n_item)
    # prepare and save
    if args.mode == 'train':
        path = os.path.join(args.train_data_dir, f'knapsack{args.knapsack_n_item}.npz')
    elif args.mode == 'test':
        path = os.path.join(args.test_data_dir, f'knapsack{args.knapsack_n_item}.npz')
    else:
        raise NotImplementedError
    kwargs = {'weight': weight, 'value': value}
    np.savez_compressed(path, **kwargs)
