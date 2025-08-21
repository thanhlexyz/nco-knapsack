from torch.utils.data import Dataset
import numpy as np
import time
import os

class Dataset(Dataset):

    def __init__(self, mode, args):
        # save args
        self.args = args
        self.mode = mode

    def prepare(self):
        # extract args
        args = self.args
        # check if data exists
        n_instance = eval(f'args.n_{self.mode}_episode')
        label = f'{args.dataset}_{self.mode}_{args.n_item}_{args.capacity}_{n_instance}.npz'
        path = os.path.join(args.dataset_dir, label)
        if os.path.exists(path):
            data        = np.load(path, allow_pickle=True)
            self.weight = np.array(data['weight'])
            self.value  = np.array(data['value'])
        else: # if not, generate new data
            print(f'[+] preparing {label}')
            tic = time.time()
            # generate random weight and value
            self.weight = np.random.rand(n_instance, args.n_item)
            self.value  = np.random.rand(n_instance, args.n_item)
            # save data
            np.savez_compressed(path, weight=self.weight, value=self.value)
            print(f'    - saved {path=}')

    def __len__(self):
        return len(self.weight)

    def __getitem__(self, i):
        item = {'weight': self.weight[i], 'value': self.value[i]}
        return item
