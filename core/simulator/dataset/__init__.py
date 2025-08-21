from torch.utils.data import DataLoader
from . import knapsack
import os

def create(args):
    dataloader_dict = {}
    # for mode in ['train', 'test']:
    for mode in ['test']:
        dataset = eval(args.dataset).Dataset(mode, args)
        dataset.prepare()
        dataloader_dict[mode] = DataLoader(dataset, batch_size=1, shuffle=mode=='train', num_workers=os.cpu_count())
    return dataloader_dict
