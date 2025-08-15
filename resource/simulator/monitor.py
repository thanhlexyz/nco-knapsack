import pandas as pd
import numpy as np
import torch
import tqdm
import os

class Monitor:

    def __init__(self, args, progress_bar=True):
        # save
        self.args = args
        # initialize progress bar
        if progress_bar:
            if args.mode == 'train':
                T = args.n_train_episode
            elif args.mode == 'test':
                T = args.n_frame * args.n_test_episode
            if progress_bar:
                self.bar = tqdm.tqdm(range(T))
            # initialize writer
            self.csv_data = {}
            self.global_step = 0

    def __update_time(self):
        self.bar.update(1)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if key in ['reward', 'objective', 'constraint']:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'loss' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
        self.bar.set_postfix(**_kwargs)

    def __display(self):
        self.bar.display()

    def step(self, info):
        # extract stats from all stations
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += 1

    ####################################################################################
    # MODIFY HERE
    ####################################################################################
    @property
    def label(self):
        args = self.args
        label = f'{args.solver}'
        return label

    def __update_csv(self, info):
        for key in info.keys():
            if 'n_user_per_action' in key:
                pass
            else:
                if key not in self.csv_data:
                    self.csv_data[key] = [float(info[key])]
                else:
                    self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        # extract args
        args = self.args
        # save data to csv
        if args.mode == 'train':
            path = os.path.join(args.train_csv_dir, f'{self.label}.csv')
        elif args.mode == 'test':
            path = os.path.join(args.test_csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)
