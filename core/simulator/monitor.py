import pandas as pd
import tqdm
import os

class Monitor:
    def __init__(self, args):
        # save
        self.args = args
        # initialize writer
        self.csv_data = {}
        self.global_step = 0
        self.bar = None

    def create_progress_bar(self, n_step):
        # initialize progress bar
        self.bar = tqdm.tqdm(range(n_step))

    def __update_time(self):
        if self.bar:
            self.bar.update(self.args.n_monitor)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            for term in ['loss', 'cost', 'objective', 'reward', 'constraint']:
                if term in key:
                    _kwargs[key] = f'{kwargs[key]:0.8f}'
        if self.bar:
            self.bar.set_postfix(**_kwargs)

    def __display(self):
        if self.bar:
            self.bar.display()
            print()

    def step(self, info):
        # extract stats from all stations
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += self.args.n_monitor

    ####################################################################################
    # MODIFY HERE
    ####################################################################################
    @property
    def label(self):
        args = self.args
        label = f'{args.dataset}_{args.n_item}_{args.solver}'
        return label

    def __update_csv(self, info):
        for key in info.keys():
            if key not in self.csv_data:
                self.csv_data[key] = [float(info[key])]
            else:
                self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        # extract args
        args = self.args
        d = os.path.join(args.csv_dir, args.mode)
        if not os.path.exists(d):
            os.makedirs(d)
        # save data to csv
        path = os.path.join(d, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)
