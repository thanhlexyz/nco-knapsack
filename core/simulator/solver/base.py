import simulator
import torch
import tqdm

from . import util

class BaseSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # load data
        self.dataloader_dict = simulator.dataset.create(args)
        # load monitor
        self.monitor = simulator.Monitor(args)

    def select_action(self, weight, value):
        raise NotImplementedError

    def test(self):
        # extract args
        self.load()
        monitor, args = self.monitor, self.args
        info = self.test_epoch()
        monitor.step(info)
        monitor.export_csv()

    def test_epoch(self):
        args = self.args
        loader = self.dataloader_dict['test']
        info = {'step': 0, 'objective': 0.0, 'constraint': 0.0}
        # step loop
        for batch in tqdm.tqdm(loader):
            weight, value = batch['weight'][0], batch['value'][0]
            # select action
            action = self.select_action(weight, value)
            # evaluation
            objective = util.get_objective(value, action)
            constraint = util.check_contraint(weight, args.capacity, action)
            info['step']       += 1
            info['objective']  += objective.item()
            info['constraint'] += constraint.item()
        info['objective']  /= info['step']
        info['constraint'] /= info['step']
        return info

    def train(self):
        # extract args
        self.load()
        monitor, args = self.monitor, self.args
        info = self.train_epoch()
        monitor.step(info)
        monitor.export_csv()

    def train_epoch(self):
        pass

    def load(self):
        pass
