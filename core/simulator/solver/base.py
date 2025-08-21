import simulator
import torch

from . import util

class BaseSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # load data
        self.dataloader_dict = simulator.dataset.create(args)
        # load monitor
        self.monitor = simulator.Monitor(args)

    def select_action(self, observation):
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
        for batch in loader:
            # select action
            action = self.select_action(batch)
            # evaluation
            objective = util.get_objective(batch['value'], action)
            constraint = util.check_contraint(batch['weight'], args.capacity, action)
            info['step']       += len(objective)
            info['objective']  += objective.sum().item()
            info['constraint'] += constraint.sum().item()
        info['objective']  /= info['step']
        info['constraint'] /= info['step']
        return info

    def load(self):
        pass
