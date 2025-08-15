import simulator
import torch

class BaseSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # load monitor
        self.monitor = simulator.Monitor(args)

    def select_action(self, observation):
        raise NotImplementedError

    def test(self):
        # extract args
        self.load()
        monitor, args = self.monitor, self.args
        monitor.create_progress_bar(args.n_test_episode)
        # test each property
        for _ in range(args.n_test_episode):
            info = self.test_episode()
            monitor.step(info)
            monitor.export_csv()

    def test_episode(self):
        # extract args
        args = self.args
        env  = self.env
        # reset environment
        observation, done = env.reset()
        info = {'episode': env.episode, 'value': 0}
        self.init_hook(observation)
        # step loop
        while not done:
            with torch.no_grad():
                action = self.select_action(observation)
            next_observation, reward, done, step_info = env.step(action)
            self.step_hook(next_observation)
            info['value'] += reward
            observation = next_observation
        return info

    def train(self):
        # extract args
        self.load()
        monitor, args = self.monitor, self.args
        monitor.create_progress_bar(args.n_train_episode * args.n_train_epoch)
        # test each property
        for episode in range(args.n_train_episode * args.n_train_epoch):
            try:
                info = self.train_episode()
                monitor.step(info)
                monitor.export_csv()
                if episode % args.n_save == 0 and episode > 0:
                    self.save()
            except KeyboardInterrupt:
                pass
        self.save()


    def init_hook(self, observation):
        pass

    def step_hook(self, next_observation):
        pass

    def load(self):
        pass

    def save(self):
        raise NotImplementedError
