from collections import namedtuple, deque
import numpy as np
import random
import torch

Transition = namedtuple('Transition',
                        ('observation', 'proto_action', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory   = deque([], maxlen=capacity)
        self.capacity = capacity

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class TrainingData:

    def __init__(self, transitions, args):
        batch                 = Transition(*zip(*transitions))
        self.observation      = torch.from_numpy(np.stack(batch.observation)).to(device=args.device, dtype=torch.float32)
        self.proto_action     = torch.from_numpy(np.stack(batch.proto_action)).to(device=args.device, dtype=torch.long)
        self.reward           = torch.tensor(batch.reward, device=args.device, dtype=torch.float32)
