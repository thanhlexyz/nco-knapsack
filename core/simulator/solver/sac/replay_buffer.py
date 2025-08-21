from collections import namedtuple, deque
import numpy as np
import random

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
