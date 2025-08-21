import torch.nn.functional as F
import torch.nn as nn
import torch

class SoftQNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.n_observation + args.n_proto_action, args.n_qf_hidden)
        self.fc2 = nn.Linear(args.n_qf_hidden, args.n_qf_hidden)
        self.fc3 = nn.Linear(args.n_qf_hidden, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
