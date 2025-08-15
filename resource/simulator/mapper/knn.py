import numpy as np
import itertools
import pyflann
import torch

class Mapper:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize fast library for approximate nearest neighbor
        self.flann         = pyflann.FLANN()
        self.proto_actions = self.generate_proto_actions()
        self.index         = self.flann.build_index(self.proto_actions, algorithm='kdtree')

    def generate_proto_actions(self):
        # extract args
        args = self.args
        # generate all combination
        n_interval_per_proto_action = int(np.ceil(args.n_action ** (1 / args.n_proto_action)))
        proto_actions               = np.array(list(itertools.product(range(n_interval_per_proto_action),
                                               repeat=args.n_proto_action)), dtype=np.float32)
        # normalize to [0, 1]
        proto_actions               = proto_actions / np.max(proto_actions)
        # cut off
        proto_actions               = proto_actions[:args.n_action, :]
        return proto_actions

    def get_best_match(self, proto_action, observation, critic):
        # extract args
        args = self.args
        # obtain k nearest neighbor
        top_k_indices, _    = self.flann.nn_index(proto_action, args.knn_k)
        top_k_proto_actions = self.proto_actions[top_k_indices[0]]
        top_k_proto_actions = torch.tensor(top_k_proto_actions)
        # obtain q-values of top k nearest actions
        observations = torch.tile(observation, [args.knn_k, 1])
        q_values     = critic.forward(observations, top_k_proto_actions)
        # pick neighbor with highest q-value
        idx = int(torch.argmax(q_values).item())
        best_action = int(top_k_indices[0][idx])
        best_proto_action = torch.tensor(self.proto_actions[best_action])
        return best_action, best_proto_action
