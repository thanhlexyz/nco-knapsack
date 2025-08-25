import torch

def get_objective(value, action):
    return torch.sum(value * action)

def check_contraint(weight, capacity, action):
    return (torch.sum(weight * action) < capacity).to(torch.float32)

def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()
