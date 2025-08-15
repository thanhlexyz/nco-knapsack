import numpy as np
import simulator

def prepare(args):
    dataloader_dict = simulator.dataset.create(args)
