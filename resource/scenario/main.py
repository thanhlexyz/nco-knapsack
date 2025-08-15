import numpy as np
import simulator

def main(args):
    solver = simulator.create_solver(args)
    if args.mode == 'test':
        solver.load()
        solver.test()
    elif args.mode == 'train':
        solver.train()
        solver.save()
    else:
        raise NotImplementedError
