from . import greedy, optimal, random

def create(args):
    return eval(args.solver).Solver(args)
