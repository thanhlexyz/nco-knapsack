from . import greedy

def create(args):
    return eval(args.solver).Solver(args)
