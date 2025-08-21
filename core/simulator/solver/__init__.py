from . import greedy, optimal, random, sac
from . import util

def create(args):
    return eval(args.solver).Solver(args)
