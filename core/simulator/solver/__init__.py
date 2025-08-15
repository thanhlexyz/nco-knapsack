from . import random

def create(args):
    return eval(args.solver).Solver(args)
