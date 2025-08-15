from . import random
from . import test
from . import opt
from . import sac

def create_solver(args):
    if args.solver == 'random':
        return random.Solver(args)
    elif args.solver == 'opt':
        return opt.Solver(args)
    elif args.solver == 'sac':
        return sac.Solver(args)
    elif args.solver == 'test':
        return test.Solver(args)
    else:
        raise NotImplementedError
    return solver
