from . import knn

def create_mapper(args):
    if args.mapper == 'knn':
        return knn.Mapper(args)
    else:
        raise NotImplementedError
    return solver
