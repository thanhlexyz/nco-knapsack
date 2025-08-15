import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def box_plot(args):
    print(f'[+] {args.scenario=} {args.metric=}')
    solvers = [
        'random',
        'sac',
        'opt',
    ]
    Y = []
    exist_solvers = []
    for solver in solvers:
        # set args
        args.solver = solver
        # load csv data
        label = f'{args.solver}'
        path = os.path.join(args.test_csv_dir, f'{label}.csv')
        if os.path.exists(path):
            print(path)
            df = pd.read_csv(path)
            y = df[args.metric].to_numpy()
            Y.append(y)
            exist_solvers.append(solver)
    # plot
    fig, ax = plt.subplots()
    ax.boxplot(Y, showfliers=True)
    ax.set_xticklabels(exist_solvers)
    # decorate
    plt.ylabel(f'{args.metric}')
    plt.xlabel('solver')
    if args.metric == 'constraint':
        plt.axhline(0)
    # save figure
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.metric}.pdf')
    plt.savefig(path)
