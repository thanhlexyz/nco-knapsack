
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

def line_plot_training(args):
    print(f'[+] {args.scenario=} {args.metric=}')
    solvers = [
        'sac',
    ]
    # plot
    lowess = sm.nonparametric.lowess
    for solver in solvers:
        # set args
        args.solver = solver
        # load csv data
        label = f'{args.solver}'
        path = os.path.join(args.train_csv_dir, f'{label}.csv')
        print(path)
        df = pd.read_csv(path)
        y = df[args.metric].to_numpy()
        x = np.arange(len(y))
        smoothed = lowess(y, x, frac=0.2)
        if 'loss' in args.metric:
            plt.plot(y, label=solver)
        else:
            plt.scatter(x, y, s=1, color='orange')
            plt.plot(smoothed[:, 1], label=f'smoothed {solver}')
    # save figure
    plt.legend()
    plt.ylabel(f'{args.metric}')
    plt.xlabel('episode')
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.metric}.pdf')
    plt.savefig(path)
