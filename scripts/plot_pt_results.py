#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import corner
import argparse
import re
import os.path

def load_table(fname):
    with open(fname, 'r') as f:
        beta = float(f.readline().strip().split()[-1])
        labels = f.readline().split()[:-1]
        table = np.loadtxt(f)
    return table, {'labels': labels, 'beta': beta}


def plot_corner(table, labels=None):
    fig = corner.corner(table[:,:-1], weights=table[:,-1], labels=labels)
    return fig


def plot_time_series(tables, betas):
    fig = plt.figure(figsize=(8,2.5*len(betas)), dpi=150)

    # W_max = []

    for i,(t,b) in enumerate(zip(tables, betas)):
        W = np.cumsum(t[:,-1])
        x = t[:,0]

        ax = fig.add_subplot(len(betas),1,1+i)
        if i != len(betas) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$t \ \left( \mathrm{steps} \right)$', fontsize=16)

        ax.set_ylabel(r'$\beta = {:g}$'.format(b), fontsize=16)

        ax.plot(W, x, ls='-', lw=1, alpha=1., c='b')

        ax.set_xlim(0., W[-1])

        # W_max.append(W[-1])
        # print(W[-1])

    # W_max = max(W_max)
    #
    # for ax in fig.get_axes():
    #     ax.set_xlim(0., W_max)

    return fig



def plot_scatter_2D(tables, betas):
    fig = plt.figure(figsize=(8,8), dpi=150)
    ax = fig.add_subplot(1,1,1)

    for i,(t,b) in enumerate(zip(tables, betas)):
        # W = np.cumsum(t[:,-1])
        x = t[:,0]
        y = t[:,1]

        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$y$', fontsize=16)

        cmap = matplotlib.cm.get_cmap('Spectral')

        c = cmap(np.linspace(0., 1., len(betas))[i])[0:3]

        ax.scatter(x, y, edgecolor='none', alpha=0.1, c=c)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Make scatter plots and histograms of '
                    'results from MCMC run.',
        add_help=True)
    parser.add_argument(
        'input',
        metavar='res_beta_1.txt res_beta_0.5.txt ...',
        type=str,
        nargs='+',
        help='Input filename. Should be an ASCII table.')
    parser.add_argument(
        '--output',
        '-o',
        metavar='PLOT.svg',
        type=str,
        help='Output plot filename.')
    args = parser.parse_args()

    fname_base, fname_ext = os.path.splitext(args.output)

    # Load input
    # regex = r'(?:beta_)([0-9]*\.?[0-9]*)(?:\.)'
    # beta = np.array([float(re.findall(regex, fn)[0]) for fn in args.input])
    # idx = np.argsort(beta)
    # table = [load_table(args.input[i])[0] for i in idx]
    table, beta = [], []
    for fn in args.input:
        t,m = load_table(fn)
        table.append(t)
        beta.append(m['beta'])

    beta = np.array(beta)
    idx = np.argsort(beta)
    table = [table[i] for i in idx]
    beta = beta[idx]

    # Plot time series
    fig = plot_time_series(table, beta)

    if args.output is not None:
        fig.savefig(
            fname_base + '_timeseries' + fname_ext,
            transparent=False, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

    # 2D scatter plot
    fig = plot_scatter_2D(table, beta)
    if args.output is not None:
        fig.savefig(
            fname_base + '_scatter2d' + fname_ext,
            transparent=False, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    main()
