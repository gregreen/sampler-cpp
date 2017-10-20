#!/usr/bin/env python

import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import corner
import argparse


def load_table(fname):
    with open(fname, 'r') as f:
        labels = f.readline().split()[:-1]
        table = np.loadtxt(f)
    return table, labels


def plot_corner(table, labels=None):
    fig = corner.corner(table[:,:-1], weights=table[:,-1], labels=labels)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Make scatter plots and histograms of '
                    'results from MCMC run.',
        add_help=True)
    parser.add_argument(
        'input',
        metavar='ASCII.txt',
        type=str,
        help='Input filename. Should be an ASCII table.')
    parser.add_argument(
        '--output',
        '-o',
        metavar='PLOT.svg',
        type=str,
        help='Output plot filename.')
    args = parser.parse_args()

    table, labels = load_table(args.input)
    fig = plot_corner(table, labels=labels)

    if args.output is not None:
        fig.savefig(args.output, transparent=False, bbox_inches='tight')
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    main()
