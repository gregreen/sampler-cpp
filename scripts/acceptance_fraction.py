#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gamma

import matplotlib
import matplotlib.pyplot as plt


def get_integrand(n, beta1, beta2):
    b = 0.5*beta2/beta1
    G = gamma(0.5*n)
    norm = 1.**(0.5*(1.-n)) / G

    def f(x):
        return norm * x**(n-1.) * np.exp(-0.5*x**2.) * gammainc(0.5*n, b*x**2.)

    return f


def estimate_integral_direct():
    for n in range(1,30):
        I,_ = quad(get_integrand(n, 1, 0.99), 0., np.inf)
        print('I_{:d} = {:.5f}'.format(n, I))



def estimate_integral_rejection_method(n, beta1, beta2, samples=10000, sigma=10.):
    u1, u2 = sigma * np.random.random((2,samples))

    print(u1.shape)

    idx = (u2 < np.sqrt(beta2/beta1) * u1)

    U1 = u1**(n-1.) * np.exp(-0.5*u1**2.)
    U2 = u2**(n-1.) * np.exp(-0.5*u2**2.)

    return np.sum(U1[idx]*U2[idx]) / np.sum(U1*U2)


def estimate_integral_rejection_method2(n, beta1, beta2, samples=10000):
    u = np.random.normal(size=(2, samples, n))
    u1, u2 = np.sum(u**2, axis=2)

    print(u1.shape)

    idx = (u2 < np.sqrt(beta2/beta1) * u1)

    return 2. * np.sum(idx) / idx.size


def plot_acceptance_fraction():
    n_range = np.arange(1, 25)
    beta2 = np.logspace(0., -2., 5)

    # print(beta2)
    # return

    fig = plt.figure(figsize=(8,8), dpi=150)
    ax = fig.add_subplot(1,1,1)

    for b in beta2:
        acceptance = [estimate_integral_rejection_method2(n, 1, b, samples=300000)
                      for n in n_range]

        ax.semilogy(n_range, acceptance, label=r'$\beta_2 = {:.2g}$'.format(b))

    ax.set_xlabel(r'$\mathrm{dimensionality}$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{acceptance \ fraction}$', fontsize=14)

    ax.legend()
    ax.grid(True, which='major', axis='both', alpha=0.25)

    ax.set_ylim(0.008, 2.)

    title = r'$\mathrm{Acceptance\ as\ a\ function\ of\ } \beta_2 \ \left( \beta_1 = 1 \right)$'
    ax.set_title(title, fontsize=18)

    fig.savefig('output/pt_acceptance_theoretical.png',
                bbox_inches='tight', transparent=False, dpi=150)
    fig.savefig('output/pt_acceptance_theoretical.pdf',
                bbox_inches='tight', transparent=False, dpi=150)
    plt.show()



def main():
    # for n in range(1,10):
    #     I = estimate_integral_rejection_method2(n, 1, 0.1, samples=100000)
    #     print('I_{:d} = {:.5f}'.format(n, I))

    plot_acceptance_fraction()

    return 0


if __name__ == '__main__':
    main()
