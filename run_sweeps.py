import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np

from tracer import trace_biconvex


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vary_D1', action='store_true')
    parser.add_argument('--vary_D1_fine', action='store_true')
    parser.add_argument('--vary_D2', action='store_true')
    parser.add_argument('--vary_lam', action='store_true')
    parser.add_argument('--vary_N', action='store_true')

    args = parser.parse_args()

    # some defaults
    D1, R1, T, R2, D2, O, lam, N, M, h = (
        150, 100, 25, 100, 150, 25, 500, 5001, 151, 15
    )
    os.makedirs('out', exist_ok=1)

    # vary distances to see aberrations:
    if args.vary_D1:
        D1_range = range(150, 500, 25)
        varied_D1s = [
            trace_biconvex(D1_, R1, T, R2, D2, O, lam, N, M, h)
            for D1_ in D1_range
        ]
        for D1_, im_ in zip(D1_range, varied_D1s):
            plt.imsave('out/vary_D1_%03d.png' % D1_, im_)
    if args.vary_D1_fine:
        D1_range = range(200, 300, 5)
        varied_D1s = [
            trace_biconvex(D1_, R1, T, R2, D2, O, lam, N, M, h=5)
            for D1_ in D1_range
        ]
        for D1_, im_ in zip(D1_range, varied_D1s):
            plt.imsave('out/vary_D1_fine_%03d.png' % D1_, im_)
    
    if args.vary_D2:
        D2_range = range(25, 500, 25)
        varied_D2s = [
            trace_biconvex(D1, R1, T, R2, D2_, O, lam, N, M, h)
            for D2_ in D2_range
        ]
        for D2_, im_ in zip(D2_range, varied_D2s):
            plt.imsave('out/vary_D2_%03d.png' % D2_, im_)

    # vary lambda:
    if args.vary_lam:
        lam_range = range(400, 725, 25)
        varied_lams = [
            trace_biconvex(D1, R1, T, R2, D2, O, lam_, N, M, h)
            for lam_ in lam_range
        ]
        for lam_, im_ in zip(lam_range, varied_lams):
            plt.imsave('out/vary_lam_%03d.png' % lam_, im_)

    # vary ray sampling:
    if args.vary_N:
        N_range = np.logspace(np.log10(15), np.log10(5001), 25).astype(np.int)
        varied_Ns = [
            trace_biconvex(D1, R1, T, R2, D2, O, lam, N_, M, h)
            for N_ in N_range
        ]
        for N_, im_ in zip(N_range, varied_Ns):
            plt.imsave('out/vary_N_%04d.png' % N_, im_)
