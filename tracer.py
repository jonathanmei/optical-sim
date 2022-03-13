import argparse
from collections import defaultdict
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne


N_AIR = 1.0003

def n_BK7(lam):
    # computes index of refraction given wavelength in micrometers
    lam2 = lam**2
    return np.sqrt(
        1 + 1.03961212*lam2/(lam2 - 0.00600069867)
        + 0.231792344*lam2/(lam2 - 0.0200179144)
        + 1.01046945*lam2/(lam2 - 103.560653)
    )

def refract(normal, d_in, n_in, n_out):
    """
    Refracts rays using Snell's Law

    Params:
        normal (`np.ndarray`): [NN, 2] normalized, point toward incident rays
        d_in (`np.ndarray`): [NN, 2] normalized, directions of incident rays
        n_in (`float`): index refraction in medium where rays are
        n_out (`float`): "" where rays are going
    
    Returns:
        `np.ndarray`: [NN, 2] normalized, directions of outgoing rays
    """
    d_perp = d_in - ne.evaluate('sum(d_in * normal, axis=1)')[..., None] * normal
    d_perp_norm = np.sqrt(ne.evaluate('sum(d_perp ** 2, axis=1)'))[..., None]  # [NN, 1]
    e_perp_norm = ne.evaluate('n_in / n_out * d_perp_norm')  # [NN, 1]
    e_n_norm = np.sqrt(1 - e_perp_norm ** 2)  # [NN, 1]
    e_n_grid = ne.evaluate('-normal * e_n_norm')  # [NN, 2]
    warnings.filterwarnings('ignore')
    e_perp_grid = ne.evaluate('d_perp / d_perp_norm * e_perp_norm') # [NN, 2]
    warnings.filterwarnings('default')
    e_perp_grid[d_perp_norm[..., 0] == 0] = 0
    e_grid = e_n_grid + e_perp_grid  # [NN, 2]
    return e_grid

def intersect_sphere(ctr, rad, ori, dir, plus):
    """
    Given a sphere and some rays, find the coordinate of intersection

    solves for t >= 0:
        || (o + t * d) - c ||^2 = r^2

    Params:
        ctr (`np.ndarray`): [2, ] center of sphere
        rad (`float`): radius of sphere
        ori (`np.ndarray`): must broadcast with [NN, 2], ray origins
        dir (`np.ndarray`): [NN, 2] ray directions
        plus (`bool`): use plus or minus in quadratic formula

    Returns:
        `np.ndarray` of [NN, ] of t values
    """
    omc = ori - ctr
    d_dot_omc = ne.evaluate('sum(dir * omc, axis=1)')  # [NN,]
    sq = np.sqrt(
        d_dot_omc ** 2 - np.linalg.norm(omc, axis=-1) ** 2 + rad ** 2
    )  # [NN, ]
    return -d_dot_omc + (sq if plus else -sq)

def trace_biconvex(D1, R1, T, R2, D2, O, lam=500, N=15, M=101, h=35):
    """
    """

    # quick sanity checks
    assert O/2 <= min(R1, R2), "R1, R2 must both be >= O/2"
    T1 = R1 - np.sqrt(R1 ** 2 - O ** 2 / 4)
    T2 = R2 - np.sqrt(R2 ** 2 - O ** 2 / 4)
    assert T1 + T2 <= T, "T must be large enough such that the spherical caps don't intersect"

    # set up the lenses and compute angle bound
    n_lam = n_BK7(lam / 1000)
    c1 = np.array([0, D2 + T - R1])
    c2 = np.array([0, D2 + R2])
    theta_max = np.arctan2(O, 2 * (D1 + T1))

    # rays to trace. This will produce extraneous rays that don't hit the sensor
    s = np.array([0, D1 + T + D2])
    thetas = -theta_max + 2 * theta_max / (N - 1) * np.arange(N)
    d_grid = np.stack(
        [
            np.sin(thetas),
            -np.cos(thetas)
        ],
        axis=-1
    )  # [N, N, 2]
    # pare them down
    hit_lens = d_grid @ [0, -1] > np.cos(theta_max)  # [N, N]
    d_grid = d_grid[hit_lens]  # [NN, 2]

    # source to first intersection
    t_grid = intersect_sphere(c1, R1, s, d_grid, False)  # [NN]
    x1_grid = s + t_grid[..., None] * d_grid  # [NN, 2]

    # first refracction
    n1_grid = x1_grid - c1
    n1_grid = n1_grid / np.linalg.norm(n1_grid, axis=-1, keepdims=1)
    e_grid = refract(n1_grid, d_grid, N_AIR, n_lam)  # [NN, 2]

    # second intersection
    u_grid = intersect_sphere(c2, R2, x1_grid, e_grid, True)  # [NN]
    x2_grid = x1_grid + u_grid[..., None] * e_grid  # [NN, 2]
    # check validity
    valid = (x2_grid[..., -1] <= D2 + T2) & (np.linalg.norm(x2_grid[..., :-1], axis=-1) <= O / 2)
    x2_grid[~valid] = np.nan
    
    # second refraction
    n2_grid = c2 - x2_grid
    n2_grid = n2_grid / np.linalg.norm(n2_grid, axis=-1, keepdims=1)
    f_grid = refract(n2_grid, e_grid, n_lam, N_AIR)  # [NN, 2]

    # hitting home
    v_grid = -x2_grid[..., -1:] / f_grid[..., -1:]  # [NN, 1]
    xy_grid = x2_grid[..., :-1] + v_grid * f_grid[..., :-1]  # [NN, 2] in mm
    # count up rays in each pixel
    ij_grid = ((xy_grid + [h/2]) * M / h)
    # skip rays that don't hit sensor or exited the lens earlier (nan's)
    warnings.filterwarnings('ignore')
    inbounds = np.all((ij_grid >= 0) & (ij_grid < M), axis=-1)
    warnings.filterwarnings('default')
    accum = defaultdict(int)
    for ij in ij_grid[inbounds]:
        ij = ij.astype(np.int)
        accum[tuple(ij)] += 1
    line = np.zeros([M], dtype=np.uint64)
    for ij, val in accum.items():
        line[tuple(ij)] = val
    dist = np.linspace(-M/2, M/2, M)
    image_dist_x, image_dist_y = np.meshgrid(dist, dist)
    image_dist = np.sqrt(image_dist_x ** 2 + image_dist_y ** 2)
    image = np.interp(image_dist, np.arange(M // 2 + (M % 2)), line[M // 2:])
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--D1', type=float, default=10)
    parser.add_argument('--R1', type=float, default=25)
    parser.add_argument('--T', type=float, default=2)
    parser.add_argument('--R2', type=float, default=8)
    parser.add_argument('--D2', type=float, default=10)
    parser.add_argument('--O', type=float, default=8)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--M', type=int, default=None)
    parser.add_argument('--h', type=float, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    kwargs = {k_: args.__dict__[k_] for k_ in ['lam', 'N', 'M', 'h']}
    kwargs = {k_: v_ for k_, v_ in kwargs.items() if v_ is not None}
    psf = trace_biconvex(
        args.D1, args.R1, args.T, args.R2, args.D2, args.O,
        **kwargs
    )

    if args.save_path is None:
        plt.figure()
        plt.imshow(psf)
        plt.show()
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=1)
        plt.imsave(args.save_path, psf)