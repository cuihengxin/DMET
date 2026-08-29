"""Finite-difference unit tests for embed_sim/grad/linalg.py (milestone M1).

Run from the repository root:

    python examples/test_example/grad_linalg_test.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from embed_sim.grad.linalg import matfun_grad, eig_subspace_grad

np.random.seed(0)
norm = np.linalg.norm


def random_spd(n):
    a = np.random.random((n, n))
    a = a + a.T
    w, u = np.linalg.eigh(a)
    return (u * (np.abs(w) + 0.5)) @ u.T


def test_matfun_grad(n=6, delta=1e-5):
    """dE/dS for E = sum B*f(S), f = S^{-1/2} and S^{+1/2}."""
    S = random_spd(n)
    dS = np.random.random((n, n))
    dS = dS + dS.T
    dS *= delta / norm(dS)
    B = np.random.random((n, n))

    w, u = np.linalg.eigh(S)
    for f, fp, name in ((lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5, 'S^-1/2'),
                        (lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5, 'S^+1/2')):
        G = matfun_grad(w, u, B, f, fp)
        ana = np.einsum('ij,ij->', G, dS)

        def fval(M):
            wm, um = np.linalg.eigh(M)
            return np.einsum('ij,ij->', B, (um * f(wm)) @ um.T)

        num = (fval(S + dS) - fval(S - dS)) / 2
        err = abs(ana - num) / abs(num)
        print(f'  matfun_grad {name:8s}: analytic {ana: .10e}  '
              f'numerical {num: .10e}  rel.err {err:.2e}')
        assert err < 1e-6, name


def test_eig_subspace_grad(n=8, nbath=3, nfo=2, delta=1e-5):
    """dE/dM for E depending on the bath/frozen-occupied eigen-subspaces only.

    The model energy is built so that it is manifestly invariant under
    rotations inside each block: E = tr(Pi_bath A) + tr(Pi_fo B).
    """
    M = random_spd(n)
    dM = np.random.random((n, n))
    dM = dM + dM.T
    dM *= delta / norm(dM)
    A = np.random.random((n, n)); A = A + A.T
    Bm = np.random.random((n, n)); Bm = Bm + Bm.T

    w, V = np.linalg.eigh(M)
    order = np.argsort(-w)
    fo_sel = order[:nfo]                       # highest eigenvalues
    bath_sel = order[nfo:nfo + nbath]
    fv_sel = order[nfo + nbath:]
    group = np.empty(n, dtype=int)
    group[bath_sel] = 0
    group[fo_sel] = 1
    group[fv_sel] = 2

    def energy(mat):
        wm, Vm = np.linalg.eigh(mat)
        om = np.argsort(-wm)
        fo = Vm[:, om[:nfo]]
        ba = Vm[:, om[nfo:nfo + nbath]]
        return (np.einsum('ij,ij->', ba @ ba.T, A)
                + np.einsum('ij,ij->', fo @ fo.T, Bm))

    # dE/dv_k = 2 A v_k for bath, 2 B v_k for frozen occupied
    bmat = np.zeros((n, n))
    bmat[:, bath_sel] = 2 * A @ V[:, bath_sel]
    bmat[:, fo_sel] = 2 * Bm @ V[:, fo_sel]

    G = eig_subspace_grad(w, V, bmat, group)
    ana = np.einsum('ij,ij->', G, dM)
    num = (energy(M + dM) - energy(M - dM)) / 2
    err = abs(ana - num) / abs(num)
    print(f'  eig_subspace_grad     : analytic {ana: .10e}  '
          f'numerical {num: .10e}  rel.err {err:.2e}')
    assert err < 1e-6


if __name__ == '__main__':
    print('M1: matrix-derivative primitives vs central differences')
    test_matfun_grad()
    test_eig_subspace_grad()
    print('all linalg gradient tests passed')
