"""Matrix-derivative primitives used by the SSDMET analytic nuclear gradient.

Two objects are needed when differentiating the DMET orbital construction:

1. :func:`matfun_grad` -- derivative of a symmetric matrix function
   ``F = f(S)`` (here ``S^{-1/2}`` and ``S^{1/2}`` of the AO overlap) with
   respect to ``S``.  Implemented with the Daleckii-Krein divided-difference
   formula, which degrades gracefully to ``f'`` on (near-)degenerate
   eigenvalues of ``S`` instead of dividing by zero.

2. :func:`eig_subspace_grad` -- derivative of the eigenvectors of the
   environment block of the LO density matrix.  The DMET energy only depends on
   the *subspaces* spanned by the bath / frozen-occupied blocks, never on the
   individual eigenvectors, so contributions from eigenvector pairs inside the
   same block cancel exactly.  We drop them explicitly, which removes the
   1/(lambda_k - lambda_l) divergence for degenerate bath (or frozen) orbitals
   and makes the result independent of the ``es_natorb`` gauge.

Both routines are validated against central finite differences in
``examples/test_example/dmet_grad_linalg_test.py``.
"""

import numpy as np


def matfun_grad(w, u, B, f, fp, degen_tol=1e-8):
    """d/dS of ``sum_{uv} B[u,v] f(S)[u,v]`` for a symmetric matrix ``S``.

    Args:
        w, u: eigenvalues / eigenvectors of ``S`` (``S = u diag(w) u^T``).
        B: (nao, nao) array, ``dE/dF`` with ``F = f(S)``.  Need not be
            symmetric.
        f, fp: callables giving ``f(w)`` and ``f'(w)`` element-wise.
        degen_tol: eigenvalues closer than this are treated as degenerate and
            the divided difference is replaced by the derivative.

    Returns:
        symmetric (nao, nao) array ``G`` with ``dE = sum_{uv} G[u,v] dS[u,v]``.
    """
    fw = np.asarray(f(w))
    fpw = np.asarray(fp(w))

    num = fw[None, :] - fw[:, None]          # f(w_j) - f(w_i)
    den = w[None, :] - w[:, None]            # w_j - w_i
    degen = np.abs(den) < degen_tol
    dd = np.where(degen, 1.0, den)
    dd = num / dd
    # Daleckii-Krein: the diagonal (and any degenerate pair) is f'
    dd = np.where(degen, 0.5 * (fpw[:, None] + fpw[None, :]), dd)

    theta = (u.T @ B @ u) * dd
    G = u @ theta @ u.T
    return 0.5 * (G + G.T)


def eig_subspace_grad(w, v, bmat, group, degen_tol=1e-8, log=None):
    """d/dM of ``sum_k b_k . v_k`` for the eigenvectors of a symmetric ``M``.

    Args:
        w, v: eigenvalues / eigenvectors of ``M`` (``M = v diag(w) v^T``).
        bmat: (n, n) array whose column ``k`` is ``dE/dv_k``.  Columns
            belonging to orbitals that do not enter the energy (e.g. frozen
            virtuals) must be zero.
        group: (n,) integer labels.  Eigenvector pairs carrying the *same*
            label are skipped: the energy is invariant under rotations inside
            a block, so those contributions cancel analytically.
        degen_tol: pairs from different blocks closer than this trigger a
            warning (the bath/frozen split is ill-defined there).

    Returns:
        symmetric (n, n) array ``G`` with ``dE = sum_ij G[i,j] dM[i,j]``.
    """
    group = np.asarray(group)
    # T[l,k] = v_l . b_k
    T = v.T @ bmat
    # den[l,k] = w_k - w_l
    den = w[None, :] - w[:, None]
    cross = group[:, None] != group[None, :]
    small = np.abs(den) < degen_tol

    risky = cross & small & (np.abs(T) > 1e-10)
    if risky.any() and log is not None:
        pairs = np.argwhere(risky)
        log.warn('SSDMET gradient: near-degenerate eigenvalues across the '
                 'bath/frozen boundary (%d pair(s), e.g. occ %.3e vs %.3e). '
                 'The gradient of the bath selection is ill-conditioned here.',
                 len(pairs), w[pairs[0, 1]], w[pairs[0, 0]])

    W = np.where(cross & ~small, 1.0 / np.where(small, 1.0, den), 0.0)
    A = T * W
    G = v @ A @ v.T
    return 0.5 * (G + G.T)
