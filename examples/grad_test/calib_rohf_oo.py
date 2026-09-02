"""定标 ROHF Pulay 的占据加权项：把 oo 项按 docc/socc × occ/vir 拆块。

已定标确认的三项固定不动（zdens=1, T1+T2=-1, vhf=-0.5），
只让 oo 项的 5 个分块系数自由：

  dd : docc-docc      ds : docc-socc      ss : socc-socc
  dv : docc-vir       sv : socc-vir

为避免过拟合，用 **多个几何同时拟合**（把方程堆起来），
再在一个留出的几何上交叉验证。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_oo.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad

BLOCKS = ['dd', 'ds', 'ss', 'dv', 'sv']


def tight(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def sym(a):
    return a + a.T


def setup(mol, imp_idx, bath_norb):
    mf = tight(mol)
    d = ssdmet.SSDMET(mf, title='cal', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()

    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))

    C, mo_occ = mf.mo_coeff, mf.mo_occ
    moFbar = g.moFbar
    docc = mo_occ == 2
    socc = (mo_occ > 0) & (mo_occ < 2)
    virt = mo_occ == 0

    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    ka = np.where(var_a, g.kappa, 0.0)
    kb = np.where(var_b, g.kappa, 0.0)
    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    Fa_mo, Fb_mo = C.T @ fa @ C, C.T @ fb @ C

    # --- the three already-calibrated terms, held fixed ---
    T1 = sym(C @ (Fa_mo @ ka.T) @ C.T + C @ (Fb_mo @ kb.T) @ C.T)
    T2 = sym(C @ (ka.T @ Fa_mo) @ C.T + C @ (kb.T @ Fb_mo) @ C.T)
    mo_a, mo_b = C[:, occa], C[:, occb]
    Pa, Pb = mo_a @ mo_a.T, mo_b @ mo_b.T
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))
    vhf = sym(Pa @ (vj[0] + vj[1] - vk[0]) @ Pa
              + Pb @ (vj[0] + vj[1] - vk[1]) @ Pb)
    fixed = -1.0 * (T1 + T2) - 0.5 * vhf

    # --- oo blocks (free) ---
    def block(rows, cols):
        M = np.zeros_like(moFbar)
        M[np.ix_(rows, cols)] = moFbar[np.ix_(rows, cols)]
        return sym(C @ M @ C.T)

    basis = [block(docc, docc), block(docc, socc), block(socc, socc),
             block(docc, virt), block(socc, virt)]

    def grad(coefs):
        sb = sbar_lowdin + fixed
        for c, b in zip(coefs, basis):
            sb = sb + c * b
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

    # exact FD gradient
    def energy(m):
        mf2 = tight(m)
        d2 = ssdmet.SSDMET(mf2, title='e', imp_idx=imp_idx,
                           bath_norb=bath_norb, verbose=0)
        d2.build(save_chk=False)
        return d2.es_mf.e_tot + d2.fo_ene

    coords = mol.atom_coords()
    exact = np.zeros((mol.natm, 3))
    h = 1e-3
    for ia in range(mol.natm):
        for x in range(3):
            e1 = e2 = 0.0
            for s in (1, -1):
                cc = coords.copy()
                cc[ia, x] += s * h
                m = mol.copy()
                m.set_geom_(cc, unit='Bohr')
                m.build(False, False)
                e = energy(m)
                if s > 0:
                    e1 = e
                else:
                    e2 = e
            exact[ia, x] = (e1 - e2) / (2 * h)
    return grad, exact, len(basis)


def rows_for(mol, imp_idx, bath_norb):
    grad, exact, nb = setup(mol, imp_idx, bath_norb)
    zero = np.zeros(nb)
    base = grad(zero)
    cols = []
    for k in range(nb):
        c = zero.copy()
        c[k] = 1.0
        cols.append((grad(c) - base).ravel())
    return np.stack(cols, axis=1), (exact - base).ravel(), grad, exact


def ch3(dx, dy):
    return gto.M(atom=f'C 0 0 0; H {0.63+dx} {0.61} {0.05}; '
                      f'H {-0.60} {0.65+dy} {-0.03}; H 0.02 -0.89 0.07',
                 basis='sto-3g', spin=1, verbose=0)


if __name__ == '__main__':
    fits = [ch3(0.0, 0.0), ch3(0.08, -0.05), ch3(-0.06, 0.09)]
    holdout = ch3(0.12, 0.11)

    A_all, r_all = [], []
    for i, m in enumerate(fits):
        A, r, _, _ = rows_for(m, [0], 1)
        A_all.append(A)
        r_all.append(r)
        print(f'  geometry {i}: rows={A.shape[0]}')
    A = np.vstack(A_all)
    r = np.concatenate(r_all)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    print('\n--- multi-geometry fit (CH3 x3) ---')
    print('  cond(A) = %.3e' % np.linalg.cond(A))
    for name, cv in zip(BLOCKS, coef):
        print(f'    {name:<3s} = {cv: .5f}')
    print('  residual after fit = %.4e' % np.abs(A @ coef - r).max())

    _, _, grad_h, exact_h = rows_for(holdout, [0], 1)
    print('\n--- hold-out geometry ---')
    print('  max|ana-FD| with fitted coefs = %.4e'
          % np.abs(grad_h(coef) - exact_h).max())
    print('\n若系数干净（如 -1,-0.5 之类）且 hold-out ~1e-7 -> 结构完备；')
    print('若 hold-out 仍 ~1e-5 -> oo 分块也不是缺失项，需另找。')
