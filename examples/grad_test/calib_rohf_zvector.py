"""定标 ROHF Z-vector 折叠项的系数（诊断脚本，非正式测试）。

思路：把解析梯度写成
    g(c_oo, c_zvec, c_zeta, c_vhf) = base + c_zeta*d_zeta + c_vhf*d_vhf
用最小二乘拟合出能复现精确（有限差分）梯度的系数组合。
若拟合出干净的 (1, 0.5) 之类，说明只是某项差个因子；
若残差不为零，说明还缺项（结构问题，不是因子问题）。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_zvector.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad


def tight(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def run(mol, imp_idx, bath_norb, label):
    mf = tight(mol)
    d = ssdmet.SSDMET(mf, title='cal', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z(); g.make_sbar()

    sbar_base = (matfun_grad(g.w_s, g.u_s, g.B_X,
                             lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                 + matfun_grad(g.w_s, g.u_s, g.B_Y,
                               lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))
    zeta0 = g.zeta.copy()
    vhf0 = g.vhf_s1occ.copy()

    mo = mf.mo_coeff
    occ = mf.mo_occ > 0
    n = mf.mo_occ[occ]
    oo = -0.5 * (mo[:, occ] @ (n[:, None] * g.moFbar[np.ix_(occ, occ)])
                 @ mo[:, occ].T)
    oo = oo + oo.T

    def grad(c_zeta, c_vhf):
        g.sbar = (sbar_base + oo
                  - c_zeta * (zeta0 + zeta0.T)
                  - c_vhf * (vhf0 + vhf0.T))
        return g.contract() + g.mfgrad.grad_nuc()

    # exact gradient by central differences
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
                c = coords.copy()
                c[ia, x] += s * h
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                e = energy(m)
                if s > 0:
                    e1 = e
                else:
                    e2 = e
            exact[ia, x] = (e1 - e2) / (2 * h)

    base = grad(0.0, 0.0)
    d_zeta = grad(1.0, 0.0) - base
    d_vhf = grad(0.0, 1.0) - base
    resid = exact - base

    A = np.stack([d_zeta.ravel(), d_vhf.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, resid.ravel(), rcond=None)
    fit_resid = np.abs(A @ coef - resid.ravel()).max()

    print(f'--- {label} ---')
    print(f'  current (1,1)      : max|ana-FD| = {np.abs(grad(1,1)-exact).max():.4e}')
    print(f'  best-fit coef      : zeta={coef[0]:.4f}  vhf={coef[1]:.4f}')
    print(f'  residual after fit : {fit_resid:.4e}')
    print(f'  grad at best fit   : {np.abs(grad(*coef)-exact).max():.4e}')
    return coef, fit_resid


if __name__ == '__main__':
    ch3 = gto.M(atom='C 0 0 0; H 0.63 0.63 0; H -0.63 0.63 0; H 0 -0.89 0',
                basis='sto-3g', spin=1, verbose=0)
    no = gto.M(atom='N 0 0 0; O 0 0 1.15', basis='sto-3g', spin=1, verbose=0)
    run(ch3, [0], 1, 'CH3 (C imp, bath=1)')
    run(no, [0], 1, 'NO (N imp, bath=1)')
    print('\n若两个体系拟合出相同的干净系数且残差~0 -> 只是因子问题；')
    print('若残差远大于 0 -> 还缺项（结构问题）。')
