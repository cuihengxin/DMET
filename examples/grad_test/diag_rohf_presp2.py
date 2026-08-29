"""分解截断 bath ROHF 的 P 响应：zdens / zPFP / vhf 三块 vs 数值。

数值 P 响应 = FD[Tr[G_P P(R)]]（纯 G_P 通道）。
解析 P 响应 = zvec 通道(zdens) + zPFP(zeta) + vhf + oo。
对三块做 lstsq 拟合数值，系数应全 ~1；偏离的块就是 bug。

运行：cd 8dmet4reac/DMET && python examples/test_example/diag_rohf_presp2.py
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
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def main():
    mol = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                     'H 0.02 -0.89 0.07',
                basis='sto-3g', spin=1, verbose=0)
    mf = tight(mol)
    P_ref = mf.make_rdm1().copy()

    d = ssdmet.SSDMET(mf, title='diag', imp_idx=[0], bath_norb=1, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()

    # save the full P-response objects
    zvec0 = g.zvec_ao.copy()
    dPa0 = g.dPa.copy()
    dPb0 = g.dPb.copy()
    zeta0 = g.zeta.copy()
    vhf0 = g.vhf_s1occ.copy()

    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))

    def grad(zvec, zeta, vhf):
        """zvec: 0/1 开关密度通道; zeta/vhf: 0/1 开关 sbar 项."""
        g.zvec_ao = zvec * zvec0
        g.dPa = zvec * dPa0
        g.dPb = zvec * dPb0
        sb = sbar_lowdin.copy()
        if zeta:
            sb = sb - (zeta0 + zeta0.T)
        if vhf:
            sb = sb - (vhf0 + vhf0.T)
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

    base = grad(0, 0, 0)
    g_zdens = grad(1, 0, 0) - base
    g_zPFP = grad(0, 1, 0) - base
    g_vhf = grad(0, 0, 1) - base

    # numerical P-response
    coords = mol.atom_coords()
    G_P = g.G_P.copy()
    num = np.zeros((mol.natm, 3))
    h = 1e-3
    for ia in range(mol.natm):
        for x in range(3):
            q1 = q2 = 0.0
            for s in (1, -1):
                c = coords.copy()
                c[ia, x] += s * h
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                mf2 = tight(m)
                q = np.einsum('ij,ij->', G_P,
                              mf2.make_rdm1()[0] + mf2.make_rdm1()[1])
                if s > 0:
                    q1 = q
                else:
                    q2 = q
            num[ia, x] = (q1 - q2) / (2 * h)

    A = np.stack([g_zdens.ravel(), g_zPFP.ravel(), g_vhf.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, num.ravel(), rcond=None)
    resid = num.ravel() - A @ coef

    print('=== P 响应三块分解 ===')
    print('  zdens(zvec) :', g_zdens[1, 0], '(示例分量)')
    print('  zPFP(zeta)  :', g_zPFP[1, 0])
    print('  vhf         :', g_vhf[1, 0])
    print('  拟合系数 (应为 1):')
    print('    c_zdens =', coef[0])
    print('    c_zPFP  =', coef[1])
    print('    c_vhf   =', coef[2])
    print('  残差 max =', np.abs(resid).max())
    print('  数值 P 响应 max =', np.abs(num).max())
    print('\n系数偏离 1 的块就是 bug。残差大 -> 三块之外还缺 oo 或别的项。')


if __name__ == '__main__':
    main()
