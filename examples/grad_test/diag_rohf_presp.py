"""定位截断 bath ROHF 梯度的系统误差（~2.3e-4）：P 响应分解。

解析全梯度 = 冻结-P 梯度 + P 响应。
  g_frozen : 关闭 Z-vector 及所有 P 响应（Z=0, oo=0, zeta=0, vhf=0, dP=0）
  g_full   : 当前代码
  P 响应解析 = g_full - g_frozen
  P 响应数值 = FD[Tr[G_P P(R)]]   （P(R) 是 ROHF 密度，G_P 固定）

若 (解析 P 响应) - (数值 P 响应) ~ 1e-6（FD 精度）-> P 响应折叠对，误差在别处；
若差 ~2.3e-4 -> P 响应有 bug，继续分解 zdens / zPFP / vhf 三块。

运行：cd 8dmet4reac/DMET && python examples/test_example/diag_rohf_presp.py
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


def sym(a):
    return a + a.T


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
    g.orb_grad(); g.lo_grad(); g.solve_z(); g.make_sbar()
    g_full = g.contract() + g.mfgrad.grad_nuc()
    G_P = g.G_P.copy()

    # frozen-P analytic gradient (all P-response channels off)
    g.Z[:] = 0.0
    g.dPa[:] = 0.0
    g.dPb[:] = 0.0
    g.zvec_ao[:] = 0.0
    g.zeta[:] = 0.0
    g.vhf_s1occ[:] = 0.0
    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))
    g.sbar = sbar_lowdin            # no oo, no zeta, no vhf
    g_frozen = g.contract() + g.mfgrad.grad_nuc()
    ana_presp = g_full - g_frozen

    # numerical P-response: FD of Tr[G_P P(R)]
    coords = mol.atom_coords()
    num_presp = np.zeros((mol.natm, 3))
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
            num_presp[ia, x] = (q1 - q2) / (2 * h)

    diff = ana_presp - num_presp
    print('=== 截断 bath CH3 (bath=1), conv_tol_grad=1e-12 ===')
    print('解析 P 响应:\n', np.round(ana_presp, 6))
    print('数值 P 响应:\n', np.round(num_presp, 6))
    print('差 (解析-数值):\n', np.round(diff, 6))
    print('max |解析-数值| =', np.abs(diff).max())
    print('max |解析 P 响应| =', np.abs(ana_presp).max())

    # 全梯度误差 vs FD（确认当前总误差量级）
    def energy(m):
        mf2 = tight(m)
        d2 = ssdmet.SSDMET(mf2, title='e', imp_idx=[0], bath_norb=1, verbose=0)
        d2.build(save_chk=False)
        return d2.es_mf.e_tot + d2.fo_ene

    num_full = np.zeros((mol.natm, 3))
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
            num_full[ia, x] = (e1 - e2) / (2 * h)
    print('\n全梯度 vs FD: max |解析-数值| =', np.abs(g_full - num_full).max())
    print('冻结-P vs FD: max |解析-数值| =', np.abs(g_frozen - num_full).max())


if __name__ == '__main__':
    main()
