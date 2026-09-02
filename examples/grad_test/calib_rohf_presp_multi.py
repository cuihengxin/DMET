"""多几何联合定标 ROHF P 响应折叠：{zdens, zPFP, ooM, ooOO, vhf}。

几何 FD 已确认 P 响应有 4.4e-4 误差、9 基完备（3.5e-7）但单几何病态。
本脚本用 3 个 CH3 几何联合 lstsq + 1 个留出几何，找跨几何稳定的系数。
若 ooM/ooOO 系数稳定且留出 ~1e-6，即为正确折叠。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_presp_multi.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad

NAMES = ['zdens', 'zPFP', 'ooS', 'ooM', 'ooOO', 'oo_rhf', 'vhf']


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


def blocks(mol):
    mf = tight(mol)
    d = ssdmet.SSDMET(mf, title='bl', imp_idx=[0], bath_norb=1, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()

    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = C.shape[1]
    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    ka = np.where(var_a, g.kappa, 0.0)
    kb = np.where(var_b, g.kappa, 0.0)
    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    occ = mo_occ > 0
    n_a = (mo_occ > 0).astype(float)
    n_b = (mo_occ == 2).astype(float)
    M = n_a[:, None] * n_a[None, :] + n_b[:, None] * n_b[None, :]
    moFbar = g.moFbar
    Zma, Zmb = C @ ka @ C.T, C @ kb @ C.T
    Pa, Pb = C[:, occa] @ C[:, occa].T, C[:, occb] @ C[:, occb].T

    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))
    zvec0, dPa0, dPb0 = g.zvec_ao.copy(), g.dPa.copy(), g.dPb.copy()
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))
    vhf = sym(Pa @ (vj[0] + vj[1] - vk[0]) @ Pa
              + Pb @ (vj[0] + vj[1] - vk[1]) @ Pb)
    terms = {
        'zPFP': sym(Zma @ fa @ Pa + Zmb @ fb @ Pb),
        # (5.7a) 占-占 的候选权重：(1+n_j) 来自含 S^{-1} 的 Pulay
        'ooS': sym(C[:, occ] @ (moFbar[np.ix_(occ, occ)]
                                * (1 + mo_occ[occ][None, :]))
                   @ C[:, occ].T),
        'ooM': sym(C[:, occ] @ (moFbar[np.ix_(occ, occ)] * M[np.ix_(occ, occ)])
                   @ C[:, occ].T),
        'ooOO': sym(C[:, occ] @ (mo_occ[occ][:, None] * moFbar[np.ix_(occ, occ)])
                    @ C[:, occ].T),
        'oo_rhf': sym(C[:, occ] @ moFbar[np.ix_(occ, occ)] @ C[:, occ].T),
        'vhf': vhf,
    }

    def grad(dens_on, nm, cv):
        g.zvec_ao = dens_on * zvec0
        g.dPa = dens_on * dPa0
        g.dPb = dens_on * dPb0
        sb = sbar_lowdin.copy()
        if nm:
            sb = sb - cv * terms[nm]
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

    base = grad(0, None, 0)
    contrib = {'zdens': grad(1, None, 0) - base}
    for nm in NAMES[1:]:
        contrib[nm] = grad(0, nm, 1.0) - base

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

    A = np.stack([contrib[k].ravel() for k in NAMES], axis=1)
    return A, num.ravel(), contrib


def ch3(dx=0.0, dy=0.0, dz=0.0):
    return gto.M(atom=f'C 0 0 0; H {0.63+dx} 0.61 0.05; '
                      f'H -0.60 {0.65+dy} -0.03; H 0.02 -0.89 {0.07+dz}',
                 basis='sto-3g', spin=1, verbose=0)


def main():
    fit_mols = [ch3(), ch3(0.08, -0.05), ch3(-0.06, 0.09, 0.04)]
    holdout = ch3(0.12, 0.11, -0.05)

    A_list, r_list, contrib_list = [], [], []
    for m in fit_mols:
        A, r, contrib = blocks(m)
        A_list.append(A)
        r_list.append(r)
        contrib_list.append(contrib)
    A = np.vstack(A_list)
    r = np.concatenate(r_list)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)

    print('--- 多几何联合定标 {zdens, zPFP, ooM, ooOO, vhf} ---')
    print('  cond = %.1e' % (sv.max() / sv.min()))
    for k, v in zip(NAMES, coef):
        print(f'    {k:<6s} = {v: .5f}')
    print('  fit 残差 = %.3e' % np.abs(A @ coef - r).max())

    # hold-out: use fitted coefs on a new geometry
    A_h, r_h, _ = blocks(holdout)
    hold = np.abs(A_h @ coef - r_h).max()
    print('  HOLD-OUT = %.3e' % hold)
    print('\n若系数跨几何稳定且 HOLD-OUT ~1e-6 -> 正确折叠；')
    print('若 ooM/ooOO 仍巨大抵消 -> 基组仍不完备或病态。')


if __name__ == '__main__':
    main()
