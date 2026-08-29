"""ROHF Pulay 折叠项的决定性定标：多几何拟合 + 留出验证 + 共线诊断。

上一轮的教训：T1 和 T2 强共线，拟合出的 (-0.832, -0.159) 之和虽然 ≈ -1，
但给两项各取 -1 会放大近 2 倍（实测把误差从 3.6e-3 推高到 3.4e-2）。
所以这里改用**单一组合基函数** Tsum = T1 + T2（推导给出的就是等权组合），
并对若干候选基组分别拟合，用留出几何交叉验证挑出真正的结构。

候选项：
  zdens : dPa/dPb/zvec_ao 通道（hcore + 2e）
  ooOO  : sym(C_occ (N·moFbar_oo) C_occ^T)
  ooOA  : sym(C_occ (N·moFbar[occ,:]) C^T)
  Tsum  : sym(Σ_σ C(F_σ Z_σ^T + Z_σ^T F_σ)C^T)      <- 推导得到的等权组合
  zPFP  : sym(Σ_σ Zmat_σ F_σ P_σ)                    <- 代码当前用的形式
  vhf   : sym(Σ_σ P_σ veff_σ P_σ)

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_folding.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad

ALL = ['zdens', 'ooOO', 'ooOA', 'Tsum', 'zPFP', 'vhf']


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


def setup(mol, imp_idx=(0,), bath_norb=1):
    """Return (grad_fn(dict of coefs), exact_FD_grad)."""
    imp_idx = list(imp_idx)
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
    occ = mo_occ > 0
    n = mo_occ[occ]

    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    ka = np.where(var_a, g.kappa, 0.0)
    kb = np.where(var_b, g.kappa, 0.0)
    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    Fa, Fb = C.T @ fa @ C, C.T @ fb @ C

    Zma, Zmb = C @ ka @ C.T, C @ kb @ C.T
    mo_a, mo_b = C[:, occa], C[:, occb]
    Pa, Pb = mo_a @ mo_a.T, mo_b @ mo_b.T
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))

    terms = {
        'ooOO': sym(C[:, occ] @ (n[:, None] * moFbar[np.ix_(occ, occ)])
                    @ C[:, occ].T),
        'ooOA': sym(C[:, occ] @ (n[:, None] * moFbar[occ, :]) @ C.T),
        'Tsum': sym(C @ (Fa @ ka.T + ka.T @ Fa) @ C.T
                    + C @ (Fb @ kb.T + kb.T @ Fb) @ C.T),
        'zPFP': sym(Zma @ fa @ Pa + Zmb @ fb @ Pb),
        'vhf': sym(Pa @ (vj[0] + vj[1] - vk[0]) @ Pa
                   + Pb @ (vj[0] + vj[1] - vk[1]) @ Pb),
    }
    dPa0, dPb0 = g.dPa.copy(), g.dPb.copy()

    def grad(coefs):
        g.dPa = coefs.get('zdens', 0.0) * dPa0
        g.dPb = coefs.get('zdens', 0.0) * dPb0
        g.zvec_ao = g.dPa + g.dPb
        sb = sbar_lowdin.copy()
        for k, v in coefs.items():
            if k != 'zdens':
                sb = sb + v * terms[k]
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

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
    return grad, exact


def design(grad, exact, names):
    zero = {k: 0.0 for k in names}
    base = grad(zero)
    cols = []
    for k in names:
        c = dict(zero)
        c[k] = 1.0
        cols.append((grad(c) - base).ravel())
    return np.stack(cols, axis=1), (exact - base).ravel()


def run(names, fit_mols, holdout_mol):
    A_list, r_list = [], []
    for m in fit_mols:
        grad, exact = setup(m)
        A, r = design(grad, exact, names)
        A_list.append(A)
        r_list.append(r)
    A = np.vstack(A_list)
    r = np.concatenate(r_list)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)

    gh, eh = setup(holdout_mol)
    hold = np.abs(gh(dict(zip(names, coef))) - eh).max()

    print(f'--- basis {names} ---')
    print('  cond = %.3e   sv ratio min/max = %.2e'
          % (sv.max() / sv.min(), sv.min() / sv.max()))
    for k, v in zip(names, coef):
        print(f'    {k:<6s} = {v: .5f}')
    print('  fit residual = %.3e' % np.abs(A @ coef - r).max())
    print('  HOLD-OUT     = %.3e' % hold)
    return coef, hold


def ch3(dx=0.0, dy=0.0, dz=0.0):
    return gto.M(atom=f'C 0 0 0; H {0.63+dx} 0.61 0.05; '
                      f'H -0.60 {0.65+dy} -0.03; H 0.02 -0.89 {0.07+dz}',
                 basis='sto-3g', spin=1, verbose=0)


if __name__ == '__main__':
    fit_mols = [ch3(), ch3(0.08, -0.05), ch3(-0.06, 0.09, 0.04)]
    holdout = ch3(0.12, 0.11, -0.05)

    print('目标：HOLD-OUT ~3e-7（FD 截断极限）。系数应是干净的小分数。\n')
    for names in (['zdens', 'ooOO', 'Tsum', 'vhf'],
                  ['zdens', 'ooOO', 'zPFP', 'vhf'],
                  ['zdens', 'ooOA', 'Tsum', 'vhf'],
                  ['zdens', 'ooOO', 'ooOA', 'Tsum', 'vhf'],
                  ['zdens', 'ooOO', 'ooOA', 'Tsum', 'zPFP', 'vhf']):
        try:
            run(names, fit_mols, holdout)
        except Exception as exc:          # keep going if one basis is singular
            print(f'--- basis {names} ---\n  FAILED: {exc}')
        print()
    print('挑 HOLD-OUT 最小且系数最干净的那组基 -> 就是正确结构。')
