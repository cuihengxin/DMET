"""定标 ROHF Z-vector 折叠：多基函数最小二乘，判定缺的是哪一项。

把解析梯度写成
    g = base(无 P 响应) + Σ_k c_k · b_k
其中候选基函数（都是 sbar 里的候选项 / 密度通道）：

  b0 zdens     : zvec_ao / dPa / dPb 通道（hcore + 2e）
  b1 oo_occocc : sym(C_occ (N·moFbar_oo) C_occ^T)          —— 当前实现用的
  b2 oo_occall : sym(C_occ (N·moFbar[occ,:]) C^T)          —— A2 的完整形式
  b3 T1        : sym(Σ_σ C (F_σ^MO Z_σ^T) C^T)             —— −½S^x 通道
  b4 T2        : sym(Σ_σ C (Z_σ^T F_σ^MO) C^T)             —— −½S^x 通道
  b5 vhf       : sym(Σ_σ P_σ veff_σ P_σ)

对精确（有限差分）梯度做最小二乘。若拟合出干净系数且残差 ~0，
系数就直接给出正确公式；再用第二个体系交叉验证，排除过拟合。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_basis.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad

NAMES = ['zdens', 'oo_occocc', 'oo_occall', 'T1', 'T2', 'vhf']


def tight(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, 'ROHF not converged'
    return mf


def sym(a):
    return a + a.T


def build_basis(mol, imp_idx, bath_norb):
    """Return (grad_fn, exact_FD_grad).  grad_fn(coefs) -> analytic gradient."""
    mf = tight(mol)
    d = ssdmet.SSDMET(mf, title='cal', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()

    # Lowdin-only sbar (no P response at all)
    sbar_base = (matfun_grad(g.w_s, g.u_s, g.B_X,
                             lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                 + matfun_grad(g.w_s, g.u_s, g.B_Y,
                               lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))

    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    occ = mo_occ > 0
    n = mo_occ[occ]
    moFbar = g.moFbar

    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    kappa = g.kappa
    ka = np.where(var_a, kappa, 0.0)
    kb = np.where(var_b, kappa, 0.0)

    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    Fa_mo = C.T @ fa @ C
    Fb_mo = C.T @ fb @ C

    # ---- candidate sbar terms ----
    b_oo_occocc = sym(C[:, occ] @ (n[:, None] * moFbar[np.ix_(occ, occ)])
                      @ C[:, occ].T)
    b_oo_occall = sym(C[:, occ] @ (n[:, None] * moFbar[occ, :]) @ C.T)
    b_T1 = sym(C @ (Fa_mo @ ka.T) @ C.T + C @ (Fb_mo @ kb.T) @ C.T)
    b_T2 = sym(C @ (ka.T @ Fa_mo) @ C.T + C @ (kb.T @ Fb_mo) @ C.T)

    mo_a, mo_b = C[:, occa], C[:, occb]
    Pa, Pb = mo_a @ mo_a.T, mo_b @ mo_b.T
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))
    veff_a = vj[0] + vj[1] - vk[0]
    veff_b = vj[0] + vj[1] - vk[1]
    b_vhf = sym(Pa @ veff_a @ Pa + Pb @ veff_b @ Pb)

    dPa0, dPb0 = g.dPa.copy(), g.dPb.copy()

    def grad(c):
        c0, c1, c2, c3, c4, c5 = c
        g.dPa = c0 * dPa0
        g.dPb = c0 * dPb0
        g.zvec_ao = g.dPa + g.dPb
        g.sbar = (sbar_base
                  + c1 * b_oo_occocc + c2 * b_oo_occall
                  + c3 * b_T1 + c4 * b_T2 + c5 * b_vhf)
        return g.contract() + g.mfgrad.grad_nuc()

    # ---- exact gradient by central differences ----
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


def fit(mol, imp_idx, bath_norb, label):
    grad, exact = build_basis(mol, imp_idx, bath_norb)
    zero = np.zeros(6)
    base = grad(zero)
    B = []
    for k in range(6):
        c = zero.copy()
        c[k] = 1.0
        B.append((grad(c) - base).ravel())
    A = np.stack(B, axis=1)
    resid = (exact - base).ravel()
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    fit_err = np.abs(A @ coef - resid).max()

    print(f'--- {label} ---')
    print('  cond(A) = %.2e' % np.linalg.cond(A))
    for name, cv in zip(NAMES, coef):
        print(f'    {name:<10s} = {cv: .5f}')
    print(f'  residual after fit = {fit_err:.4e}')
    print(f'  grad at best fit   = {np.abs(grad(coef)-exact).max():.4e}')
    return coef, grad, exact


if __name__ == '__main__':
    ch3 = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                     'H 0.02 -0.89 0.07',
                basis='sto-3g', spin=1, verbose=0)
    oh = gto.M(atom='O 0 0 0; H 0 0.13 0.95', basis='sto-3g',
               spin=1, verbose=0)

    coef1, _, _ = fit(ch3, [0], 1, 'CH3 (C imp, bath=1)')
    coef2, grad2, exact2 = fit(oh, [0], 1, 'OH (O imp, bath=1)')

    print('\n交叉验证：把 CH3 拟合出的系数用到 OH 上')
    print('  max|ana-FD| with CH3 coefs = %.4e'
          % np.abs(grad2(coef1) - exact2).max())
    print('\n若两组系数一致且交叉验证也小 -> 就是正确公式；')
    print('若系数不一致 -> 基函数仍不完备，还缺项。')
