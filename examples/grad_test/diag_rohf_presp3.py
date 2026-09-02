"""P 响应完备基拟合：找截断 bath ROHF 缺失的那一项。

候选基（全部对应进 sbar 或密度通道的项）：
  1. zdens  : zvec 通道（dP 进 hcore/2e）
  2. zPFP   : -sym(Zmat_s F_s P_s)         （定标已确认的主项）
  3. vhf    : -sym(P_s v_s(W) P_s)
  4. mfp    : -sym(G_P P)                   ← 推导 (5.7a)，怀疑缺失项
  5. ooOO   : -sym(C_occ N moFbar_oo C_occ^T)
  6. Tsum   : -sym(C(F k^T + k^T F) C^T)    ← 推导 (5.7b)

对数值 P 响应 FD 做最小二乘。系数全 ~1 且残差归零的那组就是正确形式。

运行：cd 8dmet4reac/DMET && python examples/test_example/diag_rohf_presp3.py
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
    g.orb_grad(); g.lo_grad(); g.solve_z()

    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = C.shape[1]
    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    ka = np.where(var_a, g.kappa, 0.0)
    kb = np.where(var_b, g.kappa, 0.0)
    kappa = g.kappa

    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    Fa, Fb = C.T @ fa @ C, C.T @ fb @ C

    occ = mo_occ > 0
    n = mo_occ[occ]
    moFbar = g.moFbar
    G_P = g.G_P.copy()
    P = mf.make_rdm1()
    P = P[0] + P[1] if np.ndim(P) == 3 else P

    # (5.7a) 占-占 部分的正确权重：M_ij = n_a[i]n_a[j] + n_b[i]n_b[j]
    # （张量积，不是对角 N = diag(n)）
    n_a = (mo_occ > 0).astype(float)
    n_b = (mo_occ == 2).astype(float)
    M = n_a[:, None] * n_a[None, :] + n_b[:, None] * n_b[None, :]

    # ---- sbar candidate terms (matrices; contract with S^x) ----
    Tsum = sym(C @ (Fa @ ka.T + ka.T @ Fa) @ C.T
               + C @ (Fb @ kb.T + kb.T @ Fb) @ C.T)
    Zma, Zmb = C @ ka @ C.T, C @ kb @ C.T
    mo_a, mo_b = C[:, occa], C[:, occb]
    Pa, Pb = mo_a @ mo_a.T, mo_b @ mo_b.T
    zPFP = sym(Zma @ fa @ Pa + Zmb @ fb @ Pb)
    mfp = sym(G_P @ P)                    # (5.7a), no 1/2 (coefficient to fit)
    ooOO = sym(C[:, occ] @ (n[:, None] * moFbar[np.ix_(occ, occ)])
               @ C[:, occ].T)
    ooM = sym(C[:, occ] @ (moFbar[np.ix_(occ, occ)] * M[np.ix_(occ, occ)])
              @ C[:, occ].T)

    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))
    zvec0, dPa0, dPb0 = g.zvec_ao.copy(), g.dPa.copy(), g.dPb.copy()
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))
    vhf = sym(Pa @ (vj[0] + vj[1] - vk[0]) @ Pa
              + Pb @ (vj[0] + vj[1] - vk[1]) @ Pb)
    # closed-shell v = J - K/2 (RHF-style), projected
    vj_c, vk_c = mf.get_jk(mol, g.zvec_ao)
    vhf_closed = sym((Pa + Pb) @ (vj_c[0] - 0.5 * vk_c[0]) @ (Pa + Pb))
    # RHF-like oo with no weight
    oo_rhf = sym(C[:, occ] @ moFbar[np.ix_(occ, occ)] @ C[:, occ].T)

    sbars = {'zPFP': zPFP, 'mfp': mfp, 'ooOO': ooOO, 'ooM': ooM,
             'Tsum': Tsum, 'vhf': vhf, 'vhf_c': vhf_closed,
             'oo_rhf': oo_rhf}

    def grad(dens_on, sbar_names, sbar_coefs):
        g.zvec_ao = dens_on * zvec0
        g.dPa = dens_on * dPa0
        g.dPb = dens_on * dPb0
        sb = sbar_lowdin.copy()
        for nm, cv in zip(sbar_names, sbar_coefs):
            sb = sb - cv * sbars[nm]
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

    base = grad(0, [], [])
    names = ['zdens', 'zPFP', 'mfp', 'ooOO', 'ooM', 'Tsum', 'vhf',
             'vhf_c', 'oo_rhf']
    contrib = {}
    contrib['zdens'] = grad(1, [], []) - base
    for nm in names[1:]:
        contrib[nm] = grad(0, [nm], [1.0]) - base

    # numerical P-response
    coords = mol.atom_coords()
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

    # 全自由穷举：看最小残差能到多少、系数是否干净
    Aall = np.stack([contrib[k].ravel() for k in names], axis=1)
    call, *_ = np.linalg.lstsq(Aall, num.ravel(), rcond=None)
    print('\n--- 全 9 基自由拟合 ---')
    print('  系数:', {k: round(float(v), 3) for k, v in zip(names, call)})
    print('  残差 max = %.3e' % np.abs(num.ravel() - Aall @ call).max())
    print('  cond = %.1e' % np.linalg.cond(Aall))

    for sub in ([['zdens', 'zPFP', 'vhf']],
                [['zdens', 'zPFP', 'ooM', 'vhf']],
                [['zdens', 'zPFP', 'ooM', 'ooOO', 'vhf']],
                [['zdens', 'zPFP', 'mfp', 'ooM', 'vhf']],
                [['zdens', 'zPFP', 'mfp', 'ooM', 'ooOO', 'Tsum', 'vhf']]):
        A = np.stack([contrib[k].ravel() for k in sub[0]], axis=1)
        coef, *_ = np.linalg.lstsq(A, num.ravel(), rcond=None)
        resid = num.ravel() - A @ coef
        print(f'--- 基组 {sub[0]} ---')
        print('  系数:', {k: round(float(v), 4) for k, v in zip(sub[0], coef)})
        print('  残差 max = %.3e' % np.abs(resid).max())

    # ---- 固定 zdens=zPFP=1, vhf=0.5，单独拟合 oo 类项 ----
    print('\n=== 固定 zdens=zPFP=1, vhf=0.5，拟合 oo 类项 ===')
    resid5 = num.ravel() - (contrib['zdens'].ravel()
                            + contrib['zPFP'].ravel()
                            + 0.5 * contrib['vhf'].ravel())
    print('  前固定残差 max = %.3e' % np.abs(resid5).max())
    for k in ('ooM', 'ooOO', 'mfp', 'Tsum'):
        b = contrib[k].ravel()
        c = np.dot(b, resid5) / np.dot(b, b)
        print(f'  {k:<5s}: 系数 = {c: .4f}   加后残差 = '
              f'{np.abs(resid5 - c*b).max():.3e}')

    # ooM 与 ooOO 同时拟合（两个都自由）
    A2 = np.stack([contrib['ooM'].ravel(), contrib['ooOO'].ravel()], axis=1)
    c2, *_ = np.linalg.lstsq(A2, resid5, rcond=None)
    print('  [ooM, ooOO] 联合: 系数 =', np.round(c2, 4),
          ' 残差 = %.3e' % np.abs(resid5 - A2 @ c2).max())

    # ---- 穷举扫描：固定 zdens=zPFP=1, vhf=0.5，扫 oo 类项系数 ----
    print('\n=== 扫描 oo 类项系数（zdens=zPFP=1, vhf=0.5 固定）===')
    base_vec = (contrib['zdens'].ravel() + contrib['zPFP'].ravel()
                + 0.5 * contrib['vhf'].ravel())
    for k in ('ooM', 'ooOO', 'mfp', 'oo_rhf'):
        b = contrib[k].ravel()
        best_c, best_r = 0.0, np.abs(num.ravel() - base_vec).max()
        for c in np.arange(-2.0, 2.05, 0.05):
            r = np.abs(num.ravel() - base_vec - c * b).max()
            if r < best_r:
                best_r, best_c = r, c
        print(f'  {k:<5s}: 最优系数 = {best_c: .2f}   最优残差 = {best_r:.3e}')

    # 组合 ooM 和 mfp（占-占 + 占-虚 分开）
    resid_v = num.ravel() - base_vec
    bM, bf = contrib['ooM'].ravel(), contrib['mfp'].ravel()
    A3 = np.stack([bM, bf], axis=1)
    c3, *_ = np.linalg.lstsq(A3, resid_v, rcond=None)
    print('  [ooM, mfp] 联合: 系数 =', np.round(c3, 4),
          ' 残差 = %.3e' % np.abs(resid_v - A3 @ c3).max())


if __name__ == '__main__':
    main()
