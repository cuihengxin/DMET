"""验证推导文档 DMET_ROHF梯度推导.md §5.5 的开放问题。

判断推导组合 (5.7b: Tsum) 与定标组合 (5.8: zPFP) 是否等价：
  Δ = -½ Σ_σ Tr[S^x(F_σ κ_σ^T + κ_σ^T F_σ)]        (推导, 系数 -1/2)
    - [ -Σ_σ Tr[S^x κ_σ F_σ^MO diag(n_σ)] ]          (定标, 系数 -1)
对随机 S^x 评估。若 Δ ≈ 0 -> 两者等价（只是拆法不同），
若 Δ 非零 -> 推导或定标有一个错了。

同时测 RHF：RHF 下两者应严格等价（因为 F C_occ = eps C_occ）。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_zeta.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


def tight_rohf(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def tight_rhf(m):
    mf = scf.RHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def setup(mf, mol, imp_idx, bath_norb):
    d = ssdmet.SSDMET(mf, title='z', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()
    return g


def zeta_terms(g, mf, mol):
    """返回 (Tsum, zPFP) 两个 (nmo,nmo) MO 基候选矩阵（进 sbar 用）。"""
    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = C.shape[1]
    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()

    if getattr(g, 'openshell', False):
        ka = np.where(var_a, g.kappa, 0.0)
        kb = np.where(var_b, g.kappa, 0.0)
    else:  # RHF: build kappa from the (nvir,nocc) Z-vector
        occidx = mo_occ > 0
        viridx = ~occidx
        kap = np.zeros((nmo, nmo))
        kap[np.ix_(viridx, occidx)] = g.Z
        kap[np.ix_(occidx, viridx)] = -g.Z.T
        ka = np.where(var_a, kap, 0.0)
        kb = np.where(var_b, kap, 0.0)

    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    Fa, Fb = C.T @ fa @ C, C.T @ fb @ C

    n_a = (mo_occ > 0).astype(float)      # 1 if in occa
    n_b = (mo_occ == 2).astype(float)     # 1 if in occb

    # 推导 (5.7b): -1/2 (F kappa^T + kappa^T F)  MO 基
    Tsum = -(Fa @ ka.T + ka.T @ Fa + Fb @ kb.T + kb.T @ Fb) / 2.0
    # 定标 (5.8): -kappa F diag(n)  MO 基
    zPFP = -(ka @ (Fa * n_a[None, :]) + kb @ (Fb * n_b[None, :]))
    return Tsum, zPFP


def check(mf, mol, imp_idx, bath_norb, label):
    g = setup(mf, mol, imp_idx, bath_norb)
    Tsum, zPFP = zeta_terms(g, mf, mol)

    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(5):
        S = rng.standard_normal((mol.nao, mol.nao))
        S = S + S.T
        Smo = mf.mo_coeff.T @ S @ mf.mo_coeff
        d_Tsum = np.einsum('ij,ij->', Smo, Tsum)
        d_zPFP = np.einsum('ij,ij->', Smo, zPFP)
        diffs.append(abs(d_Tsum - d_zPFP) / max(abs(d_zPFP), 1e-12))
    print(f'--- {label} ---')
    print(f'  max|Tr[S (Tsum - zPFP)] / Tr[S zPFP]| over 5 random S = '
          f'{max(diffs):.3e}')
    # magnitudes
    print(f'  ||Tsum||_max = {np.abs(Tsum).max():.3e}   '
          f'||zPFP||_max = {np.abs(zPFP).max():.3e}')
    return max(diffs)


if __name__ == '__main__':
    mol = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                     'H 0.02 -0.89 0.07',
                basis='sto-3g', spin=1, verbose=0)

    mf = tight_rohf(mol)
    check(mf, mol, [0], 1, 'ROHF CH3 (C imp, bath=1)')

    # RHF 对照（闭壳层 CH3+, bath=1）：两者应严格等价
    mf_r = tight_rhf(mol.copy())
    mf_r.mol.spin = 0
    mol_r = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                       'H 0.02 -0.89 0.07',
                  basis='sto-3g', spin=0, charge=1, verbose=0)
    mf_r = tight_rhf(mol_r)
    check(mf_r, mol_r, [0], 1, 'RHF CH3+ (closed shell, bath=1)')
    print('\n若 RHF 差值 ~1e-12（等价）而 ROHF 差值非零 -> 推导 (5.7b) 在 ROHF 不成立；')
    print('若两者都 ~1e-12 -> Tsum 与 zPFP 等价，定标残差来自别处。')
