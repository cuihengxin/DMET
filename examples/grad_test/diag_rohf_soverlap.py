"""S 空间 FD 测试：一锤定音判定 P 响应的重叠折叠（Tsum vs zPFP）。

对随机对称重叠扰动 dS，数值测
    d/deps Tr[G_P P(S0 + eps dS)]   （P(S) = ROHF 在重叠 S 下的解）
解析对比：
    A = -Tr[dS G_P P] + Tsum   （推导 (5.7a)+(5.7b)，Tsum = -1/2 Tr[dS (F k^T + k^T F)]）
    B = -Tr[dS G_P P] + zPFP   （定标 (5.8)，zPFP = -Tr[dS k F diag(n)]）
哪个匹配数值，哪个就是正确的重叠折叠。

运行：cd 8dmet4reac/DMET && python examples/test_example/diag_rohf_soverlap.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


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
    P_tot = P_ref[0] + P_ref[1] if np.ndim(P_ref) == 3 else P_ref

    d = ssdmet.SSDMET(mf, title='sv', imp_idx=[0], bath_norb=1, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()
    G_P = g.G_P.copy()

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
    Fa, Fb = C.T @ fa @ C, C.T @ fb @ C
    n_a = (mo_occ > 0).astype(float)
    n_b = (mo_occ == 2).astype(float)

    S0 = mf.get_ovlp()
    nao = S0.shape[0]

    def P_of_S(S):
        """ROHF density at overlap S (re-solve with monkeypatched overlap)."""
        mf2 = tight(mol)               # fresh solve at S0 geometry...
        # ... but force the overlap to S
        mf2.get_ovlp = lambda *a, **k: S
        mf2.kernel(dm0=P_ref.copy())
        P2 = mf2.make_rdm1()
        return P2[0] + P2[1] if np.ndim(P2) == 3 else P2

    rng = np.random.default_rng(7)
    print('  eps=1e-4 随机 dS（5 个），对比推导(Tsum) vs 定标(zPFP):')
    for it in range(5):
        dS = rng.standard_normal((nao, nao))
        dS = dS + dS.T
        dS /= np.linalg.norm(dS)

        eps = 1e-4
        P_plus = P_of_S(S0 + eps * dS)
        P_minus = P_of_S(S0 - eps * dS)
        num = (np.einsum('ij,ij->', G_P, P_plus)
               - np.einsum('ij,ij->', G_P, P_minus)) / (2 * eps)

        # 解析：orth 部分，两种候选
        # A0 = -Tr[dS G_P P]（我推导 (5.7a) 用的，无 S^-1）
        # A1 = -1/2 Tr[dS (P G_P S^{-1} + S^{-1} G_P P)]（标准 Pulay，含 S^-1）
        S_inv = np.linalg.inv(S0)
        orth_A0 = -np.einsum('ij,ij->', dS, G_P @ P_tot)
        orth_A1 = -0.5 * np.einsum('ij,ij->', dS,
                                   P_tot @ G_P @ S_inv + S_inv @ G_P @ P_tot)
        Smo = C.T @ dS @ C
        Tsum = -0.5 * (np.einsum('ij,ij->', Smo, Fa @ ka.T + ka.T @ Fa)
                       + np.einsum('ij,ij->', Smo, Fb @ kb.T + kb.T @ Fb))
        zPFP = -(np.einsum('ij,ij->', Smo, ka @ (Fa * n_a[None, :]))
                 + np.einsum('ij,ij->', Smo, kb @ (Fb * n_b[None, :])))
        # 残差 = 数值 - orth（应被 Tsum 或 zPFP 匹配）
        r_A0 = num - orth_A0
        r_A1 = num - orth_A1
        print(f'  dS[{it}]: 数值={num: .6e}')
        print(f'      orthA0={orth_A0: .6e}  Tsum匹配={abs(r_A0-Tsum):.1e}  '
              f'zPFP匹配={abs(r_A0-zPFP):.1e}')
        print(f'      orthA1={orth_A1: .6e}  Tsum匹配={abs(r_A1-Tsum):.1e}  '
              f'zPFP匹配={abs(r_A1-zPFP):.1e}')


if __name__ == '__main__':
    main()
