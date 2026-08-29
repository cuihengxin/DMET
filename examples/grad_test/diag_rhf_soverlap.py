"""RHF 版 S 空间 FD：校准数值实现与 orth 形式（RHF 折叠已知为 zeta）。

RHF 下：dE/dS 的 P 响应 = orth + zeta（验证到 9e-15）。
若 数值 FD[Tr[G_P P(S)]] 匹配 orth + zeta -> 数值实现对，orth 形式对；
否则数值实现或 orth 形式有 bug。

运行：cd 8dmet4reac/DMET && python examples/test_example/diag_rhf_soverlap.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


def tight_rhf(m):
    mf = scf.RHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def main():
    # H2O closed shell, truncated bath (G_P != 0, validated RHF path)
    mol = gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96',
                basis='sto-3g', spin=0, verbose=0)
    mf = tight_rhf(mol)
    P_tot = mf.make_rdm1()

    d = ssdmet.SSDMET(mf, title='sv', imp_idx=[0], bath_norb=1, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()
    G_P = g.G_P.copy()

    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    occ = mo_occ > 0
    n = mo_occ[occ]
    S0 = mf.get_ovlp()
    nao = S0.shape[0]

    # RHF zeta = C (zvec * eps) C^T  (only vir-occ block nonzero)
    mo_energy = mf.mo_energy
    nmo = C.shape[1]
    zvec = np.zeros((nmo, nmo))
    zvec[np.ix_(~occ, occ)] = g.Z
    zeta = C @ (zvec * mo_energy[None, :]) @ C.T

    def P_of_S(S):
        mf2 = tight_rhf(mol)
        mf2.get_ovlp = lambda *a, **k: S
        mf2.kernel(dm0=P_tot.copy())
        return mf2.make_rdm1()

    rng = np.random.default_rng(3)
    print('RHF CH3+ (bath=1): 数值 FD vs orth + zeta')
    for it in range(5):
        dS = rng.standard_normal((nao, nao))
        dS = dS + dS.T
        dS /= np.linalg.norm(dS)
        eps = 1e-4
        num = (np.einsum('ij,ij->', G_P, P_of_S(S0 + eps * dS))
               - np.einsum('ij,ij->', G_P, P_of_S(S0 - eps * dS))) / (2 * eps)
        orth = -np.einsum('ij,ij->', dS, G_P @ P_tot)
        zeta_c = np.einsum('ij,ij->', dS, zeta)
        print(f'  dS[{it}]: num={num: .6e}  orth+zeta={orth+zeta_c: .6e}'
              f'  err={abs(num-orth-zeta_c):.1e}')


if __name__ == '__main__':
    main()
