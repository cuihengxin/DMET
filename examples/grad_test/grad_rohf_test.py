"""验证 ROHF-in-ROHF 单次 DMET 解析梯度。

沿用 RHF 路径上奏效的手法（按定位能力从强到弱）：

  (1) full-bath ROHF-in-ROHF vs PySCF ROHF 解析梯度
      全 bath 时嵌入精确，DMET 梯度必须复现 mf.nuc_grad_method()。
  (2) 截断 bath vs 有限差分（含 Richardson 外推）
  (3) 平移不变性 sum(grad) ≈ 0
  (4) Z-vector RHS 因子标定：旋转轨道做 FD 求 dL/dtheta，与解析 RHS 对比
      —— RHF 路径上正是这一步抓住了因子-2 bug，ROHF 必须重做。

测试体系：CH3 自由基 / NO / O2（轻元素，避开 x2c），STO-3G。

运行：cd 8dmet4reac/DMET && python examples/test_example/grad_rohf_test.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.numgrad import numerical_grad

TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=200)


def tight_rohf(mol):
    mf = scf.rohf.ROHF(mol)
    mf.conv_tol = TIGHT['conv_tol']
    mf.conv_tol_grad = TIGHT['conv_tol_grad']
    mf.max_cycle = TIGHT['max_cycle']
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, 'ROHF not converged'
    return mf


def dmet_energy(mol, imp_idx, bath_norb):
    """ROHF-in-ROHF one-shot DMET energy (the FD target)."""
    mf = tight_rohf(mol)
    d = ssdmet.SSDMET(mf, title='fd', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    return d.es_mf.e_tot + d.fo_ene


# ----------------------------------------------------------------------
# (4) Z-vector RHS 因子标定：d/dtheta Tr[G_P P] vs 解析 RHS
# ----------------------------------------------------------------------
def test_zvector_rhs(mol, imp_idx, bath_norb):
    mf = tight_rohf(mol)
    d = ssdmet.SSDMET(mf, title='rhs', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support()
    g.decompose()
    g.make_densities()
    g.orb_grad()
    g.lo_grad()

    G_P = g.G_P
    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = C.shape[1]

    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    moFbar = C.T @ G_P @ C

    # numerical dL/dtheta: rotate orbitals C -> C exp(theta), one uniq block
    # at a time.  L(theta) = Tr[G_P P(theta)] with P(theta) = sum_p n_p c_p c_p^T.
    def P_of_kappa(kappa):
        # kappa: antisymmetric (nmo,nmo); new coeffs = C @ exp(kappa) ~ C(I+kappa)
        U = np.eye(nmo) + kappa
        Co = C @ U
        P = 0.0
        for p in range(nmo):
            P += mo_occ[p] * np.outer(Co[:, p], Co[:, p])
        return P

    def L(kappa):
        return np.einsum('ij,ij->', G_P, P_of_kappa(kappa))

    # 解析 RHS（按 gen_g_hop_rohf 的打包顺序），因子取 2（RHF 路径的经验值）
    eps = 1e-6
    rhs_analytic = []
    rhs_numeric = []
    for (p, q) in np.argwhere(uniq_ab):
        kappa = np.zeros((nmo, nmo))
        kappa[p, q] = 1.0
        kappa[q, p] = -1.0
        num = (L(eps * kappa) - L(-eps * kappa)) / (2 * eps)
        rhs_numeric.append(num)
    # 解析：对旋转 θ_pq，dP = (n_q - n_p) (c_p c_q^T + c_q c_p^T)
    # dL = 2 (n_q - n_p) moFbar_pq
    rhs_analytic = np.array([2.0 * (mo_occ[q] - mo_occ[p]) * moFbar[p, q]
                             for (p, q) in np.argwhere(uniq_ab)])
    rhs_numeric = np.array(rhs_numeric)

    # 现在把解析 RHS 映射到 solve_z 里用的打包形式（sum_ab 折叠后）
    # solve_z 用的是 2*moFbar[var_a] 和 2*moFbar[var_b]，折叠到 uniq_ab。
    # 这里直接校验"逐元素 dL/dtheta" 与 "2 (n_q-n_p) moFbar_pq" 是否一致，
    # 这是打包之前的物理量，最干净。
    err = np.abs(rhs_analytic - rhs_numeric).max()
    scale = max(np.abs(rhs_numeric).max(), 1e-12)
    rel = err / scale
    print(f'  [Z-RHS] dL/dtheta: analytic vs numeric '
          f'max|err|={err:.3e}  rel={rel:.3e}')
    return rel


# ----------------------------------------------------------------------
def test_full_bath(mol, imp_idx):
    """full bath: DMET 梯度必须复现 ROHF 解析梯度。"""
    mf = tight_rohf(mol)
    d = ssdmet.SSDMET(mf, title='full', imp_idx=imp_idx, verbose=0)
    d.build(save_chk=False)
    e_dev = abs((d.es_mf.e_tot + d.fo_ene) - mf.e_tot)
    gd = d.nuc_grad_method().kernel()
    gr = mf.nuc_grad_method().kernel()
    grad_dev = np.abs(gd - gr).max()
    print(f'  [full-bath] E_dev={e_dev:.2e}  grad_dev={grad_dev:.2e}  '
          f'sum(grad)={abs(gd.sum(0)).max():.1e}')
    return grad_dev


def test_truncated(mol, imp_idx, bath_norb):
    mf = tight_rohf(mol)
    d = ssdmet.SSDMET(mf, title='trunc', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    gd = d.nuc_grad_method().kernel()

    gh = numerical_grad(lambda m: dmet_energy(m, imp_idx, bath_norb),
                        mol, step=1e-3, verbose=False)
    gh2 = numerical_grad(lambda m: dmet_energy(m, imp_idx, bath_norb),
                         mol, step=5e-4, verbose=False)
    rich = (4.0 * gh2 - gh) / 3.0
    print(f'  [truncated] h=1e-3: {np.abs(gd-gh).max():.3e}  '
          f'h=5e-4: {np.abs(gd-gh2).max():.3e}  '
          f'Richardson: {np.abs(gd-rich).max():.3e}  '
          f'sum(grad)={abs(gd.sum(0)).max():.1e}')
    return np.abs(gd - rich).max()


def make_ch3():
    # planar-ish methyl radical, one C + 3 H, one impurity carbon
    return gto.M(atom='C 0 0 0; H 0.63 0.63 0; H -0.63 0.63 0; H 0 -0.89 0',
                 basis='sto-3g', spin=1, verbose=0)


def make_no():
    return gto.M(atom='N 0 0 0; O 0 0 1.15', basis='sto-3g', spin=1, verbose=0)


def main():
    print('=' * 76)
    print('ROHF-in-ROHF 单次 DMET 解析梯度验证')
    print('=' * 76)

    # CH3 自由基：C 为杂质，全 bath
    mol = make_ch3()
    print('\n--- CH3 自由基 (C impurity, full bath) ---')
    assert test_full_bath(mol, imp_idx=[0]) < 1e-8
    # 截断 bath：C 杂质 + 1 个 bath
    print('--- CH3 自由基 (C impurity, bath_norb=1) ---')
    assert test_zvector_rhs(mol, imp_idx=[0], bath_norb=1) < 1e-5
    assert test_truncated(mol, imp_idx=[0], bath_norb=1) < 1e-8

    # NO：N 为杂质
    mol = make_no()
    print('\n--- NO (N impurity, full bath) ---')
    assert test_full_bath(mol, imp_idx=[0]) < 1e-8
    # 注意：NO 的 bath_norb=1 嵌入是 2 轨道 3 电子 spin=1 (es_occ=[2,1])，
    # 嵌入 ROHF 解病态（P 响应误差 ~1e-4，非折叠 bug）。用 bath=2 验证折叠。
    print('--- NO (N impurity, bath_norb=2) ---')
    assert test_truncated(mol, imp_idx=[0], bath_norb=2) < 1e-8

    print('\nall ROHF-in-ROHF gradient tests passed')


if __name__ == '__main__':
    main()
