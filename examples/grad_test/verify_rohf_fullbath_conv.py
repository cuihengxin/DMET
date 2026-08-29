"""ROHF-in-ROHF full-bath 梯度 vs 全电子 ROHF 梯度，随收敛判据的行为。

用户论点：全 bath 嵌入精确，理论上 grad_dev 应到 ~1e-13（收紧 conv_tol 后）。
当前 full-bath 报 1.3e-12，差 10 倍 —— 查是收敛精度还是代码 bug。

关键：全 bath 时 G_P ~ 0（能量对 P 无显式依赖），Z-vector 及折叠项 ~ 0，
总梯度 = 冻结-P 通道 + 显式项 + nuc，因此 grad_dev 直接度量"冻结-P 通道"的精度。

若 grad_dev 随 conv_tol_grad 收紧而下降到 ~1e-13 -> 是收敛，代码对；
若卡在 ~1e-12 不降 -> 冻结-P 通道有 bug（自旋 r2 / veff 辅助 / 分自旋 2e 通道）。

运行：cd 8dmet4reac/DMET && python examples/test_example/verify_rohf_fullbath_conv.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


def run_case(mol, imp_idx, conv_tol, conv_tol_grad, label):
    mf = scf.rohf.ROHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        print(f'  {label}: global ROHF NOT converged')
        return None

    d = ssdmet.SSDMET(mf, title='fb', imp_idx=imp_idx, verbose=0)
    d.build(save_chk=False)

    # density consistency: embedded density vs exact ROHF density
    P = mf.make_rdm1()
    r1s = d.es_mf.make_rdm1()
    r1 = r1s[0] + r1s[1] if np.ndim(r1s) == 3 else r1s
    dm_act = d.es_orb @ r1 @ d.es_orb.T
    dm_core = d.fo_orb @ d.fo_orb.T * 2
    dens_err = np.abs(dm_core + dm_act - P).max()

    gd = d.nuc_grad_method().kernel()
    gr = mf.nuc_grad_method().kernel()
    grad_dev = np.abs(gd - gr).max()
    e_dev = abs((d.es_mf.e_tot + d.fo_ene) - mf.e_tot)

    print(f'  {label}: E_dev={e_dev:.2e}  grad_dev={grad_dev:.2e}  '
          f'density_err={dens_err:.2e}  sum(grad)={abs(gd.sum(0)).max():.1e}')
    return grad_dev


def main():
    mol_ch3 = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                         'H 0.02 -0.89 0.07',
                    basis='sto-3g', spin=1, verbose=0)
    print('=== CH3 自由基, full bath, C 为杂质 ===')
    for cg in (1e-10, 1e-12, 1e-13):
        run_case(mol_ch3, [0], 1e-12, cg, f'conv_tol_grad={cg:.0e}')

    mol_no = gto.M(atom='N 0 0 0; O 0 0 1.15', basis='sto-3g', spin=1,
                   verbose=0)
    print('=== NO, full bath, N 为杂质 ===')
    for cg in (1e-10, 1e-12, 1e-13):
        run_case(mol_no, [0], 1e-12, cg, f'conv_tol_grad={cg:.0e}')


if __name__ == '__main__':
    main()
