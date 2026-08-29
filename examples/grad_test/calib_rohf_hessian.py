"""测量 pyscf.soscf.newton_ah.gen_g_hop_rohf 的归一化约定。

问题：g 和 h_op 到底是 dE/dtheta / d2E/dtheta2 本身，还是它们的一半？
这决定 Z-vector 方程 h_op(Z) = -rhs 里 rhs 该用 dL/dtheta 还是 dL/dtheta/2。

做法：沿一个旋转方向 x 转轨道，数值求 ROHF 能量的一阶、二阶导，
与 x·g 和 x·h_op(x) 对比，读出比值。

运行：cd 8dmet4reac/DMET && python examples/test_example/calib_rohf_hessian.py
"""

import os
import sys

import numpy as np
import scipy.linalg as sla
from pyscf import gto, scf
from pyscf.soscf import newton_ah

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)


def tight(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def rohf_masks(mo_occ):
    occa = mo_occ > 0
    occb = mo_occ == 2
    var_a = (~occa)[:, None] & occa
    var_b = (~occb)[:, None] & occb
    return occa, occb, var_a, var_b, var_a | var_b


def main():
    mol = gto.M(atom='C 0 0 0; H 0.63 0.61 0.05; H -0.60 0.65 -0.03; '
                     'H 0.02 -0.89 0.07',
                basis='sto-3g', spin=1, verbose=0)
    mf = tight(mol)
    C, mo_occ = mf.mo_coeff, mf.mo_occ
    nmo = C.shape[1]
    occa, occb, var_a, var_b, uniq_ab = rohf_masks(mo_occ)

    g, h_op, h_diag = newton_ah.gen_g_hop_rohf(mf, C, mo_occ)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(np.count_nonzero(uniq_ab)))
    x /= np.linalg.norm(x)

    # antisymmetric generator for this direction
    K = np.zeros((nmo, nmo))
    K[uniq_ab] = x
    K = K - K.T

    def E_of_theta(t):
        Ct = C @ sla.expm(t * K)
        dm = mf.make_rdm1(Ct, mo_occ)
        return mf.energy_elec(dm)[0]

    # numerical first and second derivative along x
    for h in (1e-4, 5e-5):
        e_p, e_m, e_0 = E_of_theta(h), E_of_theta(-h), E_of_theta(0.0)
        d1 = (e_p - e_m) / (2 * h)
        d2 = (e_p - 2 * e_0 + e_m) / h ** 2
        print(f'  h={h:.0e}:  dE/dtheta(num) = {d1: .8e}   '
              f'd2E/dtheta2(num) = {d2: .8e}')

    gx = float(np.dot(g, x))
    hx = float(np.dot(x, h_op(x)))
    print(f'\n  x . g          = {gx: .8e}')
    print(f'  x . h_op(x)    = {hx: .8e}')

    e_p, e_m, e_0 = E_of_theta(1e-4), E_of_theta(-1e-4), E_of_theta(0.0)
    d1 = (e_p - e_m) / 2e-4
    d2 = (e_p - 2 * e_0 + e_m) / 1e-8
    print(f'\n  ratio  (dE/dtheta) / (x.g)        = {d1/gx if abs(gx)>1e-14 else float("nan"): .6f}')
    print(f'  ratio  (d2E/dtheta2) / (x.h_op(x)) = {d2/hx: .6f}')
    print('\n若二阶比值 = 2 -> h_op 是真实 Hessian 的一半，')
    print('   Z-vector 的 rhs 应取 dL/dtheta / 2（即我现在的 Z 大了 2 倍）。')


if __name__ == '__main__':
    main()
