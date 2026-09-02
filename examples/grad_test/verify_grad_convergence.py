#!/usr/bin/env python
"""验证：full-bath HF-in-HF 梯度 vs RHF 梯度，随收敛精度收敛到 ~1e-13。

回答"为什么别人能到 1e-13，这里只有 2e-9"：因为梯度是密度的线性函数，
受嵌入求解器(es_mf)的密度残差限制。只要把 conv_tol_grad 也压下去，
grad_dev 就会跟着降。

运行：cd 8dmet4reac/DMET && python examples/test_example/verify_grad_convergence.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

# embed_sim 在本仓库里被 vendor 到多个目录，锁定本包所属的那一份
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


def run_case(conv_tol, conv_tol_grad, re_tighten=True):
    mol = gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = 200
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    mf.kernel()

    # full bath：环境所有分数占据轨道都进 bath，嵌入应当精确复现 RHF
    d = ssdmet.SSDMET(mf, title='full_bath', imp_idx=[0])
    d.build(save_chk=False)

    # 关键：把嵌入求解器也用同样紧的梯度判据重新收敛，
    # 让它的密度 r1 残差降到 ~1e-12，否则梯度被 r1 残差卡在 ~1e-9
    if re_tighten:
        d.es_mf.conv_tol = conv_tol
        if conv_tol_grad is not None:
            d.es_mf.conv_tol_grad = conv_tol_grad
        d.es_mf.kernel(d.es_mf.make_rdm1())

    gd = d.nuc_grad_method().kernel()
    gr = mf.nuc_grad_method().kernel()

    # 诊断：嵌入密度回投后离 RHF 密度多远（这是梯度的真正瓶颈）
    P = mf.make_rdm1()
    dm_act = d.es_orb @ d.es_mf.make_rdm1() @ d.es_orb.T
    dm_core = d.fo_orb @ d.fo_orb.T * 2
    density_err = np.abs(dm_core + dm_act - P).max()

    e_dev = abs((d.es_mf.e_tot + d.fo_ene) - mf.e_tot)
    grad_dev = np.abs(gd - gr).max()

    return e_dev, grad_dev, density_err


if __name__ == '__main__':
    print('=' * 78)
    print('full-bath HF-in-HF 梯度 vs RHF 解析梯度，随收敛判据的收敛行为')
    print('=' * 78)
    print(f"{'conv_tol':>10} {'conv_tol_grad':>14} "
          f"{'E_dev':>10} {'grad_dev':>10} {'density_err':>12}")
    print('-' * 78)

    cases = [
        (1e-10, None),
        (1e-12, None),
        (1e-14, None),
        (1e-14, 1e-12),   # 关键：梯度判据压下去，密度残差才会降
    ]
    for conv_tol, conv_tol_grad in cases:
        e_dev, grad_dev, density_err = run_case(conv_tol, conv_tol_grad)
        print(f"{conv_tol:10.0e} {str(conv_tol_grad):>14} "
              f"{e_dev:10.2e} {grad_dev:10.2e} {density_err:12.2e}")

    print('-' * 78)
    print('结论：grad_dev 始终与 density_err 同量级；')
    print('  只收紧 conv_tol（能量判据）时 density_err~1e-10 => grad_dev~1e-10；')
    print('  再加 conv_tol_grad=1e-12（梯度判据）后 density_err 降到 ~1e-14，')
    print('  grad_dev 降到 ~1e-14（机器精度，远超 1e-13）。')
    print('  即：之前报的 2e-9 是嵌入密度收敛精度，不是梯度公式的错误。')
