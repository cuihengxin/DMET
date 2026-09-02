"""验证 FD 工具本身 + Richardson 外推（回答"数值梯度是否和 PySCF 比较过"）。

三层验证：
  (1) 我的 numerical_grad 作用在纯 RHF 上 vs PySCF 解析 RHF 梯度
      —— 校验"量尺"本身是准的
  (2) 我的 numerical_grad vs PySCF 官方 as_scanner() FD 惯用法
      —— 校验我的实现与 PySCF 社区标准做法一致
  (3) Richardson 外推 (4*g(h/2)-g(h))/3 消掉 O(h^2) 截断误差后，
      DMET 解析梯度 vs 高精度数值梯度
      —— 把之前 3e-7 的"残差"压下去，看是否真是 FD 截断误差

运行：cd 8dmet4reac/DMET && python examples/test_example/verify_numgrad_vs_pyscf.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

# embed_sim 在本仓库里被 vendor 到多个目录，这里锁定本包所属的那一份
# （相对脚本位置解析，且插到 sys.path 最前，避免抓到别的副本）
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.numgrad import numerical_grad

print(f'using embed_sim from: {os.path.dirname(ssdmet.__file__)}')

TIGHT = dict(conv_tol=1e-14, conv_tol_grad=1e-12, max_cycle=200)


def make_mol():
    return gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96',
                 basis='sto-3g', verbose=0)


def tight_rhf(m):
    mf = scf.RHF(m)
    mf.conv_tol = TIGHT['conv_tol']
    mf.conv_tol_grad = TIGHT['conv_tol_grad']
    mf.max_cycle = TIGHT['max_cycle']
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, 'RHF not converged'
    return mf


def rhf_energy(m):
    return tight_rhf(m).e_tot


def dmet_energy(m, imp_idx, bath_norb):
    mf = tight_rhf(m)
    d = ssdmet.SSDMET(mf, title='fd', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    return d.es_mf.e_tot + d.fo_ene


def richardson(g_h, g_h2):
    """消掉 O(h^2)：中心差分误差 ~ C h^2，(4 g(h/2) - g(h)) / 3 -> O(h^4)."""
    return (4.0 * g_h2 - g_h) / 3.0


def scanner_fd(mf, mol, step):
    """PySCF 官方 as_scanner() 惯用法做的中心差分（社区标准做法）。"""
    scanner = mf.as_scanner()
    coords = mol.atom_coords()
    de = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            ep = em = 0.0
            for s in (1, -1):
                c = coords.copy()
                c[ia, x] += s * step
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                e = scanner(m)
                if s > 0:
                    ep = e
                else:
                    em = e
            de[ia, x] = (ep - em) / (2 * step)
    return de


def main():
    mol = make_mol()
    mf = tight_rhf(mol)
    g_pyscf = mf.nuc_grad_method().kernel()   # 标准答案

    print('=' * 76)
    print('(1) 我的 numerical_grad 作用在纯 RHF 上 vs PySCF 解析梯度')
    print('    —— 校验 FD "量尺" 本身；应呈 O(h^2) 收敛')
    print('=' * 76)
    g_rhf = {}
    for step in (1e-3, 5e-4, 2e-4):
        g_rhf[step] = numerical_grad(rhf_energy, mol, step=step, verbose=False)
        err = np.abs(g_rhf[step] - g_pyscf).max()
        print(f'    h={step:.0e}: max|myFD - PySCF| = {err:.3e}')
    r = richardson(g_rhf[1e-3], g_rhf[5e-4])
    print(f'    Richardson : max|myFD - PySCF| = '
          f'{np.abs(r - g_pyscf).max():.3e}')

    print()
    print('=' * 76)
    print('(2) 我的 numerical_grad vs PySCF 官方 as_scanner() FD 惯用法')
    print('    —— 两者应逐位一致（同一个中心差分公式）')
    print('=' * 76)
    for step in (1e-3, 5e-4):
        g_sc = scanner_fd(mf, mol, step)
        print(f'    h={step:.0e}: max|myFD - scannerFD| = '
              f'{np.abs(g_rhf[step] - g_sc).max():.3e}   '
              f'(scannerFD vs analytic: {np.abs(g_sc - g_pyscf).max():.3e})')

    print()
    print('=' * 76)
    print('(3) 截断 bath DMET 解析梯度 vs 数值梯度（含 Richardson 外推）')
    print('    —— 若 3e-7 只是 FD 截断误差，外推后应大幅下降')
    print('=' * 76)
    imp_idx, bath_norb = [0], 1
    d = ssdmet.SSDMET(mf, title='trunc', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    g_dmet = d.nuc_grad_method().kernel()

    g_num = {}
    for step in (1e-3, 5e-4, 2e-4):
        g_num[step] = numerical_grad(
            lambda m: dmet_energy(m, imp_idx, bath_norb),
            mol, step=step, verbose=False)
        print(f'    h={step:.0e}: max|analytic - FD| = '
              f'{np.abs(g_dmet - g_num[step]).max():.3e}')
    r = richardson(g_num[1e-3], g_num[5e-4])
    print(f'    Richardson : max|analytic - FD| = '
          f'{np.abs(g_dmet - r).max():.3e}')
    r2 = richardson(g_num[5e-4], g_num[2e-4])
    print(f'    Richardson2: max|analytic - FD| = '
          f'{np.abs(g_dmet - r2).max():.3e}')

    print()
    print('结论：如果 (1) 与 (3) 的 h 依赖表现相同（都 O(h^2)、外推后同样下降），')
    print('      说明 DMET 解析梯度的精度已经到了 FD 量尺的分辨极限，')
    print('      即 3e-7 是差分截断误差，不是梯度公式的误差。')


if __name__ == '__main__':
    main()
