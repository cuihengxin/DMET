"""几何优化示例：用 SSDMET 解析梯度优化 H2 键长。

两种方式：
  (1) scipy.optimize.minimize (BFGS) —— 标准、无第三方依赖，推荐
  (2) 最速下降循环 —— 学习用

运行：cd 8dmet4reac/DMET && python examples/test_example/grad_geomopt.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet


def dmet_energy_and_grad(mol, imp_idx, bath_norb=1, conv_tol=1e-12):
    """返回 (E, grad) for a given molecule."""
    mf = scf.RHF(mol) if mol.spin == 0 else scf.rohf.ROHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF not converged')

    d = ssdmet.SSDMET(mf, title='g', imp_idx=imp_idx,
                      bath_norb=bath_norb, verbose=0)
    d.build(save_chk=False)
    E = d.es_mf.e_tot + d.fo_ene
    de = d.nuc_grad_method().kernel()
    return E, de


def optimize_bfgs(mol0, imp_idx, bath_norb=1, conv_tol=1e-12):
    """scipy BFGS：标准几何优化。坐标单位 Bohr（与梯度一致）。"""
    from scipy.optimize import minimize

    def fg(x):
        m = mol0.copy()
        m.set_geom_(x.reshape(-1, 3), unit='Bohr')
        m.build()
        E, de = dmet_energy_and_grad(m, imp_idx, bath_norb, conv_tol)
        return float(E), de.ravel()

    x0 = mol0.atom_coords().ravel()
    res = minimize(fg, x0, jac=True, method='BFGS',
                   options={'gtol': 1e-8, 'maxiter': 50})
    return res


def optimize_steepest(mol0, imp_idx, bath_norb=1, nsteps=15, step=0.05):
    """最速下降（学习用）."""
    coords = mol0.atom_coords().copy()
    print('  iter       E (Hartree)        R (Bohr)       |grad|max')
    for it in range(nsteps):
        m = mol0.copy()
        m.set_geom_(coords, unit='Bohr')
        m.build()
        E, de = dmet_energy_and_grad(m, imp_idx, bath_norb)
        R = np.linalg.norm(coords[1] - coords[0])
        print(f'  {it:4d}  {E: .12f}  {R: .8f}  {np.abs(de).max(): .3e}')
        coords = coords - step * de
    return coords


if __name__ == '__main__':
    print('=' * 64)
    print('H2 几何优化（SSDMET 解析梯度）')
    print('=' * 64)

    # 起始键长 3.4 Bohr（远离平衡），验证梯度方向正确
    mol0 = gto.M(atom='H 0 0 0; H 0 0 3.4', basis='sto-3g', verbose=0)

    print('\n[方式 2] 最速下降（sto-3g，观察 E 单调降、R 单调降→方向正确）')
    coords_opt = optimize_steepest(mol0, imp_idx=[0], bath_norb=1)

    print('\n[方式 1] scipy BFGS 优化（标准）')
    res = optimize_bfgs(mol0, imp_idx=[0], bath_norb=1)
    r_opt = np.linalg.norm(res.x.reshape(-1, 3)[1] - res.x.reshape(-1, 3)[0])
    print(f'  BFGS 收敛: R = {r_opt:.5f} Bohr = {r_opt*0.529177:.5f} Å')
    print(f'  最终 |grad|max = {np.abs(res.jac).max():.3e}')
    print('  （H2 sto-3g 平衡键长约 1.386 Bohr = 0.733 Å）')
    assert abs(r_opt - 1.386) < 0.05, '优化未收敛到正确键长'
    print('\n几何优化示例通过：梯度正确，优化收敛到平衡键长。')
