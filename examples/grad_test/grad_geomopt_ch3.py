"""CH3 自由基（开壳层 ROHF）结构优化：SSDMET 解析梯度 + scipy BFGS。

ROHF 优化比 RHF 更敏感，三个关键设置：
  1. warm start：每步用上一几何的密度作 SCF 初猜（开壳层 SCF 常因试探坐标发散）
  2. level_shift 重试：SCF 不收敛时加 level_shift 再算
  3. full bath（imp=C.* + bath=3 = 全部 AO）：DMET=ROHF，梯度=RHF 精确

运行：cd 8dmet4reac/DMET && python examples/test_example/grad_geomopt_ch3.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf
from scipy.optimize import minimize

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet

IMP_IDX = 'C.*'          # 杂质 = C 全部 AO
BATH_NORB = 3            # 3 个 H 各一 bath = full bath（nao=8）
CONV_TOL = 1e-11
GTOL = 1e-7


def ch3_mol(r_C, ang_deg=120.0):
    """平面 CH3，C 在原点，3 个 H 间隔 120°，C-H 键长 r_C (Bohr)。"""
    ang = np.radians(ang_deg)
    Hs = [f'H {r_C*np.cos(ang + i*2*np.pi/3):.8f} '
          f'{r_C*np.sin(ang + i*2*np.pi/3):.8f} 0' for i in range(3)]
    return gto.M(atom='C 0 0 0\n' + '\n'.join(Hs), basis='ccpvdz',
                 spin=1, verbose=0)


def make_optimizer(mol0):
    """返回 fg(x) -> (E, grad)。用 warm start + level_shift 保证开壳层 SCF 稳定。"""
    last_dm = [None]

    def fg(x):
        xa = x.reshape(-1, 3)
        m = mol0.copy()
        m.set_geom_(xa, unit='Bohr')
        m.build()
        mf = scf.rohf.ROHF(m)
        mf.conv_tol = CONV_TOL
        mf.max_cycle = 500
        mf.verbose = 0
        if last_dm[0] is None:
            mf.kernel()
        else:
            mf.kernel(dm0=last_dm[0])          # warm start
        if not mf.converged:
            mf.level_shift = 0.4               # SCF 发散时 level_shift 重试
            mf.kernel(dm0=mf.make_rdm1())
        if not mf.converged:
            raise RuntimeError('ROHF not converged')
        last_dm[0] = mf.make_rdm1()

        d = ssdmet.SSDMET(mf, title='ch3', imp_idx=IMP_IDX,
                          bath_norb=None, verbose=4)
        d.build(save_chk=False)
        E = d.es_mf.e_tot + d.fo_ene
        de = d.nuc_grad_method().kernel()
        return float(E), de.ravel()

    return fg


def main():
    # 初始 C-H 1.25 Å（2.36 Bohr），偏离平衡（~1.08 Å）
    mol0 = ch3_mol(2.36)
    fg = make_optimizer(mol0)

    print('=' * 64)
    print('CH3 自由基 ROHF-in-ROHF 结构优化（sto-3g, imp=C.*, full bath）')
    print('=' * 64)

    it = [0]

    def fg_print(x):
        xa = x.reshape(-1, 3)
        r = np.linalg.norm(xa[1] - xa[0])
        E, g = fg(x)
        it[0] += 1
        print(f'  step {it[0]:3d}:  E={E: .10f}  R(CH)={r:.4f} Bohr'
              f'  |grad|={np.abs(g).max(): .2e}')
        return E, g

    res = minimize(fg_print, mol0.atom_coords().ravel(), jac=True,
                   method='BFGS', options={'gtol': GTOL, 'maxiter': 40})
    opt = res.x.reshape(-1, 3)
    rs = [np.linalg.norm(opt[i] - opt[0]) for i in range(1, 4)]
    print('-' * 64)
    print(f'收敛: R(CH) = {rs[0]:.4f} Bohr = {rs[0]*0.5292:.8f} Å'
          f'  (3 键 {[f"{x:.4f}" for x in rs]})')
    print(f'      E = {res.fun:.10f}  |grad|max = {np.abs(res.jac).max():.2e}')
    print('参考: CH3 sto-3g ROHF 平衡 ~ R(CH)=1.08 Å，平面 120°')
    assert abs(rs[0] - 2.05) < 0.05, 'ROHF 优化未收敛到 CH3 平衡'
    print('\nROHF 结构优化验证通过。')


if __name__ == '__main__':
    main()
    mf2 = scf.ROHF(ch3_mol(2.05))
    mf2.init_guess = 'atom'
    mf2.kernel()
    from pyscf.geomopt.berny_solver import optimize
    mol_eq = optimize(mf2, maxsteps=100)
    mf3 = scf.ROHF(mol_eq)
    mf3.init_guess = 'atom'
    mf3.kernel()
    print(mf3.e_tot)
    rs = [np.linalg.norm(mol_eq.atom_coords()[i] - mol_eq.atom_coords()[0]) for i in range(1, 4)]
    print('-' * 64)
    print(f'收敛: R(CH) = {rs[0]:.4f} Bohr = {rs[0]*0.5292:.8f} Å'
          f'  (3 键 {[f"{x:.4f}" for x in rs]})')
