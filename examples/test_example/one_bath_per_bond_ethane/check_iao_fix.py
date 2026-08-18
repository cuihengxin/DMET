"""Quick check: construct_iao must build pmol with basis on ALL atoms (16 AOs
for ethane minao: 2 C x 5 + 6 H x 1) and emit no 'Basis not found' warnings."""
import sys
sys.path.insert(0, '/Users/cuihengxin/Desktop/2025-2030phd/8dmet4reac/DMET')

import numpy as np
from pyscf import gto, scf
from embed_sim import iao_helper

R = 1.54
d = 1.089
cosf, sinf = 1.0 / 3.0, 2.0 * np.sqrt(2.0) / 3.0
c0 = np.array([0.0, 0.0, 0.0])
c1 = np.array([0.0, 0.0, R])
ang = np.deg2rad([0.0, 120.0, 240.0])
ang2 = np.deg2rad([60.0, 180.0, 300.0])
u = np.array([[sinf * np.cos(a), sinf * np.sin(a), -cosf] for a in ang])
v = np.array([[sinf * np.cos(a), sinf * np.sin(a), cosf] for a in ang2])
atoms = [c0, c1] + [c0 + d * ui for ui in u] + [c1 + d * vi for vi in v]
atom_str = '; '.join(['C 0 0 0', 'C 0 0 %.6f' % R] +
                     ['H %.6f %.6f %.6f' % (x, y, z) for x, y, z in atoms[2:]])

mol = gto.M(atom=atom_str, basis='6-31g', verbose=0)
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

ao2iao, S1, pmol = iao_helper.construct_iao(mol, mf)
print('pmol.nao_nr() =', pmol.nao_nr(), '(expect 16)')
print('basis keys     =', sorted(pmol.basis.keys()), '(expect [\'C\', \'H\'])')
assert pmol.nao_nr() == 16, 'pmol still missing H basis functions!'
print('OK: fix works')
