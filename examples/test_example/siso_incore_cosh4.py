"""Co(SH)4^2- test: DFSISO 3-center SOC integrals in-core (on-the-fly) vs out-core h5.

Converges one SA-CASSCF reference, then runs DFSISO.calc_z twice:
  1. out-core (default): int3c2e_pvxp1 written to <title>_int3c2e_pvxp1.h5 and read back
  2. in-core (incore=True): same integrals generated blockwise in memory, no h5

Checks that the SOC z tensors agree and reports timing / speedup of calc_z.
"""

import sys
from pathlib import Path
import time
import os

import numpy as np
from pyscf import gto, scf, tools

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed_sim import sacasscf_mixer, siso
from examples.test_example.impurity_projector_cosh4 import get_mol
from embed_sim import myavas

title = 'siso_incore_cosh4'
mol = get_mol(verbose=4)
print('nao =', mol.nao_nr())

mf = scf.rohf.ROHF(mol).x2c().density_fit()
mf.max_memory = 100000
mf.level_shift = .1
mf.max_cycle = 500
mf.kernel()
assert mf.converged, 'ROHF did not converge'
print('E_ROHF =', mf.e_tot)

av = myavas.AVAS(mf, ['Co 3d'], minao='def2tzvp', threshold=0.5, openshell_option=3)
ncas, nelec, mo_avas = av.kernel()
ncas, nelec = int(ncas), int(nelec)
print('ncas, nelec =', ncas, nelec)
assert (ncas, nelec) == (5, 7)

es_cas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis=[0, 3, 0, 1])
es_cas.max_memory = mf.max_memory
es_cas.max_cycle_macro = 50
es_cas.kernel(mo_avas)
assert es_cas.converged, 'SA-CASSCF did not converge'
print('E_SA-CASSCF =', es_cas.e_tot)

h5_name = title + '_int3c2e_pvxp1.h5'
if os.path.exists(h5_name):
    os.remove(h5_name)

# --- out-core (reference path, writes/reads the h5) ---
mysiso_out = siso.SISO(title, es_cas, save_mag=False).density_fit()
t0 = time.perf_counter()
z_out = mysiso_out.calc_z()
t_out = time.perf_counter() - t0
print(f'\nout-core calc_z: {t_out:.2f} s  (h5: {os.path.getsize(h5_name)/1e6:.1f} MB)')
h5_size = os.path.getsize(h5_name)/1e6
os.remove(h5_name)

# --- in-core (on-the-fly, no h5) ---
mysiso_in = siso.SISO(title, es_cas, save_mag=False).density_fit(incore=True)
t0 = time.perf_counter()
z_in = mysiso_in.calc_z()
t_in = time.perf_counter() - t0
print(f'in-core  calc_z: {t_in:.2f} s')

assert not os.path.exists(h5_name), 'in-core path must not create the h5 file'
dz = np.max(np.abs(np.asarray(z_out) - np.asarray(z_in)))
print(f'max |z_out - z_in| = {dz:.3e}')
assert dz < 1e-9, 'in-core and out-core SOC integrals disagree'

print(f'\nspeedup = {t_out/t_in:.2f}x  ({t_out:.2f} s -> {t_in:.2f} s)')
print('PASSED: in-core on-the-fly int3c2e_pvxp1 matches the h5 path')
