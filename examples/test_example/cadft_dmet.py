"""DMET with configuration-averaged DFT (CADFT) references, end-to-end.

For each system the same DMET pipeline is run from three starting points:
ordinary (R)HF/(R)OHF vs configuration-averaged DFT (embed_sim.cadft).
The reference only feeds the 1-RDM that selects impurity + bath
(build_embeded_subspace); the embedded solver is RHF/ROHF + MP2 as usual,
so all columns are directly comparable.

System 1 (closed shell): stretched H2O (R_OH = 1.6 A), impurity = O.
  references : RHF, CADFT_RKS(pbe, ncore=4, ncas=2, nelecas=2)
               (HOMO/LUMO-averaged ensemble; occupation pattern [2,2,2,2,1,1])
System 2 (open shell)  : stretched OH (R = 1.5 A, doublet), impurity = O.
  references : ROHF, CADFT_UKS(pbe, ncas=4, nelecas=5, spin=1)

Benchmark: full-system MP2 on the HF reference orbitals.
Note: the printed "deviation from DMET exact condition" is ~0 only for the
HF reference (HF-in-HF one-shot exactness).  For a CADFT reference it
measures the reference-vs-embedded-HF inconsistency and is expected to be
finite; the meaningful numbers are the MP2-in-DMET totals vs full MP2.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pyscf import gto, scf, mp

from embed_sim import ssdmet, cadft

np.set_printoptions(precision=6, suppress=True)


def report(name, mydmet, e_full_mp2=None):
    es, fo = mydmet.es_mf.e_tot, mydmet.fo_ene
    dev = es + fo - mydmet.mf_or_cas.e_tot
    print(f'\n--- DMET from {name} ---')
    print(f'nimp = {len(mydmet.imp_idx)}  nbath = {mydmet.nes - len(mydmet.imp_idx)}'
          f'  nfo = {mydmet.nfo}  nfv = {mydmet.nfv}')
    print(f'reference e_tot          = {mydmet.mf_or_cas.e_tot:.8f}')
    print(f'embedded (R)HF e_tot     = {es:.8f}')
    print(f'frozen env + nuc energy  = {fo:.8f}')
    print(f'deviation (exact cond.)  = {dev:.2e}')
    e_mp2, e_corr = mydmet.mp2_solver()
    if e_full_mp2 is not None:
        print(f'MP2-in-DMET total        = {e_mp2:.8f}   err vs full MP2 = {e_mp2 - e_full_mp2:+.2e}')
    return e_mp2


def run_cadft(mf, level_shift=0.5):
    """Convergence aid for position-assigned fractional occupations.

    Core/active frontier crossings can make plain DIIS oscillate (same recipe
    as examples/CAHF.py: level shift + longer cycle count)."""
    mf.level_shift = level_shift
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf.level_shift = 0.25
        mf.kernel()
    assert mf.converged, 'CADFT reference did not converge'
    return mf


print('#' * 78)
print('# System 1: stretched H2O, impurity O (closed shell)')
print('#' * 78)
mol = gto.M(atom='O 0 0 0; H 0 0 1.6; H 0 1.52 -0.45', basis='6-31g', verbose=0)
mf_rhf = scf.RHF(mol).run()
# mp2.e_tot already includes the HF energy
e_full_mp2 = mp.MP2(mf_rhf).run().e_tot
print(f'full-system RHF  = {mf_rhf.e_tot:.8f}')
print(f'full-system MP2  = {e_full_mp2:.8f}')

mf_cadft = run_cadft(cadft.CADFT_RKS(mol, xc='pbe', ncas=2, nelecas=2))
print(f'CADFT(pbe 4+2,2) = {mf_cadft.e_tot:.8f}')

d1 = ssdmet.SSDMET(mf_rhf, title='h2o_rhfreq', imp_idx='O', bath_norb=2)
d1.build()
report('RHF reference', d1, e_full_mp2)

d2 = ssdmet.SSDMET(mf_cadft, title='h2o_cadft', imp_idx='O', bath_norb=2)
d2.build()
report('CADFT-RKS(pbe 4+2,2) reference', d2, e_full_mp2)

print()
print('#' * 78)
print('# System 2: stretched OH radical, impurity O (open shell, doublet)')
print('#' * 78)
mol = gto.M(atom='O 0 0 0; H 0 0 1.5', basis='6-31g', spin=1, verbose=0)
mf_rohf = scf.ROHF(mol).run()
# mp2.e_tot already includes the HF energy
e_full_mp2 = mp.MP2(mf_rohf).run().e_tot
print(f'full-system ROHF = {mf_rohf.e_tot:.8f}')
print(f'full-system ROHF-MP2 = {e_full_mp2:.8f}')

mf_cadft = run_cadft(cadft.CADFT_UKS(mol, xc='pbe', ncas=4, nelecas=5, spin=1))
print(f'CADFT-UKS(pbe 4,5,1) = {mf_cadft.e_tot:.8f}')

d3 = ssdmet.SSDMET(mf_rohf, title='oh_rohfreq', imp_idx='O', bath_norb=2)
d3.build()
report('ROHF reference', d3, e_full_mp2)

d4 = ssdmet.SSDMET(mf_cadft, title='oh_cadft', imp_idx='O', bath_norb=2)
d4.build()
report('CADFT-UKS(pbe 4,5) reference', d4, e_full_mp2)

print('\nAll DMET end-to-end runs finished.')
