"""Top-to-bottom CoSH4 input using a direct ANO-R0 Co 3d projector."""

import sys
sys.path.insert(0, '/Users/cuihengxin/Desktop/2025-2030phd/8dmet4reac/iao_dmet/DMET/')

import json
import numpy as np
from pyscf import gto, scf, tools
from embed_sim import sacasscf_mixer, ssdmet, siso
from embed_sim.impurity_projector import project_reference_orbitals
from examples.test_example.impurity_projector_cosh4 import get_mol


title = 'CoSH4_projected_imp_simple'
mol = get_mol(verbose=4)

# Parent basis in get_mol() is used for SCF and all molecular integrals.
mf = scf.rohf.ROHF(mol).x2c().density_fit()
mf.chkfile = title + '_rohf.chk'
mf.init_guess = 'chk'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()
assert mf.converged, 'ROHF did not converge'

# Independent reference basis: only its Co 3d radial functions define imp_orb.
ano_r0_co = gto.basis.load('ano-r0', 'Co')
imp_orb, projector_info = project_reference_orbitals(mol, aolabels = 'Co 3d', reference_basis = {'default': 'minao', 'Co': ano_r0_co})
assert imp_orb.shape == (mol.nao_nr(), 5)
assert projector_info['projection_min'] > 0.99
print('projector completeness:', projector_info['projection_eigenvalues'])
with open(title + '_projector.json', 'w') as fh:
    json.dump(projector_info, fh, indent=2)

# Direct projector DMET.  Do not pass imp_idx, basis_rot, iaopao, ip_iao,
# or imp4ip: these options belong to the old AVAS/localization branch.
mydmet = ssdmet.SSDMET(
    mf, title=title, imp_orb=imp_orb,
    bath_option={'ROMP2': 39}, threshold=1e-12,
    es_natorb=False).density_fit()
mydmet.build(save_chk=False)
assert mydmet.es_mf.converged, 'embedded ROHF did not converge'
assert mydmet.nes == 52, 'expected 52 embedded orbitals'

s = mol.intor_symmetric('int1e_ovlp')
imp_svals = np.linalg.svd(
    imp_orb.conj().T @ s @ mydmet.es_orb[:, :5], compute_uv=False)
print('impurity subspace singular values:', imp_svals)
assert np.allclose(imp_svals, 1.0, atol=1e-7, rtol=0)
print('nimp =', 5, 'nbath =', mydmet.nes - 5, 'nes =', mydmet.nes)
print('HF-in-HF error =', mydmet.es_mf.e_tot + mydmet.fo_ene - mf.e_tot)

# This AVAS is only for selecting CAS(7e,5o) inside the embedded space.
ncas, nelec, es_mo = mydmet.avas(
    'Co 3d', minao='def2tzvp', threshold=0.5, openshell_option=3)
ncas, nelec = int(ncas), int(nelec)
assert (ncas, nelec) == (5, 7)
es_cas = sacasscf_mixer.sacasscf_mixer(
    mydmet.es_mf, ncas, nelec, statelis=[0, 40, 0, 10])
es_cas.max_memory = mf.max_memory
es_cas.max_cycle_macro = 100
es_cas.kernel(es_mo)
assert es_cas.converged, 'SA-CASSCF did not converge'

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
assert np.isfinite(es_ecorr).all()
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
tools.molden.from_mcscf(total_cas, title + '_cas.molden', cas_natorb=True)

mysiso = siso.SISO(title, total_cas, save_mag=True)
mysiso.kernel()
assert len(mysiso.mag_ene) == 120
print('SISO energies written to', title + '_mag.txt')
