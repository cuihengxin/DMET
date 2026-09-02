import os, sys
sys.path.insert(0, '/Users/cuihengxin/Desktop/2025-2030phd/8dmet4reac/DMET')
import os
import numpy as np
from pyscf import gto, scf, df
from embed_sim import ssdmet, sacasscf_mixer, siso, rdiis
currentfilename = os.path.basename(sys.argv[0])
title = currentfilename.split('.py')[0]

def get_mol():
     mol = gto.M(atom = '''
Fe 0.0000000 0.0000000 0.0000000
N -1.4273948 1.5139007 0.0000000
N -1.4273948 -1.5139007 0.0000000
N 1.4273948 -1.5139007 0.0000000
N 1.4273948 1.5139007 0.0000000
C -1.2355905 2.8182897 0.0000000
C 0.0000000 3.4670852 0.0000000
C 1.2355905 2.8182897 0.0000000
C -1.2355905 -2.8182897 0.0000000
C 0.0000000 -3.4670852 0.0000000
C 1.2355905 -2.8182897 0.0000000
H -2.1154041 3.4648465 0.0000000
H 0.0000000 4.5476713 0.0000000
H 2.1154041 3.4648465 0.0000000
H 2.4144279 1.2851136 0.0000000
H -2.4144279 1.2851136 0.0000000
H -2.4144279 -1.2851136 0.0000000
H -2.1154041 -3.4648465 0.0000000
H 0.0000000 -4.5476713 0.0000000
H 2.1154041 -3.4648465 0.0000000
H 2.4144279 -1.2851136 0.0000000
     ''',
     basis={'default': 'def2svp', 'Fe':'def2tzvp'}, symmetry=0 ,spin = 2,charge = 0,verbose= 4)

     return mol

mol = get_mol()

'''
Density fitting and scalar relativistic effects can be applied together,
regardless to the order you apply the decoration.
'''
mf = scf.rohf.ROHF(mol).density_fit().x2c()

'''
Sometimes it is useful to save it to disk for re-use in later calculations.
This can be achieved by specifying a HDF5 file by setting _cderi_to_save.
The saved DF tensor can be used later by setting _cderi to the HDF5 file.
If I/O free treatment is needed, just skip the following part.
'''
cderi_fname = title + '_cderi.h5'
if not os.path.exists(cderi_fname):
    mydf = df.df.DF(mol)
    mydf.auxbasis = mf.with_df.auxbasis
    mydf._cderi_to_save = cderi_fname
    mydf.build()
    mf.with_df = mydf
else:
    mf.with_df._cderi = cderi_fname
    mf.with_df.auxmol = df.addons.make_auxmol(mol)

chk_fname = title + '_rohf.chk'
mf.diis = rdiis.RDIIS(rdiis_prop='dS',imp_idx=mol.search_ao_label(['Fe.*d']),power=0.2)
mf.chkfile = chk_fname
mf.init_guess = 'atom'
mf.level_shift = .1
mf.max_cycle = 1000
mf.max_memory = 100000
mf.kernel()

'''
Switch to DF-DMET with density_fit decoration
'''
mydmet = ssdmet.SSDMET(mf, title=title, imp_idx='Fe.*d', bath_option={'ROMP2':1e-3}, es_natorb=False, verbose=4).density_fit()
# if impurity is not assigned, the orbitals on the first atom is chosen as impurity
mydmet.build(save_chk = False,mp2method='sos')
#mydmet.build(save_chk = False)

ncas, nelec, es_mo = mydmet.avas('Fe 3d', minao='def2tzvp', threshold=0.5, openshell_option=3)

es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec, statelis = [1,0,1,0,1])
es_cas.kernel(es_mo)

es_ecorr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_ecorr
total_cas = mydmet.total_cas(es_cas)
Ha2cm = 219474.63
kcal = 627.509
print(f"Spin difference:, { (es_cas.fcisolver.e_states[2]-es_cas.fcisolver.e_states[0])*kcal}")
print(f"Spin difference:, { (es_cas.fcisolver.e_states[1]-es_cas.fcisolver.e_states[0])*kcal}")
np.savetxt(mydmet.title+'_opt.txt',(es_cas.fcisolver.e_states-np.min(es_cas.fcisolver.e_states))*kcal,fmt='%.6f')
'''
Density fitting can be used to accelerate the calculation of SOC 2e integrals
by setting a DF object to the with_df attribute.
''''''
mysiso = siso.SISO(title, total_cas, verbose=5).density_fit()


mysiso.kernel()'''
