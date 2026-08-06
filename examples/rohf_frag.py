# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:10:08 2025

@author: Songqi Cao
"""

import numpy
from pyscf import gto, scf, tools
from embed_sim import rdiis, cahf
from fragment_init_guess import init_guess_by_fragment

title, spin, charge = 'Yb-YanW23', 0, 2
basis = {'default': 'x2c-SVPall', 'Yb': 'x2c-SVPall-2c'}
frag_info = [{'symbols': ['Yb'], 'charge': 2, 'spin': 0},
             {'symbols': ['C', 'H', 'N'], 'charge': 0, 'spin': 0}]
mol = gto.M(atom=title+'.xyz', basis=basis, spin=spin, charge=charge)

new_mol, dm, mo, occ = init_guess_by_fragment(mol, title+'.xyz', frag_info, basis, spin, charge)

mf = scf.rohf.ROHF(new_mol).x2c().density_fit()
mf.level_shift = 1.5
mf.init_guess = dm
mf.mo_coeff = mo
mf.mo_occ = occ
mf.conv_tol = 1e-6
mf.chkfile = title+'.chk'
#tools.molden.from_scf(mf, title+'_rohf.molden')
mf.conv_check = False
mf.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=mol.search_ao_label(['Yb.*f','Yb.5d']), power=0.2)
mf.kernel()

mf = mf.newton()
mf.conv_check = True
mf.conv_tol = 1e-9
mf.init_guess = 'chk'
#mf.max_cycle = 1
mf.kernel()
tools.molden.from_scf(mf, title+'_rohf.molden')

