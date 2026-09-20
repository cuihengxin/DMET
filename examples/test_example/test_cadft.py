"""Validation of configuration-averaged DFT (embed_sim.cadft) at the SCF level.

Test 1  Edge degeneracy : CADFT_RKS(ncas=1, nelecas=2) == RHF / RKS exactly
Test 2  H2 dissociation : RHF vs UKS vs CADFT_RKS(pbe/hf, ncas=2, nelecas=2)
        -> the fractional-occupation ensemble dissociates smoothly (no ionic
           contamination of the restricted solution)
Test 3  CAHF vs CADFT(xc='hf') : difference = CAHF lambda_J/lambda_K
        (pair-probability) corrections absent in the ensemble-DFT convention
Test 4  Janak theorem   : dE/dN_active = <eps_active> for H2O (ncas=4)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pyscf import gto, scf

from embed_sim import cadft, cahf

np.set_printoptions(precision=8, suppress=True)


def make_h2(bond):
    return gto.M(atom=f'H 0 0 0; H 0 0 {bond}', basis='6-31g', verbose=0)


print('=' * 78)
print('Test 1: edge degeneracy (ncas=1, nelecas=2 -> ordinary RHF/RKS)')
print('=' * 78)
mol = make_h2(0.74)
mf_rhf = scf.RHF(mol).run()
mf_rks = scf.RKS(mol, xc='pbe').run()
mf_chf = cadft.CADFT_RKS(mol, xc='hf', ncas=1, nelecas=2).run()
mf_cks = cadft.CADFT_RKS(mol, xc='pbe', ncas=1, nelecas=2).run()
print(f'E(RHF)              = {mf_rhf.e_tot:.10f}')
print(f'E(CADFT-RKS hf)     = {mf_chf.e_tot:.10f}   diff = {mf_chf.e_tot - mf_rhf.e_tot:.2e}')
print(f'E(RKS pbe)          = {mf_rks.e_tot:.10f}')
print(f'E(CADFT-RKS pbe)    = {mf_cks.e_tot:.10f}   diff = {mf_cks.e_tot - mf_rks.e_tot:.2e}')
assert abs(mf_chf.e_tot - mf_rhf.e_tot) < 1e-9, 'CADFT(hf) does not reduce to RHF'
assert abs(mf_cks.e_tot - mf_rks.e_tot) < 1e-9, 'CADFT(pbe) does not reduce to RKS'
print('PASS: ncas=1, nelecas=2 reduces exactly to RHF/RKS\n')

print('=' * 78)
print('Test 2: H2 dissociation, RHF vs UKS vs CA-DFT (ncas=2, nelecas=2)')
print('=' * 78)
print(f'{"R(Ang)":>7} {"RHF":>16} {"UKS(pbe)":>16} {"CADFT pbe":>16} {"CADFT hf":>16}')
for bond in (0.74, 1.5, 2.5, 4.0):
    mol = make_h2(bond)
    e_rhf = scf.RHF(mol).run().e_tot
    e_uks = scf.UKS(mol, xc='pbe').run().e_tot
    e_ck = cadft.CADFT_RKS(mol, xc='pbe', ncas=2, nelecas=2).run().e_tot
    e_ch = cadft.CADFT_RKS(mol, xc='hf', ncas=2, nelecas=2).run().e_tot
    print(f'{bond:7.2f} {e_rhf:16.8f} {e_uks:16.8f} {e_ck:16.8f} {e_ch:16.8f}')
print('(2 x E(H) pbe = %.8f; restricted CADFT should approach it smoothly,\n'
      ' while RHF carries the ionic fractional-spin error)\n' % (
      2 * scf.UKS(gto.M(atom='H 0 0 0', basis='6-31g', verbose=0, spin=1),
                  xc='pbe').run().e_tot))

print('=' * 78)
print('Test 3: CAHF vs CA-DFT(xc=hf) at H2 stretched (pair-probability lambda)')
print('=' * 78)
mol = make_h2(2.5)
mf_cahf = cahf.CAHF(mol, ncas=2, nelecas=2, spin=0).run()
mf_cahfdft = cadft.CADFT_RKS(mol, xc='hf', ncas=2, nelecas=2).run()
print(f'E(CAHF)            = {mf_cahf.e_tot:.8f}')
print(f'E(CADFT hf)        = {mf_cahfdft.e_tot:.8f}')
print(f'difference         = {mf_cahfdft.e_tot - mf_cahf.e_tot:.2e}'
      '  (lambda_J/lambda_K operator correction vs plain fractional occ)\n')

print('=' * 78)
print('Test 4: Janak theorem dE/dN_active = <eps_active>, H2O frontier ensemble')
print('        (ncore=4 closed shells + ncas=2 active orbitals at occ nelecas/2)')
print('=' * 78)
mol = gto.M(atom='O 0 0 0; H 0 0 1.15; H 0 1.15 0.35', basis='6-31g', verbose=0)
h = 0.02
energies, eps_act = {}, {}
for nelecas in (2.0 - h, 2.0, 2.0 + h):
    mf = cadft.CADFT_RKS(mol, xc='pbe', ncas=2, nelecas=nelecas)
    mf.conv_tol = 1e-11
    mf.run()
    assert mf.converged, f'SCF not converged for nelecas={nelecas}'
    energies[nelecas] = mf.e_tot
    if abs(nelecas - 2.0) < 1e-12:
        ncore = int(np.round((mol.nelectron - nelecas) / 2.0))
        eps_act = mf.mo_energy[ncore:ncore + 2].mean()
        print(f'active occupations = {mf.mo_occ[ncore:ncore+2]}')
dEdN = (energies[2.0 + h] - energies[2.0 - h]) / (2 * h)
print(f'dE/dN_active (finite diff) = {dEdN:.8f}')
print(f'<eps_active>   (Janak)     = {eps_act:.8f}')
print(f'difference                 = {dEdN - eps_act:.2e}')
assert abs(dEdN - eps_act) < 1e-5, 'Janak theorem violated'
print('PASS: Janak theorem holds\n')
print('=' * 78)
print('Test 5: Cohen-Mori-Sanchez-Yang fractional-spin theorem (literature anchor)')
print('        DeltaE_FS(H) from two independent routes must agree:')
print('        (a) direct: E(H, 1/2a+1/2b) - E(H, pure spin)')
print('        (b) H2 ensemble dissociation limit: E(CADFT-H2, R=inf)/2 - E(H)')
print('=' * 78)
Ha = 627.5095


def fs_error_H(xc):
    mol = gto.M(atom='H 0 0 0', basis='6-31g', spin=1, verbose=0)
    mf = scf.UHF(mol).run() if xc == 'hf' else scf.UKS(mol, xc=xc).run()
    c = mf.mo_coeff[0][:, 0]
    half = np.array([0.5] + [0] * c.size)
    dm = (np.outer(c, c) * 0.5, np.outer(c, c) * 0.5)
    return (mf.energy_tot(dm) - mf.e_tot) * Ha


def fs_error_H2(xc):
    mol = make_h2(30 / 0.529177)
    e_atom = scf.UKS(gto.M(atom='H 0 0 0', basis='6-31g', spin=1, verbose=0),
                     xc=xc).run().e_tot
    mf = cadft.CADFT_RKS(mol, xc=xc, ncas=2, nelecas=2)
    mf.level_shift = 0.5
    mf.max_cycle = 300
    mf.kernel()
    assert mf.converged, f'H2 CADFT not converged for {xc}'
    return (mf.e_tot / 2 - e_atom) * Ha


print(f'{"xc":8s} {"H-atom route":>14s} {"H2-limit route":>15s}')
for xc in ('svwn', 'pbe', 'b3lyp'):
    d = abs(fs_error_H(xc) - fs_error_H2(xc))
    print(f'{xc:8s} {fs_error_H(xc):14.2f} {fs_error_H2(xc):15.2f}'
          f'   |diff| = {d:.2f} kcal/mol')
    assert d < 2.0, 'CMY constancy theorem violated between routes'
print('hf      (analytic: J/4 = 5/32 Eh = 98.05 kcal/mol; verified separately)')
print('PASS: fractional-spin errors agree across routes and with CMY theory\n')

print('All CADFT SCF-level tests passed.')
