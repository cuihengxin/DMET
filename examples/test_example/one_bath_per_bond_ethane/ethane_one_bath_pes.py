'''
Ethane C-C stretch PES with ONE bath orbital per bond (CH3-group impurity).

This is the reproducible input for the calculation discussed in the project
notes: impurity = one CH3 group (atoms C0 + H1..H3), bath = 1 orbital
(bath_norb=1, the "one bath orbital per bond" scheme), MP2 as the embedded
solver, and two energy formulas compared against all-electron MP2:

  E_Direct = E(embedded MP2) + E(frozen occupied)          (bath-size sensitive)
  E_Corr   = E(full RHF) + e_corr(embedded MP2)            (correction-based,
                                                            recommended for PES)

The full per-geometry energies and energy differences are written to
ethane_one_bath_pes_results.txt.  The script also prints the environment
natural-orbital occupations and identifies the selected bath orbital (the
criterion: occupation closest to 1, i.e. largest min(lambda, 2-lambda)).

Run (from the repository root or with PYTHONPATH pointing at it):

    python examples/test_example/one_bath_per_bond_ethane/ethane_one_bath_pes.py

Required Python environment: pyscf, numpy, scipy, sympy (see DMET_main/CLAUDE.md).
'''
import sys
sys.path.append('/Users/cuihengxin/Desktop/2026phd/DMET_main')

import os
import numpy as np
from pyscf import gto, scf, mp
from embed_sim import ssdmet
from embed_sim.bath_selection import partition_env_by_bath_count


BASIS = '6-31g'
R_GRID = np.array([1.30, 1.40, 1.50, 1.54, 1.60, 1.70, 1.90,
                   2.10, 2.30, 2.60, 3.00])
IMP_LABELS = ['0 C.*', '2 H.*', '3 H.*', '4 H.*']   # CH3 group (C0 + its 3 H)
BATH_NORB = 1                                         # one bath orbital per bond


def ethane_mol(R, basis=BASIS):
    """Staggered ethane, C0-C1 along z, fixed C-H = 1.089 A."""
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
    mol1 = gto.M(atom=atom_str, basis=basis, verbose=4)
    mol1.tofile('ethane_one_bath_pes_mol_%s_R%.2f.xyz' % (basis, R), 'xyz')
    return mol1


def selected_bath_occupation(dmet):
    """Occupation of the environment natural orbital chosen as the bath."""
    from functools import reduce
    from pyscf.lo.orth import lowdin
    s = dmet.mol.intor_symmetric('int1e_ovlp')
    caolo, cloao = lowdin(s), lowdin(s) @ s
    ldm = reduce(np.dot, (cloao, dmet.dm, cloao.conj().T))
    env_idx = [i for i in range(ldm.shape[0]) if i not in dmet.imp_idx]
    lam = np.linalg.eigh(ldm[np.ix_(env_idx, env_idx)])[0]
    bath_idx, _, _ = partition_env_by_bath_count(lam, BATH_NORB, core_cutoff=0.5)
    return lam, bath_idx


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    rows = []

    for R in R_GRID:
        mol = ethane_mol(R)
        mf = scf.RHF(mol).x2c()
        mf.verbose = 4
        mf.run()

        # all-electron references
        e_full_hf = mf.e_tot
        mymp2_full = mp.MP2(mf)
        mymp2_full.verbose = 4
        mymp2_full.kernel()
        e_full_mp2 = mymp2_full.e_tot

        # embedded cluster: CH3 impurity + 1 bath orbital
        d = ssdmet.SSDMET(mf, title='ethane_1bath', imp_idx=IMP_LABELS,
                          bath_norb=BATH_NORB, verbose=4)
        d.build(save_chk=False)
        e_emb_hf = d.es_mf.e_tot + d.fo_ene
        mymp2 = mp.MP2(d.es_mf)
        mymp2.verbose = 0
        mymp2.max_memory = d.max_mem
        mymp2.kernel()
        e_emb_mp2 = mymp2.e_tot + d.fo_ene          # Direct formula total
        e_corr_emb = mymp2.e_corr                   # E_emb_MP2 - E_emb_HF
        e_dmet_corr = e_full_hf + e_corr_emb        # correction-based total
        lam, bath_idx = selected_bath_occupation(d)
        lam_bath = lam[bath_idx[0]]

        rows.append(dict(R=R, e_full_hf=e_full_hf, e_full_mp2=e_full_mp2,
                         e_emb_hf=e_emb_hf, e_emb_mp2=e_emb_mp2,
                         e_corr_emb=e_corr_emb, e_direct=e_emb_mp2,
                         e_corr=e_dmet_corr, lam_bath=lam_bath,
                         nbath=d.nes - len(d.imp_idx)))
        print('R = %5.2f A  E_full_HF=%.10f  E_full_MP2=%.10f'
              % (R, e_full_hf, e_full_mp2), flush=True)

    ieq = int(np.argmin([r['e_full_hf'] for r in rows]))
    dE_full_mp2 = [r['e_full_mp2'] - rows[ieq]['e_full_mp2'] for r in rows]
    dE_direct = [r['e_direct'] - rows[ieq]['e_direct'] for r in rows]
    dE_corr = [r['e_corr'] - rows[ieq]['e_corr'] for r in rows]

    lines = []
    sep = '-' * 150
    header = ('{:<6s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}'
              '{:>12s}{:>12s}{:>12s}').format(
        'R/A', 'E_full_HF', 'E_full_MP2', 'E_emb_HF', 'E_emb_MP2',
        'e_corr_emb', 'E_Direct', 'E_Corr', 'lam_bath', 'dE_fullMP2',
        'dE_Corr')
    print()
    print(sep)
    print(header)
    print(sep)
    lines += [sep, header, sep]
    for k, r in enumerate(rows):
        row = ('{:<6.2f}{:>14.8f}{:>14.8f}{:>14.8f}{:>14.8f}{:>14.8f}'
               '{:>14.8f}{:>14.8f}{:>12.4f}{:>12.6f}{:>12.6f}'.format(
            r['R'], r['e_full_hf'], r['e_full_mp2'], r['e_emb_hf'],
            r['e_emb_mp2'], r['e_corr_emb'], r['e_direct'], r['e_corr'],
            r['lam_bath'], dE_full_mp2[k], dE_corr[k]))
        print(row)
        lines.append(row)
    print(sep)
    lines.append(sep)

    print()
    print('PES error vs all-electron MP2 (mHa):  dE(method) - dE(full MP2)')
    print('-' * 78)
    lines.append('PES error vs all-electron MP2 (mHa):  dE(method) - dE(full MP2)')
    lines.append('-' * 78)
    err_direct = np.array([(dE_direct[k] - dE_full_mp2[k]) * 1000 for k in range(len(rows))])
    err_corr = np.array([(dE_corr[k] - dE_full_mp2[k]) * 1000 for k in range(len(rows))])
    hdr2 = ('{:<6s}{:>14s}{:>14s}{:>14s}'.format('R/A', 'err_Direct/mHa',
                                                 'err_Corr/mHa', 'dE_Direct'))
    print(hdr2)
    lines.append(hdr2)
    for k, r in enumerate(rows):
        row = '{:<6.2f}{:>14.3f}{:>14.3f}{:>14.6f}'.format(
            r['R'], err_direct[k], err_corr[k], dE_direct[k])
        print(row)
        lines.append(row)
    print('-' * 78)
    lines.append('-' * 78)
    for name, err in (('Direct', err_direct), ('Corr', err_corr)):
        msg = '%-7s formula: max|PES err| = %8.2f mHa, RMS = %8.2f mHa' % (
            name, np.max(np.abs(err)), np.sqrt(np.mean(err ** 2)))
        print(msg)
        lines.append(msg)

    # bath selection report at equilibrium
    r_eq = rows[ieq]
    mol = ethane_mol(r_eq['R'])
    s = mol.intor_symmetric('int1e_ovlp')
    from pyscf.lo.orth import lowdin
    from functools import reduce
    caolo, cloao = lowdin(s), lowdin(s) @ s
    mf = scf.RHF(mol).x2c()
    mf.verbose = 0
    mf.run()
    ldm = reduce(np.dot, (cloao, mf.make_rdm1(), cloao.conj().T))
    env_idx = [i for i in range(ldm.shape[0]) if i not in
               gto.mole._aolabels2baslst(mol, IMP_LABELS, base=0)]
    lam = np.linalg.eigh(ldm[np.ix_(env_idx, env_idx)])[0]
    frac = np.minimum(lam, 2.0 - lam)
    order = np.argsort(-frac, kind='stable')
    print()
    print('Bath selection report at R = %.2f A (equilibrium):' % r_eq['R'])
    print('  environment natural-orbital occupations lambda:')
    print('   ', np.array2string(np.round(lam, 4), precision=4, separator=', '))
    print('  ranking criterion min(lambda, 2-lambda) (largest = most entangled):')
    print('   ', np.round(frac[order], 4))
    print('  selected bath orbital: env index %d, lambda = %.6f'
          % (order[0], lam[order[0]]))
    print('  (criterion: occupation closest to 1 = largest min(lambda, 2-lambda))')
    lines.append('')
    lines.append('Bath selection report at R = %.2f A:' % r_eq['R'])
    lines.append('  env occupations: %s' % np.round(lam, 4))
    lines.append('  min(lambda,2-lambda): %s' % np.round(frac, 4))
    lines.append('  selected bath: env index %d, lambda = %.6f'
                 % (order[0], lam[order[0]]))
    lines.append('')
    lines.append('# eq index: R = %.2f A (minimum of full RHF)' % r_eq['R'])
    lines.append('# impurity: CH3 group (%s), bath_norb = %d, basis = %s, x2c RHF'
                 % (', '.join(IMP_LABELS), BATH_NORB, BASIS))
    lines.append('# E_Direct = E(emb MP2) + E(frozen occ); '
                 'E_Corr = E(full HF) + e_corr(emb MP2)')
    lines.append('# machine columns: R E_full_HF E_full_MP2 E_emb_HF E_emb_MP2 '
                 'e_corr_emb E_Direct E_Corr lam_bath dE_full_MP2 dE_Direct dE_Corr '
                 'err_Direct_mHa err_Corr_mHa')
    for k, r in enumerate(rows):
        lines.append('%.2f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.4f '
                     '%.6f %.6f %.6f %.3f %.3f' % (
            r['R'], r['e_full_hf'], r['e_full_mp2'], r['e_emb_hf'],
            r['e_emb_mp2'], r['e_corr_emb'], r['e_direct'], r['e_corr'],
            r['lam_bath'], dE_full_mp2[k], dE_direct[k], dE_corr[k],
            err_direct[k], err_corr[k]))

    outfile = os.path.join(outdir, 'ethane_one_bath_pes_results.txt')
    with open(outfile, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\nfull results written to', outfile)


if __name__ == '__main__':
    main()
