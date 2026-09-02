'''
One bath orbital per bond (Sun & Chan, JCTC 10, 3784 (2014)): small-molecule tests.

For every test molecule the script runs the single-shot DMET of `embed_sim` with
two bath-orbital selections:

1. default: environment natural orbitals with fractional occupation (threshold
   based, `threshold=1e-12`);
2. one bath orbital per bond: exactly `nbath = n_bonds(impurity, environment)`
   environment natural orbitals with occupation closest to 1.

The DMET total energy is `E(embedded low-level) + E(frozen occupied)` and is
compared with the full-system RHF energy.  With the default (tight-threshold)
selection the embedded cluster covers the whole molecule for these small
systems, so the DMET error is ~1e-12; the per-bond selection uses a smaller
cluster and exposes the quality of the fixed-size bath.

Both `SSDMET` (full-space Lowdin) and `AODMET` (environment-only Lowdin) are
tested.  N2 (triple bond) demonstrates that a single bond count per atom pair
is too small and that `bath_norb` must be raised manually (3).  OH is an
open-shell ROHF example.

Run from the repository root (or with `PYTHONPATH` pointing at it):

    python examples/test_example/one_bath_per_bond.py
'''

import os
import numpy as np
from pyscf import gto, scf
from embed_sim import ssdmet, aodmet


BASIS = '6-31g'

# (name, atom-string, impurity AO label, spin, charge, manual bath_norb)
SYSTEMS = [
    ('H2',  'H 0 0 0; H 0 0 0.74', '0 H.*', 0, 0, None),
    ('LiH', 'Li 0 0 0; H 0 0 1.60', '0 Li.*', 0, 0, None),
    ('H2O', 'O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', '0 O.*', 0, 0, None),
    ('CH4', 'C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; '
            'H 0.629 -0.629 -0.629; H -0.629 0.629 -0.629', '0 C.*', 0, 0, None),
    ('F2',  'F 0 0 0; F 0 0 1.41', '0 F.*', 0, 0, None),
    # triple bond: one bond per atom pair is too small; nbath=3 is required
    ('N2',  'N 0 0 0; N 0 0 1.10', '0 N.*', 0, 0, 3),
    ('OH',  'O 0 0 0; H 0 0 0.97', '0 O.*', 1, 0, None),
]


def make_mf(atom, spin, charge):
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, charge=charge, verbose=0)
    if spin:
        mf = scf.ROHF(mol).x2c()
    else:
        mf = scf.RHF(mol).x2c()
    mf.verbose = 0
    mf.run()
    return mol, mf


def dmet_energy(dmet):
    """Total DMET energy = embedded low-level energy + frozen-occupied energy."""
    return dmet.es_mf.e_tot + dmet.fo_ene


def run_dmet(cls, mf, title, imp_label, **kwargs):
    d = cls(mf, title=title, imp_idx=imp_label, verbose=0, **kwargs)
    d.build(save_chk=False)
    return d


def main():
    lines = []
    summary = []
    sep = '-' * 118
    header = ('{:<5s}{:>5s}{:>8s}{:>11s}{:>11s}{:>18s}{:>18s}{:>13s}'.format(
        'mol', 'nimp', 'nclust', 'nbath(def)', 'nbath(1/b)',
        'E_dmet(def)', 'E_dmet(1/b)', 'err(1/b)/mHa'))
    print(sep)
    print(header)
    print(sep)
    lines += [sep, header, sep]

    for name, atom, imp_label, spin, charge, nbath_manual in SYSTEMS:
        mol, mf = make_mf(atom, spin, charge)
        e_full = mf.e_tot

        # default threshold-based bath
        d_def = run_dmet(ssdmet.SSDMET, mf, name + '_def', imp_label)
        e_def = dmet_energy(d_def)
        nimp = len(d_def.imp_idx)
        nbath_def = d_def.nes - nimp

        # one bath orbital per bond (or manual count for N2)
        if nbath_manual is not None:
            kwargs = {'bath_norb': nbath_manual}
            tag = f'nbath={nbath_manual}'
        else:
            kwargs = {'bath_norb': 'per_bond'}
            tag = 'per_bond'
        d_pb = run_dmet(ssdmet.SSDMET, mf, name + '_pb', imp_label, **kwargs)
        e_pb = dmet_energy(d_pb)
        nbath_pb = d_pb.nes - nimp
        nclust = d_pb.nes
        err_mha = (e_pb - e_full) * 1000.0

        row = ('{:<5s}{:>5d}{:>8d}{:>11d}{:>11d}{:>18.10f}{:>18.10f}{:>13.3f}'.format(
            name, nimp, nclust, nbath_def, nbath_pb, e_def, e_pb, err_mha))
        print(row)
        lines.append(row)

        # open-shell AODMET per-bond comparison for closed-shell systems
        if spin == 0:
            d_ao = run_dmet(aodmet.AODMET, mf, name + '_ao', imp_label, **kwargs)
            e_ao = dmet_energy(d_ao)
            err_ao_mha = (e_ao - e_full) * 1000.0
            row_ao = ('{:<5s}{:>5s}{:>8s}{:>11s}{:>11s}{:>18s}{:>18.10f}{:>13.3f}'.format(
                name + '/AO', '', '', '', '', '', e_ao, err_ao_mha))
            print(row_ao)
            lines.append(row_ao)

        # demonstrate the N2 triple-bond error message
        if nbath_manual is not None:
            try:
                run_dmet(ssdmet.SSDMET, mf, name + '_pb1', imp_label, bath_norb=1)
                msg = f'{name}: bath_norb=1 unexpectedly accepted'
            except ValueError as err:
                msg = f'{name}: bath_norb=1 -> ValueError: {str(err)[:90]}'
            print('   ' + msg)
            lines.append('   ' + msg)

        efull_row = '{:<5s}{:>57s}{:>18.10f}'.format(name, 'E(full RHF, x2c)', e_full)
        print(efull_row)
        lines.append(efull_row)
        print(sep)
        lines.append(sep)
        summary.append((name, nimp, nbath_def, nbath_pb, e_def, e_pb, e_full, err_mha))

    print()
    print('Note: err(1/b) = E_dmet(one-per-bond) - E(full RHF) in mHa;')
    print('with the default tight threshold the embedded cluster equals the full')
    print('molecule for these systems (nbath(def) = all environment orbitals).')

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'one_bath_per_bond_results.txt')
    with open(outfile, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
        fh.write('\n# machine-readable summary: name nimp nbath_def nbath_pb '
                 'E_def E_pb E_full err_pb_mHa\n')
        for row in summary:
            fh.write('%s %d %d %d %.10f %.10f %.10f %.6f\n' % row)
    print(f'\nresults written to {outfile}')


if __name__ == '__main__':
    main()
