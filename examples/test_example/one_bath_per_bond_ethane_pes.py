'''
Ethane C-C bond-stretch potential energy surface (PES) test for the
"one bath orbital per bond" bath selection.

Motivation: removing bath orbitals shifts the *absolute* DMET energy by tens
to hundreds of mHa (see one_bath_per_bond.py).  What usually matters in
practice is the *relative* energy along a geometry scan (a PES).  This script
therefore compares, at a series of stretched C-C distances R, the energy
difference dE(R) = E(R) - E(R_eq) of:

  1. full-system RHF (x2c)          -> reference;
  2. SSDMET, default threshold bath  -> embeds the whole molecule (sanity);
  3. SSDMET, impurity C0, fixed 4 bath orbitals (3 C-H + 1 C-C bonds);
  4. SSDMET, impurity CH3 group, fixed 1 bath orbital (the C-C bond),
     i.e. the SN2-style edge-group setup;
  5. AODMET, impurity C0, fixed 4 bath orbitals.

The bath count is kept FIXED along the scan (as chosen at the equilibrium
geometry) so that the comparison isolates the bath-selection error rather
than the changing bond count.  Note that the automatic 'per_bond' mode
counts the stretched C-C pair as non-bonded beyond ~1.98 A and then raises a
ValueError (the leftover environment orbital is too entangled to freeze);
that failure mode is also demonstrated below.

Run from the repository root (or with PYTHONPATH pointing at it):

    python examples/test_example/one_bath_per_bond_ethane_pes.py
'''

import os
import tempfile
import numpy as np
from pyscf import gto, scf
from embed_sim import ssdmet, aodmet
from embed_sim.bath_selection import count_imp_env_bonds

try:
    # keep matplotlib/fontconfig caches away from a possibly non-writable HOME
    _plot_home = os.path.join(tempfile.gettempdir(), 'eth_pes_home')
    os.makedirs(_plot_home, exist_ok=True)
    os.environ['HOME'] = _plot_home
    os.environ['MPLCONFIGDIR'] = os.path.join(_plot_home, '.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASIS = '6-31g'
R_GRID = np.array([1.30, 1.40, 1.50, 1.54, 1.60, 1.70, 1.90,
                   2.10, 2.30, 2.60, 3.00])


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
    return gto.M(atom=atom_str, basis=basis, verbose=0)


def dmet_energy(dmet):
    return dmet.es_mf.e_tot + dmet.fo_ene


def run_dmet(cls, mf, title, imp_label, **kwargs):
    d = cls(mf, title=title, imp_idx=imp_label, verbose=0, **kwargs)
    d.build(save_chk=False)
    return d


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    results = {}   # method -> array of E(R)
    nbath = {}     # method -> array of bath counts
    auto_bonds = []
    imp_c = '0 C.*'
    imp_ch3 = ['0 C.*', '2 H.*', '3 H.*', '4 H.*']

    for R in R_GRID:
        mol = ethane_mol(R)
        mf = scf.RHF(mol).x2c()
        mf.verbose = 0
        mf.run()
        e_full = mf.e_tot
        auto_bonds.append(count_imp_env_bonds(
            mol, gto.mole._aolabels2baslst(mol, imp_c, base=0)))

        # default threshold-based bath (whole molecule for these systems)
        d_def = run_dmet(ssdmet.SSDMET, mf, 'e_def', imp_c)
        # fixed 4 bath orbitals, impurity = one carbon atom
        d_c0 = run_dmet(ssdmet.SSDMET, mf, 'e_c0', imp_c, bath_norb=4)
        # fixed 1 bath orbital (the C-C bond), impurity = CH3 group
        d_ch3 = run_dmet(ssdmet.SSDMET, mf, 'e_ch3', imp_ch3, bath_norb=1)
        # AODMET variant, impurity = one carbon atom
        d_ao = run_dmet(aodmet.AODMET, mf, 'e_ao', imp_c, bath_norb=4)

        for tag, d in (('full', None), ('default', d_def), ('C0/4', d_c0),
                       ('CH3/1', d_ch3), ('AO-C0/4', d_ao)):
            if tag == 'full':
                results.setdefault(tag, []).append(e_full)
                continue
            results.setdefault(tag, []).append(dmet_energy(d))
            nbath.setdefault(tag, []).append(d.nes - len(d.imp_idx))

        print('R = %5.2f A   E_full = %.10f' % (R, e_full), flush=True)

    for tag in results:
        results[tag] = np.array(results[tag])
        if tag in nbath:
            nbath[tag] = np.array(nbath[tag])
    auto_bonds = np.array(auto_bonds)

    # reference energy differences
    ieq = int(np.argmin(results['full']))
    dE_full = results['full'] - results['full'][ieq]

    lines = []
    sep = '-' * 92
    header = ('{:<8s}{:>8s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(
        'R/A', 'autoNb', 'dE_full', 'dE_def', 'dE_C0/4',
        'dE_CH3/1', 'dE_AO', 'err(C0)/mHa'))
    print()
    print(sep)
    print(header)
    print(sep)
    lines += [sep, header, sep]
    for k, R in enumerate(R_GRID):
        err_c0 = (results['C0/4'][k] - results['C0/4'][ieq] - dE_full[k]) * 1000.0
        row = ('{:<8.2f}{:>8d}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.3f}'.format(
            R, auto_bonds[k], dE_full[k],
            results['default'][k] - results['default'][ieq],
            results['C0/4'][k] - results['C0/4'][ieq],
            results['CH3/1'][k] - results['CH3/1'][ieq],
            results['AO-C0/4'][k] - results['AO-C0/4'][ieq],
            err_c0))
        print(row)
        lines.append(row)
    print(sep)
    lines.append(sep)

    # PES errors dE(method) - dE(full), in mHa
    def pes_err(tag):
        return (results[tag] - results[tag][ieq] - dE_full) * 1000.0

    print()
    for tag in ('default', 'C0/4', 'CH3/1', 'AO-C0/4'):
        err = pes_err(tag)
        print('%-8s PES error: max|dE| = %7.2f mHa, RMS = %7.2f mHa'
              % (tag, np.max(np.abs(err)), np.sqrt(np.mean(err**2))))
        lines.append('%-8s PES error: max|dE| = %7.2f mHa, RMS = %7.2f mHa'
                     % (tag, np.max(np.abs(err)), np.sqrt(np.mean(err**2))))
    lines.append('')
    lines.append('# eq index: R = %.2f A (minimum of full RHF)' % R_GRID[ieq])
    lines.append('# autoNb: covalent-radius bond count of impurity C0 '
                 '(drops to 3 when C-C > ~1.98 A; automatic per_bond then raises)')
    for tag in results:
        lines.append('# %s E(R): %s' % (tag, ' '.join('%.10f' % x for x in results[tag])))

    with open(os.path.join(outdir, 'one_bath_per_bond_ethane_pes_results.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        styles = [('full', 'k-', 'full RHF'),
                  ('default', 'k--', 'DMET default'),
                  ('C0/4', 'ro-', 'DMET C0, 4 bath'),
                  ('CH3/1', 'bs-', 'DMET CH3, 1 bath'),
                  ('AO-C0/4', 'g^-', 'AODMET C0, 4 bath')]
        for tag, fmt, lab in styles:
            ax1.plot(R_GRID, results[tag] - results[tag][ieq], fmt, label=lab)
        ax1.set_xlabel('R(C-C) / Angstrom')
        ax1.set_ylabel('dE / Hartree')
        ax1.set_title('Ethane C-C stretch (6-31G, RHF/x2c)')
        ax1.legend(fontsize=8)
        for tag, fmt, lab in styles[1:]:
            ax2.plot(R_GRID, pes_err(tag), fmt, label=lab)
        ax2.axhline(0, color='k', lw=0.6)
        ax2.set_xlabel('R(C-C) / Angstrom')
        ax2.set_ylabel('dE(method) - dE(RHF) / mHa')
        ax2.set_title('PES error relative to full RHF')
        ax2.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(outdir, 'one_bath_per_bond_ethane_pes.png')
        fig.savefig(png, dpi=150)
        print('\nplot saved to', png)


if __name__ == '__main__':
    main()
