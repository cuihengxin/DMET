'''
Ethane C-C bond-stretch PES test for the "one bath orbital per bond" bath
selection, following a two-stage validation workflow:

Stage 1 - HF-in-HF exactness check:
    E(DMET, embedded HF) + E(frozen occ)  vs  E(full HF).
    If the embedding is exact (full bath), the deviation is ~1e-12.  A
    truncated one-orbital-per-bond bath must FAIL this check, and the size of
    the deviation quantifies the embedding error of the bath selection.

Stage 2 - MP2-in-HF energy differences:
    E(DMET, embedded MP2) + E(frozen occ)  vs  E(all-electron MP2),
    compared as PES energy differences dE(R) = E(R) - E(R_eq) along the
    stretched C-C coordinate.  This is the DMET-vs-all-electron comparison.

Methods tested along R = 1.3 ... 3.0 A (6-31G, x2c RHF reference):
  1. SSDMET, default threshold bath   (embeds the whole molecule: sanity);
  2. SSDMET, impurity C0,  fixed 4 bath orbitals (3 C-H + 1 C-C bonds);
  3. SSDMET, impurity CH3 group, fixed 1 bath orbital (the C-C bond);
  4. AODMET, impurity C0,  fixed 4 bath orbitals.

The bath count is kept fixed (chosen at equilibrium) along the scan, since the
automatic 'per_bond' mode stops counting the stretched C-C pair as a bond
beyond ~1.98 A and raises a ValueError.

Run from the repository root (or with PYTHONPATH pointing at it):

    python examples/test_example/one_bath_per_bond_ethane_pes.py
'''

import os
import tempfile
import numpy as np
from pyscf import gto, scf, mp
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


def dmet_hf_energy(dmet):
    """HF-in-HF DMET total energy."""
    return dmet.es_mf.e_tot + dmet.fo_ene


def dmet_mp2_energy(dmet):
    """MP2-in-HF DMET total energy = embedded MP2 + frozen occupied energy."""
    from pyscf import mp as _mp
    if dmet.es_mf.mol.spin != 0:
        mymp2 = _mp.UMP2(dmet.es_mf)
    else:
        mymp2 = _mp.MP2(dmet.es_mf)
    mymp2.verbose = 0
    mymp2.max_memory = dmet.max_mem
    mymp2.kernel()
    return mymp2.e_tot + dmet.fo_ene


def run_dmet(cls, mf, title, imp_label, **kwargs):
    d = cls(mf, title=title, imp_idx=imp_label, verbose=0, **kwargs)
    d.build(save_chk=False)
    return d


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    imp_c = '0 C.*'
    imp_ch3 = ['0 C.*', '2 H.*', '3 H.*', '4 H.*']

    e_hf = {}    # method -> HF energy array
    e_mp2 = {}   # method -> MP2 energy array
    nbath = {}
    auto_bonds = []

    for R in R_GRID:
        mol = ethane_mol(R)
        mf = scf.RHF(mol).x2c()
        mf.verbose = 0
        mf.run()
        auto_bonds.append(count_imp_env_bonds(
            mol, gto.mole._aolabels2baslst(mol, imp_c, base=0)))

        e_hf.setdefault('full', []).append(mf.e_tot)
        mymp2 = mp.MP2(mf)
        mymp2.verbose = 0
        mymp2.kernel()
        e_mp2.setdefault('full', []).append(mymp2.e_tot)

        specs = [('default', ssdmet.SSDMET, imp_c, {}),
                 ('C0/4', ssdmet.SSDMET, imp_c, {'bath_norb': 4}),
                 ('CH3/1', ssdmet.SSDMET, imp_ch3, {'bath_norb': 1}),
                 ('AO-C0/4', aodmet.AODMET, imp_c, {'bath_norb': 4})]
        for tag, cls, imp, kw in specs:
            d = run_dmet(cls, mf, 'e_' + tag.replace('/', '_'), imp, **kw)
            e_hf.setdefault(tag, []).append(dmet_hf_energy(d))
            e_mp2.setdefault(tag, []).append(dmet_mp2_energy(d))
            nbath.setdefault(tag, []).append(d.nes - len(d.imp_idx))

        print('R = %5.2f A   E_full(HF) = %.10f   E_full(MP2) = %.10f'
              % (R, mf.e_tot, e_mp2['full'][-1]), flush=True)

    for tag in e_hf:
        e_hf[tag] = np.array(e_hf[tag])
        e_mp2[tag] = np.array(e_mp2[tag])
    nbath = {k: np.array(v) for k, v in nbath.items()}
    auto_bonds = np.array(auto_bonds)
    ieq = int(np.argmin(e_hf['full']))

    lines = []
    print()
    print('=' * 96)
    print('Stage 1: HF-in-HF exactness check   dE = E(DMET,HF-in-HF) - E(full HF)')
    print('=' * 96)
    h1 = ('{:<8s}{:>9s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(
        'R/A', 'autoNb', 'dE(def)/mHa', 'dE(C0/4)/mHa', 'dE(CH3/1)/mHa',
        'dE(AO)/mHa'))
    print(h1)
    lines += ['=' * 96, 'Stage 1: HF-in-HF exactness check   '
                        'dE = E(DMET,HF-in-HF) - E(full HF)', '=' * 96, h1]
    for k, R in enumerate(R_GRID):
        row = ('{:<8.2f}{:>9d}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}'.format(
            R, auto_bonds[k],
            (e_hf['default'][k] - e_hf['full'][k]) * 1000,
            (e_hf['C0/4'][k] - e_hf['full'][k]) * 1000,
            (e_hf['CH3/1'][k] - e_hf['full'][k]) * 1000,
            (e_hf['AO-C0/4'][k] - e_hf['full'][k]) * 1000))
        print(row)
        lines.append(row)
    print('-' * 96)
    lines.append('-' * 96)
    for tag in ('default', 'C0/4', 'CH3/1', 'AO-C0/4'):
        dev = (e_hf[tag] - e_hf['full']) * 1000
        note = '  [exact HF-in-HF]' if np.max(np.abs(dev)) < 1e-6 else ''
        msg = ('HF-in-HF %-8s: max|dE| = %10.4f mHa (at R_eq: %8.4f mHa)%s'
               % (tag, np.max(np.abs(dev)), dev[ieq], note))
        print(msg)
        lines.append(msg)

    print()
    print('=' * 96)
    print('Stage 2: MP2-in-HF PES   dE_MP2(R) = E(R) - E(R_eq), R_eq = %.2f A'
          % R_GRID[ieq])
    print('=' * 96)
    dE_mp2_full = e_mp2['full'] - e_mp2['full'][ieq]
    h2 = ('{:<8s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}'.format(
        'R/A', 'dE(MP2-full)', 'dE(def)', 'dE(C0/4)', 'dE(CH3/1)', 'dE(AO)'))
    print(h2)
    lines += ['=' * 96, 'Stage 2: MP2-in-HF PES   dE_MP2(R) = E(R) - E(R_eq), '
                        'R_eq = %.2f A' % R_GRID[ieq], '=' * 96, h2]
    for k, R in enumerate(R_GRID):
        row = ('{:<8.2f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}'.format(
            R, dE_mp2_full[k],
            e_mp2['default'][k] - e_mp2['default'][ieq],
            e_mp2['C0/4'][k] - e_mp2['C0/4'][ieq],
            e_mp2['CH3/1'][k] - e_mp2['CH3/1'][ieq],
            e_mp2['AO-C0/4'][k] - e_mp2['AO-C0/4'][ieq]))
        print(row)
        lines.append(row)
    print('-' * 96)
    lines.append('-' * 96)

    print()
    print('MP2 PES error vs all-electron MP2:  dE(method) - dE(MP2-full) / mHa')
    print('-' * 96)
    lines.append('MP2 PES error vs all-electron MP2:  dE(method) - dE(MP2-full) / mHa')
    lines.append('-' * 96)
    pes_err = {}
    for tag in ('default', 'C0/4', 'CH3/1', 'AO-C0/4'):
        err = (e_mp2[tag] - e_mp2[tag][ieq] - dE_mp2_full) * 1000
        pes_err[tag] = err
        msg = ('%-8s max|dE| = %8.2f mHa, RMS = %8.2f mHa'
               % (tag, np.max(np.abs(err)), np.sqrt(np.mean(err ** 2))))
        print(msg)
        lines.append(msg)

    lines.append('')
    lines.append('# eq index: R = %.2f A (minimum of full RHF)' % R_GRID[ieq])
    lines.append('# autoNb: covalent-radius bond count of impurity C0 '
                 '(drops to 3 when C-C > ~1.98 A; automatic per_bond then raises)')
    for tag in ('full', 'default', 'C0/4', 'CH3/1', 'AO-C0/4'):
        lines.append('# %-8s E(HF) : %s' % (tag, ' '.join('%.10f' % x for x in e_hf[tag])))
        lines.append('# %-8s E(MP2): %s' % (tag, ' '.join('%.10f' % x for x in e_mp2[tag])))
        if tag in nbath:
            lines.append('# %-8s nbath : %s' % (tag, ' '.join(str(x) for x in nbath[tag])))

    with open(os.path.join(outdir, 'one_bath_per_bond_ethane_pes_results.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    if HAS_MPL:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        styles = [('default', 'k--', 'DMET default'),
                  ('C0/4', 'ro-', 'DMET C0, 4 bath'),
                  ('CH3/1', 'bs-', 'DMET CH3, 1 bath'),
                  ('AO-C0/4', 'g^-', 'AODMET C0, 4 bath')]
        ax = axs[0, 0]
        ax.axhline(0, color='k', lw=0.6)
        for tag, fmt, lab in styles:
            ax.plot(R_GRID, (e_hf[tag] - e_hf['full']) * 1000, fmt, label=lab)
        ax.set_xlabel('R(C-C) / Angstrom')
        ax.set_ylabel('mHa')
        ax.set_title('Stage 1: HF-in-HF exactness (dE vs full HF)')
        ax.legend(fontsize=8)

        ax = axs[0, 1]
        ax.plot(R_GRID, e_hf['full'] - e_hf['full'][ieq], 'k-', label='full RHF')
        for tag, fmt, lab in styles:
            ax.plot(R_GRID, e_hf[tag] - e_hf[tag][ieq], fmt, label=lab)
        ax.set_xlabel('R(C-C) / Angstrom')
        ax.set_ylabel('dE / Hartree')
        ax.set_title('HF PES (sanity)')
        ax.legend(fontsize=8)

        ax = axs[1, 0]
        ax.plot(R_GRID, dE_mp2_full, 'k-', label='full MP2')
        for tag, fmt, lab in styles:
            ax.plot(R_GRID, e_mp2[tag] - e_mp2[tag][ieq], fmt, label=lab)
        ax.set_xlabel('R(C-C) / Angstrom')
        ax.set_ylabel('dE / Hartree')
        ax.set_title('Stage 2: MP2-in-HF PES')
        ax.legend(fontsize=8)

        ax = axs[1, 1]
        ax.axhline(0, color='k', lw=0.6)
        for tag, fmt, lab in styles:
            ax.plot(R_GRID, pes_err[tag], fmt, label=lab)
        ax.set_xlabel('R(C-C) / Angstrom')
        ax.set_ylabel('mHa')
        ax.set_title('MP2 PES error vs all-electron MP2')
        ax.legend(fontsize=8)

        fig.tight_layout()
        png = os.path.join(outdir, 'one_bath_per_bond_ethane_pes.png')
        fig.savefig(png, dpi=150)
        print('\nplot saved to', png)


if __name__ == '__main__':
    main()
