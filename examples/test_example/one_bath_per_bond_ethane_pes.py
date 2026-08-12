'''
Ethane C-C bond-stretch PES test for bath-orbital selection, following a
two-stage validation workflow with an explicit exactness gate:

Stage 1 - HF-in-HF exactness check:
    |E(DMET, embedded HF) + E(frozen occ) - E(full HF)| must be below a
    tolerance before any correlated calculation is performed.
    * default (threshold) bath   -> exact by construction (whole molecule);
    * one-bath-per-bond (truncated) -> NOT exact; deviation = embedding error;
    * grow-to-exact: start from one-bath-per-bond and add concentric shells
      (virtual + occupied, `embed_sim.exact_bath.grow_bath_to_exact`) until
      the exact condition holds.  This is the "impurity -> add orbitals"
      strategy implemented with `concentric_loc`.

Stage 2 - MP2-in-HF energy differences:
    E(DMET, embedded MP2) + E(frozen occ) vs E(all-electron MP2), compared as
    PES energy differences dE(R) = E(R) - E(R_eq).  Two gates are compared:
    * grow-hf : stop when the HF-in-HF exact condition holds (near-empty
      virtuals may remain frozen; their MP2 correlation is then part of the
      DMET-vs-full gap);
    * grow-full: additionally grow until no virtual stays frozen (embedded
      space = whole molecule), so MP2-in-HF reproduces all-electron MP2.

Methods along R = 1.3 ... 3.0 A (6-31G, x2c RHF):
  1. SSDMET, default threshold bath;
  2. SSDMET, impurity C0,  fixed 4 bath orbitals (one per bond);
  3. SSDMET, impurity CH3 group, fixed 1 bath orbital (the C-C bond);
  4. AODMET, impurity C0,  fixed 4 bath orbitals;
  5. grow-hf / grow-full starting from C0/4.

Run from the repository root (or with PYTHONPATH pointing at it):

    python examples/test_example/one_bath_per_bond_ethane_pes.py
'''

import io
import os
import tempfile
import contextlib
import numpy as np
from pyscf import gto, scf, mp
from embed_sim import ssdmet, aodmet
from embed_sim.bath_selection import count_imp_env_bonds
from embed_sim.exact_bath import grow_bath_to_exact

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
    return dmet.es_mf.e_tot + dmet.fo_ene


def dmet_mp2_energy(dmet):
    if dmet.es_mf.mol.spin != 0:
        mymp2 = mp.UMP2(dmet.es_mf)
    else:
        mymp2 = mp.MP2(dmet.es_mf)
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

    e_hf = {}
    e_mp2 = {}
    nbath = {}
    auto_bonds = []
    grow_summary = {'grow-hf': [], 'grow-full': []}

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

        for gate in ('grow-hf', 'grow-full'):
            with contextlib.redirect_stdout(io.StringIO()):
                d = run_dmet(ssdmet.SSDMET, mf, 'e_' + gate, imp_c,
                             bath_norb=4, es_natorb=False)
                d0 = dmet_hf_energy(d) - mf.e_tot
                d, hist = grow_bath_to_exact(
                    d, tol=1e-6,
                    include_all_virtuals=(gate == 'grow-full'))
            e_hf.setdefault(gate, []).append(dmet_hf_energy(d))
            e_mp2.setdefault(gate, []).append(dmet_mp2_energy(d))
            nbath.setdefault(gate, []).append(d.nes - len(d.imp_idx))
            grow_summary[gate].append(dict(
                dev0_mha=d0 * 1000, rounds=len(hist),
                add_vir=sum(h['add_vir'] for h in hist),
                add_occ=sum(h['add_occ'] for h in hist),
                nbath=d.nes - len(d.imp_idx), nfo=d.nfo, nfv=d.nfv,
                dev_final=hf_in_hf(d)))

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
    print('=' * 100)
    print('Stage 1: HF-in-HF exactness check   dE = E(DMET,HF-in-HF) - E(full HF)')
    print('=' * 100)
    h1 = ('{:<8s}{:>9s}{:>12s}{:>12s}{:>12s}{:>12s}'.format(
        'R/A', 'autoNb', 'dE(def)/mHa', 'dE(C0/4)/mHa', 'dE(CH3/1)/mHa',
        'dE(AO)/mHa'))
    print(h1)
    lines += ['=' * 100, 'Stage 1: HF-in-HF exactness check   '
                        'dE = E(DMET,HF-in-HF) - E(full HF)', '=' * 100, h1]
    for k, R in enumerate(R_GRID):
        row = ('{:<8.2f}{:>9d}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}'.format(
            R, auto_bonds[k],
            (e_hf['default'][k] - e_hf['full'][k]) * 1000,
            (e_hf['C0/4'][k] - e_hf['full'][k]) * 1000,
            (e_hf['CH3/1'][k] - e_hf['full'][k]) * 1000,
            (e_hf['AO-C0/4'][k] - e_hf['full'][k]) * 1000))
        print(row)
        lines.append(row)
    print('-' * 100)
    lines.append('-' * 100)
    for tag in ('default', 'C0/4', 'CH3/1', 'AO-C0/4'):
        dev = (e_hf[tag] - e_hf['full']) * 1000
        note = '  [exact HF-in-HF]' if np.max(np.abs(dev)) < 1e-6 else ''
        msg = ('HF-in-HF %-8s: max|dE| = %10.4f mHa (at R_eq: %8.4f mHa)%s'
               % (tag, np.max(np.abs(dev)), dev[ieq], note))
        print(msg)
        lines.append(msg)

    print()
    print('Grow-to-exact summary (start: C0 impurity, 4 bath orbitals):')
    print('-' * 100)
    lines.append('')
    lines.append('Grow-to-exact summary (start: C0 impurity, 4 bath orbitals):')
    lines.append('-' * 100)
    hg = ('{:<8s}{:>10s}{:>11s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}'.format(
        'gate', 'R/A', 'dev0/mHa', 'rounds', 'addVir', 'addOcc', 'nbath',
        'devF/mHa'))
    print(hg)
    lines.append(hg)
    for gate in ('grow-hf', 'grow-full'):
        for k, R in enumerate(R_GRID):
            s = grow_summary[gate][k]
            row = ('{:<8s}{:>10.2f}{:>11.3f}{:>10d}{:>10d}{:>10d}{:>10d}{:>10.6f}'.format(
                gate, R, s['dev0_mha'], s['rounds'], s['add_vir'],
                s['add_occ'], s['nbath'], s['dev_final'] * 1000))
            print(row)
            lines.append(row)
    print('-' * 100)
    lines.append('-' * 100)

    print()
    print('=' * 100)
    print('Stage 2: MP2-in-HF PES   dE_MP2(R) = E(R) - E(R_eq), R_eq = %.2f A'
          % R_GRID[ieq])
    print('=' * 100)
    dE_mp2_full = e_mp2['full'] - e_mp2['full'][ieq]
    tags2 = ('default', 'C0/4', 'CH3/1', 'grow-hf', 'grow-full')
    h2 = ('{:<8s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}'.format(
        'R/A', 'dE(MP2-full)', 'dE(def)', 'dE(C0/4)', 'dE(CH3/1)',
        'dE(grow-hf)', 'dE(grow-full)'))
    print(h2)
    lines += ['=' * 100, 'Stage 2: MP2-in-HF PES   dE_MP2(R) = E(R) - E(R_eq), '
                        'R_eq = %.2f A' % R_GRID[ieq], '=' * 100, h2]
    for k, R in enumerate(R_GRID):
        row = ('{:<8.2f}{:>13.6f}'.format(R, dE_mp2_full[k]))
        for tag in tags2:
            row += '{:>13.6f}'.format(e_mp2[tag][k] - e_mp2[tag][ieq])
        print(row)
        lines.append(row)
    print('-' * 100)
    lines.append('-' * 100)

    print()
    print('MP2 PES error vs all-electron MP2:  dE(method) - dE(MP2-full) / mHa')
    print('-' * 100)
    lines.append('MP2 PES error vs all-electron MP2:  dE(method) - dE(MP2-full) / mHa')
    lines.append('-' * 100)
    pes_err = {}
    for tag in tags2:
        err = (e_mp2[tag] - e_mp2[tag][ieq] - dE_mp2_full) * 1000
        pes_err[tag] = err
        msg = ('%-8s max|dE| = %8.2f mHa, RMS = %8.2f mHa'
               % (tag, np.max(np.abs(err)), np.sqrt(np.mean(err ** 2))))
        print(msg)
        lines.append(msg)

    lines.append('')
    lines.append('# eq index: R = %.2f A (minimum of full RHF)' % R_GRID[ieq])
    lines.append('# autoNb: covalent-radius bond count of impurity C0 '
                 '(drops to 3 when C-C > ~1.98 A)')
    for tag in ('full',) + tags2:
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
                  ('grow-hf', 'm^-', 'grow-to-exact (HF gate)'),
                  ('grow-full', 'cD-', 'grow-to-exact (full space)')]
        ax = axs[0, 0]
        ax.axhline(0, color='k', lw=0.6)
        for tag, fmt, lab in styles[:4]:
            ax.plot(R_GRID, (e_hf[tag] - e_hf['full']) * 1000, fmt, label=lab)
        ax.set_xlabel('R(C-C) / Angstrom')
        ax.set_ylabel('mHa')
        ax.set_title('Stage 1: HF-in-HF exactness (dE vs full HF)')
        ax.legend(fontsize=8)

        ax = axs[0, 1]
        ax.plot(R_GRID, e_hf['full'] - e_hf['full'][ieq], 'k-', label='full RHF')
        for tag, fmt, lab in styles[:4]:
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


def hf_in_hf(dmet):
    return abs(dmet.es_mf.e_tot + dmet.fo_ene - dmet.mf_or_cas.e_tot)


if __name__ == '__main__':
    main()
