#!/usr/bin/env python
'''
Test / benchmark for the pair-resolved (PNO) bath, `embed_sim.pno_bath`.

    python examples/test_example/pno_bath_test.py

Design notes accumulated while testing
--------------------------------------
* A small impurity inside a big environment is required, otherwise there is
  nothing to select.  (H2O/6-31G with imp = O leaves nfv = 1.)
* `bath_norb` must be set, otherwise the default threshold of 1e-12 pulls every
  environment occupied orbital into the bath and nfo = 0, so the occupied side
  of the expansion is never exercised.
* Stage 3 measures the *total* correlation energy at fixed bath size.  On that
  metric the occupation ranking is expected to win: truncating MP2 natural
  orbitals by occupation is the Eckart-Young optimal low-rank truncation of the
  amplitude, so any reweighting can only move away from the optimum.  Stage 5 is
  the metric the pair-energy criterion is actually aimed at -- the *change* of
  the correlation energy between two structures (Bensberg & Neugebauer 2022).

Stages
------
1. structural sanity   orthonormality, completeness, DMET exactness
2. convergence         MP2-in-HF correlation energy vs. embedded space size
3. total energy        three rankings at equal embedded space size
                         occ          bare PNO occupation; in fixed-size mode
                                      this exactly reproduces global BNO-like
                                      natural orbitals
                         energy       exact share of the pair energy per PNO
                         occ_x_pair   occupation * pair energy (double counts
                                      the pair strength; kept as a control)
4. energy target       `e_target` mode, threshold in Hartree
5. energy DIFFERENCE   the same three rankings judged on a C-C stretch, which
                       is what the pair-energy criterion is meant to improve
'''

import contextlib
import io
import os
import sys

import numpy as np
from pyscf import gto, scf, mp

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from embed_sim import ssdmet                                    # noqa: E402
from embed_sim.ssdmet_pno import (expand_bath_pno,              # noqa: E402
                                  check_exactness)

HARTREE2KCAL = 627.5094740631


def ethane_geom(dcc):
    '''Rigid CH3 groups, only the C-C distance changes.'''
    z, hz = dcc / 2, 0.397
    top = [(0.0, 1.017), (-0.881, -0.508), (0.881, -0.508)]
    out = [f'C 0 0 {z}', f'C 0 0 {-z}']
    out += [f'H {x} {y} {z + hz}' for x, y in top]
    out += [f'H {x} {-y} {-z - hz}' for x, y in top]
    return '; '.join(out)


SYSTEMS = {
    'ethane': dict(atom=ethane_geom(1.532), basis='cc-pVDZ', imp='0 C',
                   bath_norb=4),
    'water': dict(atom='O 0 0 0.117; H 0 0.757 -0.470; H 0 -0.757 -0.470',
                  basis='cc-pVTZ', imp='0 O', bath_norb=2),
}


@contextlib.contextmanager
def quiet():
    '''SSDMET.dump_flags writes to stdout regardless of verbose.'''
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_dmet(mf, imp, bath_norb):
    '''Build plain DMET; widen bath_norb until the env partition is valid.'''
    last = None
    for extra in range(0, 12):
        try:
            with quiet():
                d = ssdmet.SSDMET(mf, title='pno_tmp', imp_idx=imp,
                                  es_natorb=False, bath_norb=bath_norb + extra,
                                  verbose=0)
                d.build(chk_fname_load=None, save_chk=False)
            return d, bath_norb + extra
        except ValueError as err:          # partition_env_by_bath_count
            last = err
    raise RuntimeError(f'no valid bath_norb found: {last}')


def expand(d, **kw):
    with quiet():
        return expand_bath_pno(d, verbose=0, **kw)


def dmet_mp2(d):
    with quiet():
        m = mp.RMP2(d.es_mf)
        m.verbose = 0
        m.kernel()
    return m.e_corr


# ---------------------------------------------------------------- stage 1
def stage1_sanity(mf, imp, bath_norb):
    print('\n[1] structural sanity')
    d, used = make_dmet(mf, imp, bath_norb)
    base = check_exactness(d)
    print(f'    plain DMET (bath_norb={used})')
    print(f'      nes={d.nes:3d} nfo={d.nfo:3d} nfv={d.nfv:3d}  '
          f'orth={base["orthonormality"]:.1e}  dev={base["exactness_deviation"]:.1e}')
    ok = base['passed']
    if d.nfo == 0:
        print('      WARNING: nfo = 0, occupied side not exercised')
    if d.nfv < 5:
        print('      WARNING: nfv < 5, too little to select from')

    for t in (1e-4, 1e-6, 1e-8):
        d, _ = make_dmet(mf, imp, bath_norb)
        expand(d, t_pno=t, t_pair=1e-6)
        chk = check_exactness(d)
        flag = 'ok ' if chk['passed'] else 'FAIL'
        print(f'    t_pno={t:.0e} : nes={d.nes:3d} nfo={d.nfo:3d} nfv={d.nfv:3d}  '
              f'orth={chk["orthonormality"]:.1e}  dev={chk["exactness_deviation"]:.1e}  '
              f'missing={chk["missing_orbitals"]}  [{flag}]')
        ok = ok and chk['passed']
    return ok, used


# ---------------------------------------------------------------- stage 2
def stage2_convergence(mol, mf, imp, bath_norb, e_full):
    print(f'\n[2] MP2-in-HF convergence   (full MP2 Ecorr = {e_full:.8f})')
    print('    scheme          nes  %AO   nfo  nfv     Ecorr(emb)    error   recovered')
    d, _ = make_dmet(mf, imp, bath_norb)
    e0 = dmet_mp2(d)
    print(f'    plain bath    {d.nes:5d} {100*d.nes/mol.nao:5.1f} {d.nfo:5d}{d.nfv:5d} '
          f'{e0:14.8f} {e0-e_full:9.2e} {100*e0/e_full:8.2f}%')
    for t in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        d, _ = make_dmet(mf, imp, bath_norb)
        expand(d, t_pno=t, t_pair=1e-7)
        e = dmet_mp2(d)
        print(f'    PNO t={t:.0e}  {d.nes:5d} {100*d.nes/mol.nao:5.1f} {d.nfo:5d}{d.nfv:5d} '
              f'{e:14.8f} {e-e_full:9.2e} {100*e/e_full:8.2f}%')


# ---------------------------------------------------------------- stage 3
SCORES = ('occ', 'energy', 'occ_x_pair')


def stage3_total_energy(mf, imp, bath_norb, e_full):
    print('\n[3] equal-size comparison, TOTAL energy (more negative = better)')
    d0, _ = make_dmet(mf, imp, bath_norb)
    nfo0, nfv0 = d0.nfo, d0.nfv
    print('    add(o,v)  nes  ' + ''.join(f'{s:>16s}' for s in SCORES) + '   best')
    for nadd in (2, 4, 8, 12, 16):
        n_v, n_o = min(nadd, nfv0), min(nadd, nfo0)
        if n_v == 0 and n_o == 0:
            continue
        res, nes = {}, None
        for s in SCORES:
            d, _ = make_dmet(mf, imp, bath_norb)
            expand(d, nbath_vir=n_v, nbath_occ=n_o, t_pair=0.0, t_pno=0.0,
                   score=s)
            res[s], nes = dmet_mp2(d), d.nes
        best = min(res, key=lambda k: res[k])
        print(f'      ({n_o:2d},{n_v:2d}) {nes:5d}'
              + ''.join(f'{res[s]:16.8f}' for s in SCORES) + f'   {best}')
    print(f'    (plain nes={d0.nes}, nfo={nfo0}, nfv={nfv0}; '
          f'full MP2 Ecorr={e_full:.8f})')


# ---------------------------------------------------------------- stage 4
def stage4_energy_target(mf, imp, bath_norb, e_full):
    print('\n[4] energy-targeted threshold  (discarded is a semi-canonical '
          'MP2 estimate, not an error bar)')
    print('    target(Ha)  nes   discarded(occ) discarded(vir)   Ecorr(emb)    error')
    for e_target in (1e-2, 1e-3, 1e-4):
        d, _ = make_dmet(mf, imp, bath_norb)
        info = expand(d, e_target=e_target, t_pair=1e-9)
        e = dmet_mp2(d)
        do = info['occ'].get('e_discarded', 0.0)
        dv = info['vir'].get('e_discarded', 0.0)
        print(f'    {e_target:.0e}    {d.nes:5d}   {do:13.3e}  {dv:13.3e}  '
              f'{e:13.8f} {e-e_full:9.2e}')


# ---------------------------------------------------------------- stage 5
def stage5_energy_difference(spec, d_ref=1.532, d_str=2.20):
    '''The metric the pair-energy criterion is actually aimed at.

    HF is exact in DMET (E(es_mf) + E(frozen) == E(mf)), so the entire error of
    the reaction energy sits in the correlation part.  We therefore compare
    d(Ecorr) between the two structures against full MP2.
    '''
    print(f'\n[5] ENERGY DIFFERENCE, C-C {d_ref} -> {d_str} Ang '
          '(equal bath size, error in kcal/mol)')
    geoms = {}
    for tag, dcc in (('ref', d_ref), ('str', d_str)):
        mol = gto.M(atom=ethane_geom(dcc), basis=spec['basis'], verbose=0)
        mf = scf.RHF(mol).run()
        geoms[tag] = (mol, mf, mp.MP2(mf).run().e_corr)
    dE_full = geoms['str'][2] - geoms['ref'][2]
    print(f'    full MP2 d(Ecorr) = {dE_full:.8f} Ha = '
          f'{dE_full*HARTREE2KCAL:.4f} kcal/mol')

    # plain bath baseline
    base = {}
    for tag in ('ref', 'str'):
        d, _ = make_dmet(geoms[tag][1], spec['imp'], spec['bath_norb'])
        base[tag] = dmet_mp2(d)
    err0 = (base['str'] - base['ref'] - dE_full) * HARTREE2KCAL
    print(f'    plain bath                      error = {err0:9.4f} kcal/mol')

    print('    add(o,v)  ' + ''.join(f'{s:>14s}' for s in SCORES) + '    best')
    for nadd in (2, 4, 8, 12):
        errs = {}
        sizes = None
        for s in SCORES:
            e = {}
            nes = []
            for tag in ('ref', 'str'):
                d, _ = make_dmet(geoms[tag][1], spec['imp'], spec['bath_norb'])
                n_v, n_o = min(nadd, d.nfv), min(nadd, d.nfo)
                expand(d, nbath_vir=n_v, nbath_occ=n_o, t_pair=0.0, t_pno=0.0,
                       score=s)
                e[tag] = dmet_mp2(d)
                nes.append(d.nes)
            errs[s] = (e['str'] - e['ref'] - dE_full) * HARTREE2KCAL
            sizes = nes
        best = min(errs, key=lambda k: abs(errs[k]))
        print(f'      ({nadd:2d})  nes={sizes[0]}/{sizes[1]}  '
              + ''.join(f'{errs[s]:14.4f}' for s in SCORES) + f'   {best}')


def main():
    ok = True
    for name, spec in SYSTEMS.items():
        mol = gto.M(atom=spec['atom'], basis=spec['basis'], verbose=0)
        mf = scf.RHF(mol).run()
        e_full = mp.MP2(mf).run().e_corr
        print('=' * 84)
        print(f"{name} / {spec['basis']} / imp={spec['imp']} / nao={mol.nao}")
        print('=' * 84)
        good, used = stage1_sanity(mf, spec['imp'], spec['bath_norb'])
        ok &= good
        stage2_convergence(mol, mf, spec['imp'], used, e_full)
        stage3_total_energy(mf, spec['imp'], used, e_full)
        stage4_energy_target(mf, spec['imp'], used, e_full)

    print('=' * 84)
    print('reaction-energy test (ethane C-C stretch)')
    print('=' * 84)
    stage5_energy_difference(SYSTEMS['ethane'])

    print('\n' + '=' * 84)
    print('structural sanity:', 'PASSED' if ok else 'FAILED')
    print('=' * 84)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
