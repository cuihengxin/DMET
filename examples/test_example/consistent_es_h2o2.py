"""
Test embed_sim.consistent_es on H2O2.

The point being tested: two DMET calculations that partition their
environment differently end up correlating a different number of electrons in
a different number of orbitals, so their truncation errors do not cancel in an
energy difference.  `match_es_occ_vir` promotes the missing frozen orbitals
back into the embedded space until nfo / nfv / nes agree.

Stage 0   scan `threshold`, tabulate how the ES size responds, pick a pair
          that actually differs.
Stage 1   match that pair; check nfo / nfv / nes / nelec_ES align and that the
          promoted orbitals really correspond to the reference ones.
Stage 2   DMET exactness condition  E(ES mf) + E(fo) - E(total mf).
          Promoting frozen-OCCUPIED orbitals must drive this towards zero;
          promoting frozen-VIRTUAL orbitals leaves it untouched by
          construction (only the frozen-occupied block enters fo_ene).
Stage 3   embedded MP2 before/after matching, against full-electron MP2.
Stage 4   the real use case: two *geometries* (equilibrium and stretched O-O),
          matched through the cross-molecule overlap.  In H2O2 this only ever
          promotes OCCUPIED orbitals, so:
Stage 5   cross-structure promotion of VIRTUAL orbitals only, by giving the two
          geometries different thresholds.  Sharp check: dm_fo is untouched, so
          fo_ene must be bit-for-bit unchanged.
Stage 6   the branch where both objects have to grow.  Auto-skips if the system
          cannot produce such a pair.

Run:  python consistent_es_h2o2.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyscf import gto, scf, mp
from embed_sim import ssdmet
from embed_sim.consistent_es import match_es_occ_vir, dmet_deviation

BASIS = 'cc-pvdz'
IMP = '0 O.*'          # impurity = the first O atom
R_EQ = 1.452           # equilibrium O-O
R_STRETCH = 1.90       # stretched O-O
THRESHOLDS = [1e-12, 1e-8, 1e-5, 1e-3, 1e-2, 3e-2, 1e-1]

HARTREE2KCAL = 627.5094740631


# ------------------------------------------------------------------ helpers

def h2o2_geometry(r_oo=R_EQ):
    """HOOH, r(OH)=0.965 A, a(OOH)=100.0 deg, dihedral(HOOH)=119.1 deg."""
    r_oh, ang, dih = 0.965, np.deg2rad(100.0), np.deg2rad(119.1)
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([r_oo, 0.0, 0.0])
    h1 = o1 + r_oh * np.array([np.cos(ang), np.sin(ang), 0.0])
    # seen from O2 the bond axis points along -x; the perpendicular component
    # is rotated out of the H1-O1-O2 plane by the dihedral angle
    perp = np.array([0.0, np.cos(dih), np.sin(dih)])
    h2 = o2 + r_oh * (np.cos(ang) * np.array([-1.0, 0.0, 0.0]) + np.sin(ang) * perp)
    atoms = [('O', o1), ('O', o2), ('H', h1), ('H', h2)]
    return '\n'.join(f'{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}' for s, c in atoms)


def build_mf(r_oo=R_EQ, label=''):
    mol = gto.Mole()
    mol.atom = h2o2_geometry(r_oo)
    mol.basis = BASIS
    mol.verbose = 3
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    print(f'[{label}] r(OO)={r_oo:.3f} A   nao={mol.nao}   '
          f'nelec={mol.nelectron}   E(RHF)={mf.e_tot:.10f}')
    return mol, mf


def build_dmet(mf, title, threshold):
    """es_natorb=False is required: append_bath_by_env_idx rebuilds es_dm and
    would reuse a stale self.es_occ otherwise."""
    d = ssdmet.SSDMET(mf, title=title, imp_idx=IMP, threshold=threshold,
                      es_natorb=False)
    d.build(save_chk=False)
    return d


def sizes(d):
    """(nes, nfo, nfv, nelec_ES) -- everything that has to match."""
    return (d.nes, d.nfo, d.nfv, d.mol.nelectron - 2 * d.nfo)


def show(tag, d):
    nes, nfo, nfv, nel = sizes(d)
    print(f'  {tag:26s} nes={nes:3d}  nfo={nfo:3d}  nfv={nfv:3d}  '
          f'nelec_ES={nel:3d}  dev={dmet_deviation(d):12.4e}')


def banner(text):
    print('\n' + '=' * 78)
    print(text)
    print('=' * 78)


def scan_thresholds(mf, prefix, thresholds=THRESHOLDS):
    """Build one DMET per threshold; return {threshold: dmet}."""
    out = {}
    for t in thresholds:
        d = build_dmet(mf, f'{prefix}_t{t:g}', t)
        out[t] = d
    return out


def print_scan(table, header):
    print(f'\n{header}')
    print(f'  {"threshold":>12s} {"nbath":>6s} {"nes":>5s} {"nfo":>5s} '
          f'{"nfv":>5s} {"nelec_ES":>9s} {"DMET deviation":>18s}')
    for t, d in table.items():
        nbath = d.nes - len(d.imp_idx)
        nes, nfo, nfv, nel = sizes(d)
        print(f'  {t:12g} {nbath:6d} {nes:5d} {nfo:5d} {nfv:5d} {nel:9d} '
              f'{dmet_deviation(d):18.6e}')


# ================================================================== Stage 0
banner('Stage 0: how the embedded space responds to `threshold`')
mol, mf = build_mf(R_EQ, label='equilibrium')

scan = scan_thresholds(mf, 'h2o2_eq')
print_scan(scan, 'H2O2 equilibrium, impurity = O(0):')

# snapshot the partition BEFORE anything is matched in place -- the objects in
# `scan` are reused below and match_es_occ_vir mutates them
scan_sizes = {t: (d.nfo, d.nfv, d.nes) for t, d in scan.items()}

# reference: untruncated bath, where HF-in-HF embedding is exact
d_exact = scan[THRESHOLDS[0]]
dev_exact = dmet_deviation(d_exact)
assert abs(dev_exact) < 1e-8, (
    f'the untruncated bath (threshold={THRESHOLDS[0]:g}) is NOT exact '
    f'({dev_exact:.6e}). The problem is in the embedding itself, not in the '
    'occ/vir matching -- fix that first.')
print(f'\nOK: untruncated bath satisfies the DMET exactness condition '
      f'({dev_exact:.3e})')

# pick a pair whose (nfo, nfv) actually differ: the largest ES vs the smallest
by_nfo = sorted(scan.items(), key=lambda kv: (kv[1].nfo, kv[1].nfv))
t_big, d_big = by_nfo[0]      # smallest nfo  -> largest ES
t_small, d_small = by_nfo[-1]  # largest nfo  -> smallest ES

assert (d_big.nfo, d_big.nfv) != (d_small.nfo, d_small.nfv), (
    'every threshold in THRESHOLDS gave the same partition; widen the scan')

print(f'\nselected pair:  big ES  threshold={t_big:g}   '
      f'small ES  threshold={t_small:g}')
show(f'big ES   (thr={t_big:g})', d_big)
show(f'small ES (thr={t_small:g})', d_small)

dev_big_before = dmet_deviation(d_big)
dev_small_before = dmet_deviation(d_small)
fo_big_before = d_big.fo_ene
fo_small_before = d_small.fo_ene

# ================================================================== Stage 3a
# (energies must be taken BEFORE the spaces are modified in place)
banner('Stage 3a: reference energies BEFORE matching')
mymp2_full = mp.MP2(mf)
mymp2_full.kernel()
e_full = mymp2_full.e_tot
print(f'full-electron MP2   E_tot = {e_full:.10f}   '
      f'E_corr = {mymp2_full.e_corr:.10f}')

e_big_before, ec_big_before = d_big.mp2_solver()
e_small_before, ec_small_before = d_small.mp2_solver()

# ================================================================== Stage 1
banner('Stage 1: match the two embedded spaces')
res = match_es_occ_vir(d_big, d_small)

print('\nafter matching:')
show('big ES', d_big)
show('small ES', d_small)

assert d_big.nfo == d_small.nfo, f'nfo mismatch: {d_big.nfo} vs {d_small.nfo}'
assert d_big.nfv == d_small.nfv, f'nfv mismatch: {d_big.nfv} vs {d_small.nfv}'
assert d_big.nes == d_small.nes, f'nes mismatch: {d_big.nes} vs {d_small.nes}'
assert (d_big.mol.nelectron - 2 * d_big.nfo) == \
       (d_small.mol.nelectron - 2 * d_small.nfo)
print('OK: nfo, nfv, nes and nelec_ES all match')

score_lists = list(res['score_occ'].values()) + list(res['score_vir'].values())
all_scores = (np.concatenate([np.atleast_1d(v) for v in score_lists])
              if score_lists else np.array([]))
if all_scores.size:
    print(f'promoted-orbital scores: n={all_scores.size}  '
          f'min={all_scores.min():.6f}  max={all_scores.max():.6f}')
    assert all_scores.min() > 0.9, (
        f'a promoted orbital overlaps the reference subspace by only '
        f'{all_scores.min():.4f}; the wrong orbital was picked. '
        f'all scores: {all_scores}')
    print('OK: every promoted orbital overlaps the reference space by > 0.9')
else:
    raise AssertionError('nothing was promoted, yet the sizes differed')

# ================================================================== Stage 2
banner('Stage 2: DMET exactness condition   E(ES mf) + E(fo) - E(total mf)')
dev_big_after = dmet_deviation(d_big)
dev_small_after = dmet_deviation(d_small)

print(f'\n  {"":22s} {"before":>16s} {"after":>16s}')
print(f'  {"big ES":22s} {dev_big_before:16.4e} {dev_big_after:16.4e}')
print(f'  {"small ES":22s} {dev_small_before:16.4e} {dev_small_after:16.4e}')
print(f'  {"full bath (exact)":22s} {"":>16s} {dev_exact:16.4e}')

for tag, d0, d1 in (('big ES', dev_big_before, dev_big_after),
                    ('small ES', dev_small_before, dev_small_after)):
    assert abs(d1) <= abs(d0) + 1e-10, (
        f'{tag}: promoting frozen orbitals made the exactness deviation '
        f'WORSE, {d0:.6e} -> {d1:.6e}')
print('OK: no object moved away from the exactness condition')

# Only the frozen-occupied block enters fo_ene.  An object that gained
# occupied orbitals must improve; one that gained only virtuals must keep
# fo_ene bit-for-bit (its ES mean field may still relax in the larger space).
for key, tag, d0, d1, f0, f1 in (
        ('a', 'big ES', dev_big_before, dev_big_after, fo_big_before, d_big.fo_ene),
        ('b', 'small ES', dev_small_before, dev_small_after, fo_small_before, d_small.fo_ene)):
    n_occ = len(res['promoted_occ'].get(key, []))
    n_vir = len(res['promoted_vir'].get(key, []))
    if n_occ:
        assert abs(d1) < abs(d0), (
            f'{tag} gained {n_occ} occupied orbital(s) but the deviation did '
            f'not shrink: {d0:.6e} -> {d1:.6e}')
        print(f'OK: {tag} gained {n_occ} occ (+{n_vir} vir) and moved towards '
              f'exactness, {d0:.4e} -> {d1:.4e}')
    elif n_vir:
        assert abs(f1 - f0) < 1e-10, (
            f'{tag} gained only virtual orbitals, so dm_fo and fo_ene must be '
            f'unchanged, but fo_ene moved {f0:.12f} -> {f1:.12f}')
        print(f'OK: {tag} gained {n_vir} vir only; fo_ene unchanged, '
              f'deviation {d0:.4e} -> {d1:.4e}')

# ================================================================== Stage 3b
banner('Stage 3b: embedded MP2 AFTER matching')
e_big_after, ec_big_after = d_big.mp2_solver()
e_small_after, ec_small_after = d_small.mp2_solver()

print(f'\n  {"":24s} {"E_tot":>18s} {"E_corr":>16s} {"E - E(full MP2)":>18s}')


def row(tag, e, ec):
    print(f'  {tag:24s} {e:18.10f} {ec:16.10f} {e - e_full:18.10f}')


row('full-electron MP2', e_full, mymp2_full.e_corr)
row(f'big ES   before', e_big_before, ec_big_before)
row(f'big ES   after', e_big_after, ec_big_after)
row(f'small ES before', e_small_before, ec_small_before)
row(f'small ES after', e_small_after, ec_small_after)

gap_before = abs(e_big_before - e_small_before)
gap_after = abs(e_big_after - e_small_after)
print(f'\n  |E(big) - E(small)|    before = {gap_before:.8f} Ha '
      f'({gap_before * HARTREE2KCAL:8.3f} kcal/mol)')
print(f'  {"":21s} after  = {gap_after:.8f} Ha '
      f'({gap_after * HARTREE2KCAL:8.3f} kcal/mol)')
if gap_after < gap_before:
    print('OK: matching brought the two embedded spaces closer in energy')
else:
    print('NOTE: the two energies did not converge towards each other. '
          'Matching only equalises the SIZE of the space, not its content; '
          'inspect the promoted-orbital scores above.')

# ================================================================== Stage 4
banner('Stage 4: two DIFFERENT geometries (cross-molecule overlap path)')
mol_s, mf_s = build_mf(R_STRETCH, label='stretched')

scan_s = scan_thresholds(mf_s, 'h2o2_st')
print_scan(scan_s, f'H2O2 stretched r(OO)={R_STRETCH} A, impurity = O(0):')
scan_s_sizes = {t: (d.nfo, d.nfv, d.nes) for t, d in scan_s.items()}

# find a threshold at which the two geometries disagree -- that is exactly the
# LS/HS situation that motivated this module
t_pes = None
for t in THRESHOLDS:
    if scan_sizes[t][:2] != scan_s_sizes[t][:2]:
        t_pes = t
        break

if t_pes is None:
    print('\nNOTE: the two geometries give identical ES sizes at every '
          'threshold scanned; match_es_occ_vir will be a no-op here. '
          'Using the tightest threshold anyway to exercise the code path.')
    t_pes = THRESHOLDS[-1]
else:
    print(f'\nthe two geometries disagree at threshold={t_pes:g} '
          '-- this is the LS/HS failure mode, reproduced')

d_eq = build_dmet(mf, 'h2o2_pes_eq', t_pes)
d_st = build_dmet(mf_s, 'h2o2_pes_st', t_pes)

print('\nbefore matching:')
show(f'equilibrium r(OO)={R_EQ}', d_eq)
show(f'stretched   r(OO)={R_STRETCH}', d_st)

match_es_occ_vir(d_eq, d_st)

print('\nafter matching:')
show('equilibrium', d_eq)
show('stretched', d_st)

assert d_eq.nfo == d_st.nfo, f'nfo: {d_eq.nfo} vs {d_st.nfo}'
assert d_eq.nfv == d_st.nfv, f'nfv: {d_eq.nfv} vs {d_st.nfv}'
assert d_eq.nes == d_st.nes, f'nes: {d_eq.nes} vs {d_st.nes}'
assert (d_eq.mol.nelectron - 2 * d_eq.nfo) == \
       (d_st.mol.nelectron - 2 * d_st.nfo)
print(f'OK: both geometries now correlate the same number of electrons '
      f'({d_eq.mol.nelectron - 2 * d_eq.nfo}) in the same number of orbitals '
      f'({d_eq.nes})')

e_eq, _ = d_eq.mp2_solver()
e_st, _ = d_st.mp2_solver()
print(f'\nembedded MP2   E(eq) = {e_eq:.10f}   E(stretched) = {e_st:.10f}')
print(f'E(stretched) - E(eq) = {(e_st - e_eq) * HARTREE2KCAL:.4f} kcal/mol')

# full-electron reference for the same difference
mymp2_s = mp.MP2(mf_s)
mymp2_s.kernel()
d_full = (mymp2_s.e_tot - e_full) * HARTREE2KCAL
print(f'full-electron MP2 reference for the same difference = {d_full:.4f} kcal/mol')
print(f'error of the matched embedding = {(e_st - e_eq) * HARTREE2KCAL - d_full:.4f} kcal/mol')


# ================================================================== Stage 5
# Stage 4 only ever promoted OCCUPIED orbitals across geometries (nfv happened
# to agree at every threshold).  Exercise the cross-structure VIRTUAL branch on
# its own by letting the two geometries use different thresholds.
banner('Stage 5: cross-structure promotion of VIRTUAL orbitals only')


def find_pair(sizes_a, sizes_b, predicate):
    """First (t_a, t_b) whose (nfo, nfv) satisfy `predicate`, else (None, None)."""
    for ta, (nfo_a, nfv_a, _) in sizes_a.items():
        for tb, (nfo_b, nfv_b, _) in sizes_b.items():
            if predicate(nfo_a, nfv_a, nfo_b, nfv_b):
                return ta, tb
    return None, None


t_a, t_b = find_pair(scan_sizes, scan_s_sizes,
                     lambda fo_a, fv_a, fo_b, fv_b: fo_a == fo_b and fv_a > fv_b)

if t_a is None:
    print('SKIP: no (threshold_eq, threshold_stretched) pair has equal nfo but '
          'different nfv, so the virtual-only cross-structure branch cannot be '
          'reached in this system. Widen THRESHOLDS or change R_STRETCH.')
else:
    print(f'equilibrium @ threshold={t_a:g}   vs   stretched @ threshold={t_b:g}')
    d5a = build_dmet(mf, 'h2o2_s5_eq', t_a)
    d5b = build_dmet(mf_s, 'h2o2_s5_st', t_b)

    print('\nbefore matching:')
    show(f'equilibrium (thr={t_a:g})', d5a)
    show(f'stretched   (thr={t_b:g})', d5b)

    fo5a_before, fo5b_before = d5a.fo_ene, d5b.fo_ene
    dev5a_before, dev5b_before = dmet_deviation(d5a), dmet_deviation(d5b)

    res5 = match_es_occ_vir(d5a, d5b)

    print('\nafter matching:')
    show('equilibrium', d5a)
    show('stretched', d5b)

    assert not res5['promoted_occ'], \
        f'expected a virtual-only promotion, but got {res5["promoted_occ"]}'
    assert res5['promoted_vir'], 'nothing was promoted from the virtual block'
    assert d5a.nfo == d5b.nfo and d5a.nfv == d5b.nfv and d5a.nes == d5b.nes
    print('OK: nfo, nfv and nes match after a virtual-only promotion')

    # the sharp test: dm_fo did not move, so fo_ene must be bit-for-bit equal
    for tag, d, f0 in (('equilibrium', d5a, fo5a_before),
                       ('stretched', d5b, fo5b_before)):
        assert abs(d.fo_ene - f0) < 1e-10, (
            f'{tag}: only virtual orbitals were promoted, so fo_ene must not '
            f'change, but it moved {f0:.12f} -> {d.fo_ene:.12f}')
    print('OK: fo_ene unchanged on both sides (dm_fo untouched by a virtual '
          'promotion)')

    for tag, d0, d1 in (('equilibrium', dev5a_before, dmet_deviation(d5a)),
                        ('stretched', dev5b_before, dmet_deviation(d5b))):
        assert abs(d1) <= abs(d0) + 1e-10, \
            f'{tag}: deviation got worse, {d0:.6e} -> {d1:.6e}'
        print(f'  {tag:12s} deviation {d0:12.4e} -> {d1:12.4e} '
              f'(ES mean field relaxes in the larger space)')

    vir_scores = np.concatenate([np.atleast_1d(v)
                                 for v in res5['score_vir'].values()])
    print(f'promoted virtual scores: {np.round(vir_scores, 4).tolist()}')

# ================================================================== Stage 6
# The branch where BOTH objects have to grow (one short on occupied, the other
# short on virtuals).  Not reachable in every system.
banner('Stage 6: both objects grow (occ on one side, vir on the other)')

t_a, t_b = find_pair(scan_sizes, scan_s_sizes,
                     lambda fo_a, fv_a, fo_b, fv_b:
                     (fo_a - fo_b) * (fv_a - fv_b) < 0)

if t_a is None:
    print('SKIP: in this system nfo and nfv move together with `threshold`, so '
          'no pair leaves one object short on occupied and the other short on '
          'virtuals. The "both objects grow" branch is not reachable here; '
          'each side\'s growth path is covered separately by Stages 1, 4 and 5.')
else:
    print(f'equilibrium @ threshold={t_a:g}   vs   stretched @ threshold={t_b:g}')
    d6a = build_dmet(mf, 'h2o2_s6_eq', t_a)
    d6b = build_dmet(mf_s, 'h2o2_s6_st', t_b)

    print('\nbefore matching:')
    show(f'equilibrium (thr={t_a:g})', d6a)
    show(f'stretched   (thr={t_b:g})', d6b)

    res6 = match_es_occ_vir(d6a, d6b)

    print('\nafter matching:')
    show('equilibrium', d6a)
    show('stretched', d6b)

    grew = set(res6['promoted_occ']) | set(res6['promoted_vir'])
    assert grew == {'a', 'b'}, f'expected both objects to grow, got {grew}'
    assert d6a.nfo == d6b.nfo and d6a.nfv == d6b.nfv and d6a.nes == d6b.nes
    print('OK: both objects grew and the sizes match')

banner('all stages passed')
