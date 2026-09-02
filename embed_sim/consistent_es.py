"""
Align the *size* of the embedded space (ES) between two DMET calculations.

Motivation
----------
`ssdmet.SSDMET.ROHF()` sets ``mol.nelectron = mol.nelectron - 2*nfo``, so the
number of electrons treated inside the embedded space is fixed entirely by
``nfo`` (the number of frozen-occupied orbitals).  Likewise
``nes = nao - nfo - nfv``.  Two structures (e.g. the low-spin and high-spin
geometries of a spin-crossover complex) therefore end up correlating a
*different* number of electrons in a *different* number of orbitals whenever
their entanglement spectra partition differently.  The truncation error is then
not the same on both sides and does not cancel in an energy difference.

This module makes ``nfo`` and ``nfv`` -- hence the ES occupied and virtual
counts -- equal between two DMET objects, by promoting the missing frozen
orbitals back into the embedded space.  Which frozen orbitals to promote is
decided by their overlap with the *reference* structure's ES occupied (resp.
virtual) space.

Only promotion (frozen -> ES) is possible with the machinery in
`consistent_bath.append_bath_by_env_idx`, so the side with the larger ``nfo``
is always the one that grows, i.e. both structures are levelled *up* to the
larger embedded space.

See also `consistent_bath.py`, which matches the bath space as a whole but
does not control the occupied/virtual split.
"""

import numpy as np
from functools import reduce
from pyscf import gto, lib

from embed_sim.consistent_bath import append_bath_by_env_idx


def _cross_ovlp(mol_a, mol_b):
    """AO overlap <a|b> between two Mole objects (same object -> plain S)."""
    if mol_a is mol_b:
        return mol_a.intor_symmetric('int1e_ovlp')
    return gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)


def _es_occ_vir_orb(dmet):
    """ES occupied / virtual orbitals of `dmet`, expressed in its own AO basis."""
    if dmet.es_mf is None:
        raise RuntimeError('embedded subspace not built; run build() first')
    mo_occ = np.asarray(dmet.es_mf.mo_occ)
    if mo_occ.ndim == 2:  # UHF-like, use the alpha channel
        mo_occ = mo_occ[0] + mo_occ[1]
    mo = dmet.es_mf.mo_coeff
    occ_mask = mo_occ > 1e-8
    es_occ_orb = lib.dot(dmet.es_orb, mo[:, occ_mask])
    es_vir_orb = lib.dot(dmet.es_orb, mo[:, ~occ_mask])
    return es_occ_orb, es_vir_orb


def dmet_deviation(dmet):
    """
    Deviation from the DMET exactness condition,
    ``E(ES mean field) + E(frozen occupied) - E(total mean field)``.

    This vanishes when the frozen-occupied block consists of environment
    natural orbitals with occupation exactly 2, i.e. when no entangled orbital
    has been pushed out of the embedded space.  Truncating the bath makes it
    finite; promoting the discarded orbitals back into the ES drives it back
    towards zero.

    The two blocks act through different channels:

    * promoting a frozen **occupied** orbital changes ``dm_fo``, hence
      ``fo_ene`` directly;
    * promoting a frozen **virtual** orbital leaves ``fo_ene`` untouched
      (``dm_fo`` is unchanged), but it enlarges the variational space of the
      embedded mean field, so ``es_mf.e_tot`` may still drop.

    So a virtual-only promotion must leave ``fo_ene`` bit-for-bit identical,
    while the deviation itself may improve.
    """
    if dmet.es_mf is None:
        raise RuntimeError('embedded subspace not built; run build() first')
    return dmet.es_mf.e_tot + dmet.fo_ene - dmet.mf_or_cas.e_tot


def _check_promotable(dmet, name):
    if dmet.es_natorb:
        raise ValueError(
            f'{name}: es_natorb=True is not supported here. Rebuilding the ES '
            'reuses the stale self.es_occ in make_es_dm(). Rebuild the DMET '
            'object with es_natorb=False.')
    if dmet.lo_cloes is None:
        raise RuntimeError(f'{name}: run build() first (lo_cloes is None)')


def select_frozen_to_promote(dmet_tgt, ref_orb, mol_ref, space='occ',
                             n_promote=1, log=None):
    """
    Pick which frozen orbitals of `dmet_tgt` to promote into its embedded space.

    Each candidate is scored by how much of it lies inside the reference
    subspace spanned by `ref_orb`::

        M      = C_frozen^T S_cross C_ref
        score_i = sum_j |M_ij|^2                 in [0, 1]

    Parameters
    ----------
    dmet_tgt : SSDMET
        The object that will be grown.
    ref_orb : ndarray
        Reference orbitals in the AO basis of `mol_ref` (ES occupied orbitals
        when ``space='occ'``, ES virtual orbitals when ``space='vir'``).
    mol_ref : Mole
        Molecule the reference orbitals belong to.
    space : {'occ', 'vir'}
        Whether to draw candidates from the frozen-occupied or the
        frozen-virtual block.
    n_promote : int
        How many orbitals to promote.

    Returns
    -------
    local_idx : list[int]
        Indices *within* the chosen frozen block (0-based).
    scores : ndarray
        Score of every candidate, in block order.
    """
    if space == 'occ':
        cand = dmet_tgt.fo_orb
    elif space == 'vir':
        cand = dmet_tgt.fv_orb
    else:
        raise ValueError(f"space must be 'occ' or 'vir', got {space!r}")

    if n_promote > cand.shape[1]:
        raise ValueError(
            f'requested {n_promote} orbitals from the frozen-{space} block, '
            f'but only {cand.shape[1]} are available')

    s_cross = _cross_ovlp(dmet_tgt.mol, mol_ref)
    ovlp = reduce(lib.dot, (cand.conj().T, s_cross, ref_orb))
    scores = np.einsum('ij,ij->i', ovlp, ovlp.conj()).real

    order = np.argsort(scores)[::-1]
    local_idx = sorted(order[:n_promote].tolist())

    if log is not None:
        log.info(f'  frozen-{space} candidates: {cand.shape[1]}, promoting {n_promote}')
        for rank, i in enumerate(order[:min(len(order), n_promote + 3)]):
            mark = '  <== promote' if i in local_idx else ''
            log.info(f'    rank {rank:3d}  local_idx {int(i):4d}  '
                     f'score {scores[i]:.6f}{mark}')
    return local_idx, scores


def match_es_occ_vir(dmet_a, dmet_b, verbose=True):
    """
    Make the ES occupied and virtual orbital counts equal for two DMET objects.

    The object with the larger ``nfo`` promotes ``|nfo_a - nfo_b|`` orbitals out
    of its frozen-occupied block; the object with the larger ``nfv`` promotes
    ``|nfv_a - nfv_b|`` out of its frozen-virtual block.  Selection is by
    overlap with the other object's ES occupied / virtual space.

    Both objects are modified in place (whichever needs to grow).

    Returns
    -------
    dict with keys 'before', 'after', 'promoted_occ', 'promoted_vir',
    'score_occ', 'score_vir'.
    """
    _check_promotable(dmet_a, 'dmet_a')
    _check_promotable(dmet_b, 'dmet_b')

    log = dmet_a.log if verbose else None

    def _sizes(d):
        return dict(nimp=len(d.imp_idx), nes=d.nes, nfo=d.nfo, nfv=d.nfv,
                    nelec_es=d.mol.nelectron - 2 * d.nfo,
                    deviation=dmet_deviation(d))

    before = {'a': _sizes(dmet_a), 'b': _sizes(dmet_b)}

    if log is not None:
        log.info('=' * 70)
        log.info('Embedded-space occ/vir count matching')
        log.info(f"{'':6s} {'nes':>6s} {'nfo':>6s} {'nfv':>6s} {'nelec_ES':>10s} "
                 f"{'DMET deviation':>18s}")
        for key in ('a', 'b'):
            s = before[key]
            log.info(f"{key:6s} {s['nes']:6d} {s['nfo']:6d} {s['nfv']:6d} "
                     f"{s['nelec_es']:10d} {s['deviation']:18.10e}")

    dn_o = dmet_a.nfo - dmet_b.nfo
    dn_v = dmet_a.nfv - dmet_b.nfv

    result = {'before': before, 'promoted_occ': {}, 'promoted_vir': {},
              'score_occ': {}, 'score_vir': {}}

    # (grower, reference) for the occupied and the virtual block independently
    plan = []
    if dn_o > 0:
        plan.append(('a', 'b', 'occ', dn_o))
    elif dn_o < 0:
        plan.append(('b', 'a', 'occ', -dn_o))
    if dn_v > 0:
        plan.append(('a', 'b', 'vir', dn_v))
    elif dn_v < 0:
        plan.append(('b', 'a', 'vir', -dn_v))

    if not plan:
        if log is not None:
            log.info('nfo and nfv already match; nothing to do.')
            log.info('=' * 70)
        result['after'] = {'a': _sizes(dmet_a), 'b': _sizes(dmet_b)}
        return result

    objs = {'a': dmet_a, 'b': dmet_b}

    # Group by grower: all promotions for one object must go into a SINGLE
    # append_bath_by_env_idx call, because that call renumbers the frozen
    # blocks and would invalidate a second batch of indices.
    for grower_key in ('a', 'b'):
        jobs = [p for p in plan if p[0] == grower_key]
        if not jobs:
            continue
        tgt = objs[grower_key]
        nimp = len(tgt.imp_idx)
        nbath = tgt.nes - nimp
        nfo = tgt.nfo

        if log is not None:
            log.info('')
            log.info(f'growing dmet_{grower_key}:')

        env_idx = []
        for _, ref_key, space, n_need in jobs:
            ref = objs[ref_key]
            ref_occ_orb, ref_vir_orb = _es_occ_vir_orb(ref)
            ref_orb = ref_occ_orb if space == 'occ' else ref_vir_orb
            local_idx, scores = select_frozen_to_promote(
                tgt, ref_orb, ref.mol, space=space, n_promote=n_need, log=log)
            # env index convention of append_bath_by_env_idx:
            #   [0, nbath)                    -> current bath
            #   [nbath, nbath+nfo)            -> frozen occupied
            #   [nbath+nfo, nbath+nfo+nfv)    -> frozen virtual
            offset = nbath if space == 'occ' else nbath + nfo
            env_idx.extend(offset + i for i in local_idx)
            result[f'promoted_{space}'][grower_key] = local_idx
            result[f'score_{space}'][grower_key] = scores[local_idx]

        append_bath_by_env_idx(tgt, sorted(env_idx))

    after = {'a': _sizes(dmet_a), 'b': _sizes(dmet_b)}
    result['after'] = after

    if log is not None:
        log.info('')
        log.info(f"{'':6s} {'nes':>6s} {'nfo':>6s} {'nfv':>6s} {'nelec_ES':>10s} "
                 f"{'DMET deviation':>18s}")
        for key in ('a', 'b'):
            s = after[key]
            log.info(f"{key:6s} {s['nes']:6d} {s['nfo']:6d} {s['nfv']:6d} "
                     f"{s['nelec_es']:10d} {s['deviation']:18.10e}")
        for key in ('a', 'b'):
            d0, d1 = abs(before[key]['deviation']), abs(after[key]['deviation'])
            if d1 > d0 + 1e-10:
                log.warn(f'dmet_{key}: DMET exactness deviation grew, '
                         f'{d0:.6e} -> {d1:.6e}; the promoted orbitals are '
                         'probably not the entangled ones')
            elif d1 < d0:
                log.info(f'dmet_{key}: DMET exactness deviation improved, '
                         f'{d0:.6e} -> {d1:.6e}')

    if after['a']['nfo'] != after['b']['nfo']:
        raise RuntimeError(f"nfo still differs after matching: {after}")
    if after['a']['nfv'] != after['b']['nfv']:
        raise RuntimeError(f"nfv still differs after matching: {after}")
    if dmet_a.mol.nao == dmet_b.mol.nao and after['a']['nes'] != after['b']['nes']:
        raise RuntimeError(f"nes still differs after matching: {after}")

    if log is not None:
        log.info('embedded spaces matched: nfo, nfv and nes are now equal')
        log.info('=' * 70)
    return result
