"""Grow the DMET embedded space until the HF-in-HF exact condition holds.

The HF-in-HF exact condition of DMET reads

    E(embedded HF) + E(frozen occupied) == E(full low-level HF)

It is satisfied by construction when the bath spans *all* environment natural
orbitals with non-trivial occupation (the exact bath).  A truncated bath
(e.g. one orbital per bond) violates it; the deviation

    |E(es_mf) + E(fo_ene) - E(mf_or_cas)|

is a direct measure of the embedding error.

This module implements the growth strategy suggested for DMET_main: start from
an initial embedded space (e.g. impurity + one-bath-orbital-per-bond), check
the HF-in-HF deviation, and if it exceeds a tolerance, add orbitals to the
bath using the concentric localization machinery of `concentric_loc.py`
(virtual and occupied shells around the impurity atoms).  Growth stops when
the deviation is below the tolerance, i.e. the exact condition is met.  Only
then is the embedded space considered safe for a correlated calculation.

Note: the concentric growth refreshes the embedded density from ``lo2es``, so
the SSDMET/AODMET object must be built with ``es_natorb=False`` (same
constraint as the MP2 bath expansion).

Two stopping rules are provided:

- HF gate (default): stop when |E(es)+E(fo)-E(full)| <= tol.  Near-empty
  environment virtuals may remain frozen; they are irrelevant for the
  mean-field energy but still contribute to correlation (MP2), so the
  correlated DMET energy still differs from all-electron MP2 by the frozen
  environment correlation - this difference is the genuine DMET-vs-full gap.
- full-space gate (``include_all_virtuals=True``): additionally grow until
  no virtual orbitals are left frozen, so the embedded virtual space is
  complete and MP2-in-HF reproduces all-electron MP2 (useful as a reference
  that isolates the embedding error from the correlation-truncation error).
"""

import numpy as np
from pyscf import lib


def hf_in_hf_deviation(dmet):
    """HF-in-HF exactness deviation |E(es_mf) + E(fo_ene) - E(full)| in Hartree."""
    return abs(dmet.es_mf.e_tot + dmet.fo_ene - dmet.mf_or_cas.e_tot)


def grow_bath_to_exact(dmet, tol=1e-6, max_rounds=20, proj_bas='sto-3g',
                       atoms=None, shell_order=('vir', 'occ'),
                       include_all_virtuals=False):
    """Grow the bath of an already-built SSDMET/AODMET until the exact
    condition is satisfied.  Mutates ``dmet`` in place.

    Args:
        dmet: SSDMET/AODMET object built with ``es_natorb=False``.
        tol: HF-in-HF deviation tolerance in Hartree.
        max_rounds: maximum number of (virtual + occupied) shell additions.
        proj_bas: projection basis for the concentric localization
            (e.g. 'sto-3g', 'minao').
        atoms: atom indices around which shells are grown; default: impurity
            atoms.
        shell_order: order in which virtual ('vir') and occupied ('occ')
            shells are added each round.
        include_all_virtuals: if True, also grow virtual shells until
            ``nfv == 0`` so that the embedded space spans the whole molecule.

    Returns:
        tuple ``(dmet, history)``; ``history`` is a list of dicts with the
        per-round growth summary.  Raises RuntimeError if the tolerance is
        not reached within ``max_rounds``.
    """
    from embed_sim import concentric_loc
    from embed_sim.bath_selection import imp_atom_indices

    if dmet.es_natorb:
        raise RuntimeError(
            'grow_bath_to_exact requires es_natorb=False: the concentric '
            'growth refreshes es_dm from lo2es (same constraint as bath_option)')
    if dmet.lo_cloes is None:
        raise RuntimeError('Run dmet.build() first')

    nimp = len(dmet.imp_idx)
    if atoms is None:
        atoms = imp_atom_indices(dmet.mol, dmet.imp_idx)

    history = []
    dev = hf_in_hf_deviation(dmet)
    dmet.log.info('grow_bath_to_exact: initial nbath=%d nfo=%d nfv=%d '
                  'HF-in-HF deviation=%.3e', dmet.nes - nimp, dmet.nfo,
                  dmet.nfv, dev)

    for rnd in range(max_rounds):
        if dev <= tol and not (include_all_virtuals and dmet.nfv > 0):
            break
        nvir0, nfo0 = dmet.nfv, dmet.nfo
        for kind in shell_order:
            if kind == 'vir' and dmet.nfv > 0:
                concentric_loc.concentric_localization(
                    dmet, proj_bas, 1, atoms, threshold=1e-8)
            elif kind == 'occ' and dmet.nfo > 0:
                concentric_loc.concentric_occ_localization(
                    dmet, proj_bas, 1, atoms, threshold=1e-8)
        add_vir = nvir0 - dmet.nfv
        add_occ = nfo0 - dmet.nfo
        dev = hf_in_hf_deviation(dmet)
        history.append(dict(round=rnd, add_vir=add_vir, add_occ=add_occ,
                            nbath=dmet.nes - nimp, nfo=dmet.nfo,
                            nfv=dmet.nfv, deviation=dev))
        dmet.log.info('grow_bath_to_exact: round %d +%d vir +%d occ -> '
                      'nbath=%d nfo=%d nfv=%d deviation=%.3e',
                      rnd, add_vir, add_occ, dmet.nes - nimp, dmet.nfo,
                      dmet.nfv, dev)

    if dev > tol:
        raise RuntimeError(
            'HF-in-HF exact condition not reached after %d rounds: '
            'deviation=%.3e > tol=%.3e' % (max_rounds, dev, tol))
    if include_all_virtuals and dmet.nfv > 0:
        # Concentric virtual shells leave the weakly coupled tail in the
        # kernel; for the full-space gate simply merge the remaining frozen
        # virtuals into the bath (embedded space = whole molecule).
        nimp = len(dmet.imp_idx)
        nbath = dmet.nes - nimp
        q_emb = dmet.lo_cloes[:, :dmet.nes]
        q_fv_rest = dmet.lo_cloes[:, dmet.nes + dmet.nfo:]
        q_fo = dmet.lo_cloes[:, dmet.nes:dmet.nes + dmet.nfo]
        n_merged = dmet.nfv
        dmet.lo_cloes = np.hstack([q_emb, q_fv_rest, q_fo])
        dmet.nes += n_merged
        dmet.nfv = 0
        dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
        dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes:dmet.nes + dmet.nfo])
        dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes + dmet.nfo:])
        dmet.es_int1e = dmet.make_es_int1e()
        if hasattr(dmet, 'es_cderi'):
            dmet.es_cderi = dmet.make_es_cderi()
        else:
            dmet.es_int2e = dmet.make_es_int2e()
        dm_arg = dmet.dm_pair if (dmet.open_shell and dmet.dm_pair is not None) else dmet.dm
        dmet.es_dm = dmet.make_es_dm(dmet.open_shell, dmet.lo_cloes[:, :dmet.nes],
                                     dmet.cloao, dm_arg)
        dmet.es_mf = dmet.ROHF()
        dmet.calc_fo_ene()
        dev = hf_in_hf_deviation(dmet)
        history.append(dict(round=len(history), add_vir=n_merged,
                            add_occ=0, nbath=dmet.nes - nimp, nfo=dmet.nfo,
                            nfv=0, deviation=dev,
                            note='merge remaining frozen virtuals'))
        dmet.log.info('grow_bath_to_exact: merged %d remaining frozen '
                      'virtuals into bath (full-space gate); deviation=%.3e',
                      history[-1]['add_vir'], dev)
    return dmet, history
