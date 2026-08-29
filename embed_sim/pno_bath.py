'''
Pair-resolved (PNO-style) bath construction for single-shot DMET.

This module is an ALTERNATIVE to `embed_sim.BNO_bath`; it does not modify or
depend on the selection logic there (only the plotting helper is reused).

Motivation
----------
`BNO_bath` builds one *global* MP2 density in the frozen-occupied / frozen-virtual
space, diagonalizes it, and keeps natural orbitals whose occupation exceeds a
threshold eta.  Because the density is summed over *all* occupied pairs before
diagonalization, an environment orbital that is decisive for a single pair but
irrelevant on average gets averaged away, and eta itself carries no energy
meaning.

Following Bensberg & Neugebauer, J. Chem. Phys. 157, 064102 (2022)
("Orbital pair selection for relative energies in DLPNO-CC"), we instead

  1. resolve the MP2 density *per orbital pair* (pair natural orbitals, PNOs),
  2. rank / screen pairs by their semi-canonical MP2 **pair energy** (Hartree),
  3. take the union of the retained PNOs as the new bath, and
  4. report the correlation energy that the truncation actually discards,

so that the truncation parameter is expressed in Hartree rather than in
occupation numbers.  Step 2 is what makes the extension to reaction-energy
driven selection ("Delta-bath", Eq. 16 of the paper) a one-line change: replace
the pair energy by the *change* of the pair energy between two structures.

Note on domains: DLPNO additionally restricts each pair to a spatial PAO domain.
That is a cost device for linear scaling and is deliberately *not* implemented
here -- the frozen space of a DMET calculation is small enough that the full
pair amplitudes are affordable, and the domain plays no role in the accuracy of
the selection.

Conventions
-----------
Both the virtual-side and the occupied-side bath are built by the same routine:

  virtual side   pairs (i,j) run over embedded-space occupied orbitals,
                 PNOs live in the frozen-virtual space;
  occupied side  pairs (a,b) run over embedded-space virtual orbitals,
                 "hole PNOs" live in the frozen-occupied space.

This mirrors the half-transformed amplitudes t_ijAB / t_IJab used by `BNO_bath`,
so the two schemes can be compared at equal embedded-space size.

What the pair energies mean here
--------------------------------
Unlike `BNO_bath`, which builds the amplitudes over the *whole* virtual space and
projects the resulting density onto the frozen space, the amplitudes here are
built directly inside the frozen space.  The pair energy

    eps_ij = sum_{ab in frozen} t_ij^ab ( 2 (ia|jb) - (ib|ja) )

is therefore the part of the pair correlation energy that the current cluster
*cannot* describe -- excitations that already fit inside the embedded space are
excluded by construction.  This is exactly the quantity that should drive the
decision "should this environment orbital be promoted to bath?", and it makes
the reported numbers directly interpretable: `e_discarded` is the amount of
missing MP2 correlation energy that the truncated bath still fails to recover,
in Hartree.

Public entry points
-------------------
build_pno_rotations   basis-agnostic worker; returns orthogonal rotations of the
                      frozen spaces, split into (bath | rest)
get_RPNO_bath         drop-in replacement for `BNO_bath.get_RMP2_bath`
'''

import numpy as np

from pyscf import ao2mo, lib
from pyscf.lib import logger

from embed_sim.BNO_bath import make_histogram

# maximum number of doubles in an (ov|ov) integral tensor before we warn
_ERI_WARN_SIZE = 2.0e8


def _mo_eri(mf, C1, C2, C3, C4):
    '''(C1 C2 | C3 C4) in chemists' notation, returned as a 4-index array.'''
    n1, n2, n3, n4 = (C.shape[1] for C in (C1, C2, C3, C4))
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None:
        eri = with_df.ao2mo((C1, C2, C3, C4), compact=False)
    else:
        eri = ao2mo.general(mf.mol, (C1, C2, C3, C4), compact=False)
    return np.asarray(eri).reshape(n1, n2, n3, n4)


def _semicanonicalize(fock_ao, C):
    '''Diagonalize the AO Fock matrix inside the span of C.

    C must have orthonormal columns w.r.t. the AO overlap.  Returns the orbital
    energies, the rotated coefficients and the rotation itself.
    '''
    if C.shape[1] == 0:
        return np.zeros(0), C.copy(), np.zeros((0, 0))
    f = C.conj().T @ fock_ao @ C
    e, u = np.linalg.eigh(f)
    return e, C @ u, u


def _pno_density(T, diagonal_pair):
    '''DLPNO pair density, Riplinger & Neese, JCP 138, 034106 (2013), Eq. 8.

    D^ij = (1 + delta_ij)^-1 ( Tt^dag T + Tt T^dag ),  Tt = 2 T - T^T
    '''
    Tt = 2.0 * T - T.T
    D = Tt.conj().T @ T + Tt @ T.conj().T
    if diagonal_pair:
        D = D * 0.5
    return 0.5 * (D + D.conj().T)


def _pair_energy(T, K):
    '''Semi-canonical MP2 energy of a single pair: sum_ab t_ab (2 K_ab - K_ba).'''
    return float(np.sum(T * (2.0 * K - K.T)))


def _collect_pairs(eri, e_pair_orb, e_target_orb, sign, log, label):
    '''Build per-pair amplitudes, energies and PNOs for one side of the bath.

    Parameters
    ----------
    eri : (np, nt, np, nt) array
        (p t | p t) integrals; `p` indexes the orbitals that carry the pair
        label, `t` indexes the space the PNOs are constructed in.
    e_pair_orb : (np,) array
        Orbital energies of the pair-labelling space.
    e_target_orb : (nt,) array
        Orbital energies of the PNO space.
    sign : +1 or -1
        +1 when the pair label is occupied and the PNO space is virtual,
        -1 for the mirror case.  Fixes the sign of the energy denominator.

    Returns
    -------
    dict with keys 'pairs', 'K', 'eps', 'weight', 'occ', 'vec', 'e_target'.
    Pair (p,q) is stored once with p <= q; `weight` is 1 for p == q and 2
    otherwise, so that E_MP2 = sum(weight * eps).
    '''
    npair_orb = len(e_pair_orb)
    ntarget = len(e_target_orb)
    esum = e_target_orb[:, None] + e_target_orb[None, :]

    pairs, Ks, epss, weights, occs, vecs, econs = [], [], [], [], [], [], []
    for p in range(npair_orb):
        for q in range(p, npair_orb):
            K = np.ascontiguousarray(eri[p, :, q, :])
            denom = sign * (e_pair_orb[p] + e_pair_orb[q] - esum)
            T = K / denom
            eps = _pair_energy(T, K)
            D = _pno_density(T, p == q)
            n, v = np.linalg.eigh(D)
            order = np.argsort(n)[::-1]
            n, v = n[order], v[:, order]

            # exact partition of the pair energy over this pair's PNOs.
            # sum_ab A_ab B_ab and sum_ab A_ab B_ba are both invariant under a
            # two-sided orthogonal transform, so summing c over the complete
            # PNO basis reproduces eps exactly.
            Tt = v.T @ T @ v
            Kt = v.T @ K @ v
            M = Tt * (2.0 * Kt - Kt.T)
            c = 0.5 * (M.sum(axis=0) + M.sum(axis=1))

            pairs.append((p, q))
            Ks.append(K)
            epss.append(eps)
            weights.append(1.0 if p == q else 2.0)
            occs.append(n)
            vecs.append(v)
            econs.append(c)

    epss = np.asarray(epss)
    weights = np.asarray(weights)
    e_mp2 = float(np.sum(weights * epss))
    log.info('%s: %d pairs, %d candidate orbitals, semi-canonical E(MP2) = %.10f',
             label, len(pairs), ntarget, e_mp2)
    return dict(pairs=pairs, K=Ks, eps=epss, weight=weights, occ=occs, vec=vecs,
                econ=econs, e_target=e_target_orb, sign=sign,
                e_pair=e_pair_orb, e_mp2=e_mp2)


def _complement(B, ndim):
    '''Orthonormal complement of the columns of B inside an ndim-dim space.'''
    m = B.shape[1]
    if m == 0:
        return np.eye(ndim)
    if m >= ndim:
        return np.zeros((ndim, 0))
    P = np.eye(ndim) - B @ B.conj().T
    w, v = np.linalg.eigh(P)
    C = v[:, np.argsort(w)[::-1][:ndim - m]]
    C, _ = np.linalg.qr(C)
    return C


def _recovered_energy(data, B):
    '''Exact semi-canonical MP2 energy recoverable inside the subspace B.

    The amplitudes are *re-solved* in B (not merely projected), i.e. B is
    semi-canonicalized first, which is what actually happens once the orbitals
    are promoted into the embedded cluster.
    '''
    if B.shape[1] == 0:
        return 0.0, np.zeros(len(data['pairs']))
    f = B.conj().T @ np.diag(data['e_target']) @ B
    et, w = np.linalg.eigh(f)
    Bt = B @ w
    esum = et[:, None] + et[None, :]
    sign = data['sign']

    per_pair = np.empty(len(data['pairs']))
    for k, (p, q) in enumerate(data['pairs']):
        K = Bt.conj().T @ data['K'][k] @ Bt
        denom = sign * (data['e_pair'][p] + data['e_pair'][q] - esum)
        per_pair[k] = _pair_energy(K / denom, K)
    return float(np.sum(data['weight'] * per_pair)), per_pair


def _select_pnos(data, t_pno, t_pair, nbath, score, lindep, log, label,
                 t_orb_energy=None):
    '''Screen pairs and PNOs, return the orthonormal bath basis of this side.

    Candidate selection
    -------------------
    A pair survives if its total MP2 pair energy |w * eps_ij| exceeds `t_pair`
    (DLPNO `T_CutPairs`).  Within a surviving pair a PNO is a candidate if its
    occupation exceeds `t_pno` (DLPNO `T_CutPNO`), or -- if `t_orb_energy` is
    given -- if the energy it carries exceeds that value in Hartree.

    Ranking (`score`)
    -----------------
    'energy'      |c_n| * w, the exact share of the pair energy carried by that
                  PNO.  Hartree, comparable across pairs.
    'occ'         the bare PNO occupation.
    'occ_x_pair'  occupation times pair energy.  Kept only for comparison: the
                  occupation already scales like |t_ij|^2 and the pair energy
                  like |t_ij|^2 |K_ij|, so this double counts the strength of a
                  pair and over-weights pairs that are already well described.

    Two selection modes
    -------------------
    threshold mode (`nbath is None`)
        genuine DLPNO semantics -- the bath is the span of the union of the
        surviving PNOs.  The SVD only removes linear dependence.

    fixed-size mode (`nbath` given)
        the candidates are weighted by sqrt(score) and the leading `nbath` left
        singular vectors are kept.  Because

            sum_ij D^ij = M diag(n) M^T = (M diag(sqrt(n)))(M diag(sqrt(n)))^T,

        weighting by sqrt(n) and taking left singular vectors is *exactly*
        diagonalizing the weighted sum of the pair densities.  So `score='occ'`
        reproduces global BNO-like natural orbitals while `score='energy'`
        produces energy-weighted natural orbitals; the two differ only by the
        weight, which makes the equal-size comparison a controlled experiment.
    '''
    ndim = len(data['e_target'])
    empty_info = dict(n_pair_kept=0, n_cand=0, n_union=0)
    if ndim == 0:
        return np.zeros((0, 0)), empty_info

    eps = data['eps']
    weight = data['weight']
    pair_weight = np.abs(weight * eps)

    keep_pair = pair_weight >= t_pair
    e_dropped_pairs = float(np.sum((weight * eps)[~keep_pair]))
    log.info('%s: pair screening |w*eps| >= %.2e keeps %d/%d pairs '
             '(pair energy dropped: %.3e Ha)',
             label, t_pair, keep_pair.sum(), len(eps), e_dropped_pairs)

    cols, scores = [], []
    for k in range(len(eps)):
        if not keep_pair[k]:
            continue
        n = data['occ'][k]
        e_orb = np.abs(data['econ'][k]) * weight[k]
        if t_orb_energy is not None:
            sel = np.nonzero(e_orb > t_orb_energy)[0]
        else:
            sel = np.nonzero(n > t_pno)[0]
        if sel.size == 0:
            continue
        if score == 'energy':
            s = e_orb[sel]
        elif score == 'occ':
            s = n[sel]
        elif score == 'occ_x_pair':
            s = n[sel] * pair_weight[k]
        else:
            raise ValueError(f'unknown score {score!r}; use "energy", "occ" '
                             'or "occ_x_pair"')
        cols.append(data['vec'][k][:, sel])
        scores.append(s)

    if not cols:
        log.warn('%s: no PNO survived the screening', label)
        return np.zeros((ndim, 0)), dict(n_pair_kept=int(keep_pair.sum()),
                                         n_cand=0, n_union=0)

    M = np.hstack(cols)
    s_all = np.concatenate(scores)
    ncand = M.shape[1]

    if nbath is None:
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        B = U[:, sv > lindep]
    else:
        U, sv, _ = np.linalg.svd(M * np.sqrt(np.maximum(s_all, 0.0))[None, :],
                                 full_matrices=False)
        rank = int((sv > lindep * sv.max()).sum()) if sv.size else 0
        m = min(nbath, ndim, rank)
        if m < nbath:
            log.warn('%s: requested %d bath orbitals but only %d independent '
                     'directions are available', label, nbath, m)
        B = U[:, :m]

    cut = ('|c| > %.2e Ha' % t_orb_energy if t_orb_energy is not None
           else 'n > %.2e' % t_pno)
    log.info('%s: %d candidate PNOs (%s, from %d pairs) -> %d bath orbitals '
             '(of %d available)',
             label, ncand, cut, len(cols), B.shape[1], ndim)
    return B, dict(n_pair_kept=int(keep_pair.sum()), n_cand=int(ncand),
                   n_union=int(B.shape[1]),
                   e_dropped_pairs=e_dropped_pairs)


def _tune_to_energy(data, e_target_drop, t_pair, score, lindep,
                    log, label, maxiter=18):
    '''Bisect the per-orbital energy threshold to hit a target discarded energy.

    The knob is `t_orb_energy` (Hartree), so the whole selection is expressed in
    energy units: keep every PNO carrying more than `t_orb_energy` of its pair
    energy, and choose that threshold so the MP2 correlation energy left behind
    is below `e_target_drop`.

    Lowering the threshold only adds candidates, so the retained subspaces are
    nested and the discarded energy decreases monotonically; bisection is safe.
    '''
    quiet = logger.new_logger(None, verbose=0)
    lo, hi = 1e-14, 1e-1          # tight (keeps more) / loose (keeps less)
    best = None
    for _ in range(maxiter):
        mid = np.sqrt(lo * hi)
        B, info = _select_pnos(data, 0.0, t_pair, None, score, lindep,
                               quiet, label, t_orb_energy=mid)
        e_keep, _ = _recovered_energy(data, B)
        drop = abs(data['e_mp2'] - e_keep)
        log.debug('%s: t_orb_energy = %.3e -> nbath = %d, discarded = %.3e Ha',
                  label, mid, B.shape[1], drop)
        if drop > e_target_drop:
            hi = mid                      # too coarse, tighten
        else:
            lo = mid                      # acceptable, try coarser
            best = (mid, B, info, drop)
    if best is None:
        log.warn('%s: cannot reach a discarded energy of %.2e Ha even at the '
                 'tightest threshold; using t_orb_energy = %.2e',
                 label, e_target_drop, lo)
        B, info = _select_pnos(data, 0.0, t_pair, None, score, lindep, log,
                               label, t_orb_energy=lo)
        e_keep, _ = _recovered_energy(data, B)
        return B, info, lo, abs(data['e_mp2'] - e_keep)
    t_used, B, info, drop = best
    log.info('%s: energy-targeted threshold t_orb_energy = %.3e Ha '
             '(discarded %.3e Ha <= target %.3e Ha)',
             label, t_used, drop, e_target_drop)
    return B, info, t_used, drop


def build_pno_rotations(mf, es_mf, ao2eo, ao2core, ao2vir,
                        t_pno=1e-7, t_pair=1e-5, t_orb_energy=None,
                        nbath_occ=None, nbath_vir=None,
                        e_target=None, score='energy', occ_side=True,
                        vir_side=True, lindep=1e-6, verbose=None):
    '''Pair-resolved bath selection in the frozen occupied / virtual spaces.

    Parameters
    ----------
    mf : pyscf SCF object
        Full-system mean-field reference (closed shell).
    es_mf : pyscf SCF object
        Mean-field solution inside the current embedded space.
    ao2eo, ao2core, ao2vir : ndarray
        AO -> embedded / frozen-occupied / frozen-virtual coefficients, each
        with orthonormal columns.
    t_pno : float
        PNO occupation threshold (DLPNO `T_CutPNO` analogue).  Ignored when
        `t_orb_energy` is given.
    t_pair : float
        Pair-energy threshold in Hartree (DLPNO `T_CutPairs` analogue).  Pairs
        contributing less than this are not allowed to add bath orbitals.
    t_orb_energy : float or None
        Alternative, energy-native candidate criterion: keep a PNO if the share
        of the pair energy it carries exceeds this value (Hartree).  Takes
        precedence over `t_pno`.
    nbath_occ, nbath_vir : int or None
        If given, keep exactly this many bath orbitals on that side, ranked by
        `score`.  Use for equal-size comparisons against `BNO_bath`.
    e_target : float or None
        If given, bisect `t_orb_energy` per side so that the discarded
        semi-canonical MP2 correlation energy stays below `e_target` Hartree.
    score : {'energy', 'occ', 'occ_x_pair'}
        Ranking used in fixed-size mode.  'energy' is the exact share of the
        pair energy carried by each PNO (Hartree); 'occ' is the bare occupation
        and reproduces global BNO-like natural orbitals; 'occ_x_pair' is kept
        only to demonstrate that multiplying occupation by pair energy double
        counts the pair strength.
    occ_side, vir_side : bool
        Whether to expand into the frozen-occupied / frozen-virtual space.

    Returns
    -------
    rot : dict
        'core_bath', 'core_rest', 'vir_bath', 'vir_rest' -- orthogonal
        rotations *within* the coefficient space of `ao2core` / `ao2vir`.
        [bath | rest] is a complete orthonormal basis of the respective space,
        so the total span is preserved exactly.
    info : dict
        Diagnostics, including the discarded correlation energy per side.
    '''
    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing pair-resolved (PNO) bath')

    if np.any(np.abs(mf.mo_occ - 1.0) < 1e-8):
        raise NotImplementedError(
            'pno_bath currently supports closed-shell references only; '
            'singly occupied orbitals detected in the full-system reference')
    if np.any(np.abs(es_mf.mo_occ - 1.0) < 1e-8):
        raise NotImplementedError(
            'pno_bath currently supports closed-shell references only; '
            'singly occupied orbitals detected in the embedded reference')

    fock_ao = mf.get_fock()
    nfo = ao2core.shape[1]
    nfv = ao2vir.shape[1]

    es_occ = lib.dot(ao2eo, es_mf.mo_coeff[:, es_mf.mo_occ > 0])
    es_vir = lib.dot(ao2eo, es_mf.mo_coeff[:, es_mf.mo_occ == 0])

    # one common Fock operator for every space -> consistent denominators
    e_eo, C_eo, _ = _semicanonicalize(fock_ao, es_occ)
    e_ev, C_ev, _ = _semicanonicalize(fock_ao, es_vir)
    e_fo, C_fo, U_fo = _semicanonicalize(fock_ao, ao2core)
    e_fv, C_fv, U_fv = _semicanonicalize(fock_ao, ao2vir)

    info = {}
    rot = {}

    # ---------------- virtual side: pairs = embedded occupied ----------------
    if vir_side and nfv > 0 and C_eo.shape[1] > 0:
        size = C_eo.shape[1] ** 2 * nfv ** 2
        if size > _ERI_WARN_SIZE:
            log.warn('virtual-side (ov|ov) tensor has %.2e elements (%.1f GB)',
                     size, size * 8 / 2 ** 30)
        eri = _mo_eri(mf, C_eo, C_fv, C_eo, C_fv)
        data_v = _collect_pairs(eri, e_eo, e_fv, +1, log, 'virtual side')
        if e_target is not None:
            Bv, iv, t_used, drop = _tune_to_energy(
                data_v, e_target, t_pair, score, lindep, log, 'virtual side')
            iv['t_orb_energy'] = t_used
        else:
            Bv, iv = _select_pnos(data_v, t_pno, t_pair, nbath_vir, score,
                                  lindep, log, 'virtual side',
                                  t_orb_energy=t_orb_energy)
            iv['t_pno'] = t_pno
            iv['t_orb_energy'] = t_orb_energy
        e_keep, _ = _recovered_energy(data_v, Bv)
        iv.update(e_mp2=data_v['e_mp2'], e_recovered=e_keep,
                  e_discarded=data_v['e_mp2'] - e_keep, ndim=nfv)
        log.info('virtual side: E(MP2) recovered by bath = %.10f of %.10f '
                 '(%.2f%%), discarded = %.3e Ha',
                 e_keep, data_v['e_mp2'],
                 100 * e_keep / data_v['e_mp2'] if data_v['e_mp2'] else 0.0,
                 data_v['e_mp2'] - e_keep)
        bins = np.array([10 ** -x for x in range(0, 11)][::-1])
        all_occ = np.concatenate([o for o in data_v['occ']])
        all_occ = all_occ[all_occ > 0]     # pair densities are not strictly PSD
        log.info('virtual-side PNO occupation histogram (all pairs)')
        log.info('%s', make_histogram(all_occ, bins, labels=True, show_number=True))
        info['vir'] = iv
        rot['vir_bath'] = U_fv @ Bv
        rot['vir_rest'] = U_fv @ _complement(Bv, nfv)
    else:
        rot['vir_bath'] = np.zeros((nfv, 0))
        rot['vir_rest'] = np.eye(nfv)
        info['vir'] = dict(n_union=0, ndim=nfv)

    # ------------- occupied side: pairs = embedded virtual (mirror) ----------
    if occ_side and nfo > 0 and C_ev.shape[1] > 0:
        size = C_ev.shape[1] ** 2 * nfo ** 2
        if size > _ERI_WARN_SIZE:
            log.warn('occupied-side (ov|ov) tensor has %.2e elements (%.1f GB)',
                     size, size * 8 / 2 ** 30)
        eri = _mo_eri(mf, C_ev, C_fo, C_ev, C_fo)
        data_o = _collect_pairs(eri, e_ev, e_fo, -1, log, 'occupied side')
        if e_target is not None:
            Bo, io, t_used, drop = _tune_to_energy(
                data_o, e_target, t_pair, score, lindep, log, 'occupied side')
            io['t_orb_energy'] = t_used
        else:
            Bo, io = _select_pnos(data_o, t_pno, t_pair, nbath_occ, score,
                                  lindep, log, 'occupied side',
                                  t_orb_energy=t_orb_energy)
            io['t_pno'] = t_pno
            io['t_orb_energy'] = t_orb_energy
        e_keep, _ = _recovered_energy(data_o, Bo)
        io.update(e_mp2=data_o['e_mp2'], e_recovered=e_keep,
                  e_discarded=data_o['e_mp2'] - e_keep, ndim=nfo)
        log.info('occupied side: E(MP2) recovered by bath = %.10f of %.10f '
                 '(%.2f%%), discarded = %.3e Ha',
                 e_keep, data_o['e_mp2'],
                 100 * e_keep / data_o['e_mp2'] if data_o['e_mp2'] else 0.0,
                 data_o['e_mp2'] - e_keep)
        info['occ'] = io
        rot['core_bath'] = U_fo @ Bo
        rot['core_rest'] = U_fo @ _complement(Bo, nfo)
    else:
        rot['core_bath'] = np.zeros((nfo, 0))
        rot['core_rest'] = np.eye(nfo)
        info['occ'] = dict(n_union=0, ndim=nfo)

    nb_o = rot['core_bath'].shape[1]
    nb_v = rot['vir_bath'].shape[1]
    log.info('Number of newly added bath orbitals = %s (%s from core, %s from virtual)',
             nb_o + nb_v, nb_o, nb_v)
    log.info('')
    info['nbath_occ'] = nb_o
    info['nbath_vir'] = nb_v
    return rot, info


def get_RPNO_bath(mf, es_mf, ao2eo, ao2core, ao2vir, lo2core, lo2vir,
                  verbose=None, **kwargs):
    '''Drop-in replacement for `BNO_bath.get_RMP2_bath`.

    Same call signature and same return convention -- the new bath orbitals and
    the remaining frozen orbitals are returned in the LO basis.  Keyword
    arguments are forwarded to `build_pno_rotations`; note that this routine
    takes `t_pno` / `t_pair` rather than a single `eta`.
    '''
    rot, info = build_pno_rotations(mf, es_mf, ao2eo, ao2core, ao2vir,
                                    verbose=verbose, **kwargs)
    lo2bath = np.hstack((lib.dot(lo2core, rot['core_bath']),
                         lib.dot(lo2vir, rot['vir_bath'])))
    lo2core_new = lib.dot(lo2core, rot['core_rest'])
    lo2vir_new = lib.dot(lo2vir, rot['vir_rest'])
    return lo2bath, lo2core_new, lo2vir_new
