"""Verification suite for the shift-and-invert (resolvent) CL bath expansion.

The point of shift-and-invert CL is to put an energy denominator into the
otherwise denominator-free (alpha = 0) CL selection, by replacing the coupling
operator F by the resolvent R = (P F P - sigma)^{-1} of the same
candidate-block Fock matrix.  Because R is a *function* of the operator it
replaces, the construction comes with several exact identities that make it
testable without any reference data:

  T1  Krylov identity.  The cumulative CL shells must span
      K_n(A, C_0) = span{C_0, A C_0, ..., A^{n-1} C_0} for whatever A is used.
      This is Prop. 2.1 of the CL/BNO note, checked for the new operator.

  T2  Exhausted-span invariance.  R = g(P F P) with g(x) = 1/(x - sigma)
      injective on the spectrum, so the *exhausted* Krylov space is IDENTICAL
      to the Fock one.  Only the ORDER of promotion changes.  This is the
      sharpest single test: it must hold to machine precision, and it fails
      loudly under almost any plumbing bug.

  T3  sigma -> -infinity limit.  (F - sigma)^{-1} = tau(1 + tau F + ...) with
      tau = -1/sigma, and the leading term drops out of every shell SVD by
      orthogonality (X_n^dag X_ker,n = 0), so shell n of shift-invert must
      converge to shell n of plain Fock CL as sigma -> -inf.  This checks the
      operator build and demonstrates that sigma is a one-parameter family
      interpolating CL <-> denominator-weighted selection.

  T4  Definiteness / order reversal.  With sigma strictly below the candidate
      spectrum, all 1/(eps - sigma) > 0 and the resolvent's eigenvalue order is
      exactly the reverse of the Fock order.

  T5  AO-representation round trip.  build_resolvent_op returns an AO matrix
      R_ao; an independent construction of M = scale*(F_cand - sigma)^{-1} in
      candidate coordinates must satisfy C^dag R_ao C = M.

  T6  DMET exactness (HF-in-HF).  Promoting orbitals between EO and FO/FV is a
      repartition of one and the same total space, so the reference determinant
      stays exactly representable: es_mf.e_tot + fo_ene - mf.e_tot must remain
      at SCF precision for EVERY shell count and EVERY coupling operator.
      This exercises the whole reassembly path (es_int1e/es_dm/ROHF/fo_ene).

  T7  Basis integrity.  The reassembled lo_cloes must stay orthonormal and
      complete.

  T8  Production-vs-reference span.  The span promoted by the production
      routine must equal the span from the independent reference recursion.

  T9  eps distribution (the physics claim, not a correctness test).  Fock CL
      should promote BOTH ends of the candidate spectrum (Lanczos converges on
      extremes); shift-invert CL should concentrate on the low-eps end.

  T10 Negative control.  A sigma inside the candidate spectrum must be caught
      and relocated outside it.

Run from the DMET project root:  python examples/test_example/shift_invert_cl.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from embed_sim import ssdmet, concentric_loc  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=120)

TITLE = 'shift_invert_cl'
PROJ_BAS = 'sto-3g'
ATOMS_A = [0]
THRESHOLD = 1e-6          # CL rank threshold, same default as production
SIGMA_OFFSET = 0.05

_RESULTS = []


def record(name, value, tol, mode='below'):
    """Register one numerical check and print a PASS/FAIL line."""
    if mode == 'below':
        ok = bool(value < tol)
        rel = '<'
    else:
        ok = bool(value > tol)
        rel = '>'
    _RESULTS.append((name, ok, value, tol))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {value:.3e} {rel} {tol:.1e}')
    return ok


# ---------------------------------------------------------------------------
# linear-algebra helpers (candidate coordinates: the metric is the identity)
# ---------------------------------------------------------------------------

def orth(X, rtol=1e-10):
    """Orthonormal basis of the column space of X, with a relative rank cut."""
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0], 0))
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros((X.shape[0], 0))
    r = int(np.sum(s > rtol * max(s[0], 1.0)))
    return U[:, :r]


def projector(X):
    Q = orth(X)
    return Q @ Q.T.conj()


def subspace_dist(A, B):
    """||P_A - P_B||_F.  Zero iff the two column spaces coincide."""
    return float(np.linalg.norm(projector(A) - projector(B)))


def subspace_overlap(A, B):
    """||P_A P_B||_F / sqrt(min(k_A, k_B)) in [0, 1]; 1 = nested spaces."""
    QA, QB = orth(A), orth(B)
    k = min(QA.shape[1], QB.shape[1])
    if k == 0:
        return float('nan')
    return float(np.linalg.norm(QA.T.conj() @ QB) / np.sqrt(k))


# ---------------------------------------------------------------------------
# independent reference implementation of the CL recursion
# ---------------------------------------------------------------------------
# Everything is done in "candidate coordinates": a vector x means the orbital
# cand_AO @ x.  Since cand_AO^dag S cand_AO = 1 these coordinates are
# orthonormal, and the production coupling block C_n^dag O C_ker,n equals
# X_n^dag (cand_AO^dag O cand_AO) X_ker,n exactly.  The two implementations are
# therefore algebraically equivalent and any disagreement is a bug.

def make_fake_mol(mol, atoms_A, proj_bas, verbose=0):
    fake = gto.Mole()
    fake.verbose = verbose
    fake.unit = 'Bohr'
    fake.symmetry = False
    fake.atom = [(mol.atom_symbol(i), mol.atom_coord(i, unit='Bohr'))
                 for i in atoms_A]
    fake.basis = proj_bas
    fake.charge = 0
    fake.spin = 0
    try:
        fake.build(False, False)
    except RuntimeError:
        fake.spin = 1
        fake.build(False, False)
    return fake


def shell_zero(mol, cand_AO, atoms_A, proj_bas, threshold=THRESHOLD):
    """Projection-basis shell 0, in candidate coordinates."""
    fake = make_fake_mol(mol, atoms_A, proj_bas)
    s_pb = fake.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake, mol)
    c_prime = np.linalg.inv(s_pb) @ s_cross @ cand_AO
    _, sig, Vh = np.linalg.svd(c_prime.T.conj() @ s_cross @ cand_AO,
                               full_matrices=True)
    r0 = int(np.sum(sig > threshold)) if sig.size else 0
    return Vh[:r0, :].T.conj(), Vh[r0:, :].T.conj()


def cl_recursion(A, X0, Xker, n_shell, threshold=THRESHOLD):
    """Reference CL shell recursion for coupling operator A.

    Returns (shells, kernel) with shells[0] = X0.
    """
    shells = [X0]
    ker = Xker
    for _ in range(n_shell):
        prev = shells[-1]
        if prev.shape[1] == 0 or ker.shape[1] == 0:
            shells.append(np.zeros((X0.shape[0], 0)))
            continue
        B = prev.T.conj() @ A @ ker
        _, sig, Vh = np.linalg.svd(B, full_matrices=True)
        r = int(np.sum(sig > threshold)) if sig.size else 0
        shells.append(ker @ Vh[:r, :].T.conj())
        ker = ker @ Vh[r:, :].T.conj()
    return shells, ker


def cumulative(shells, upto=None):
    sel = shells if upto is None else shells[:upto]
    sel = [s for s in sel if s.shape[1] > 0]
    if not sel:
        return np.zeros((0, 0))
    return np.hstack(sel)


def resolvent_matrix(F_cand, sigma, normalize=True, stable=True):
    """Reference construction of the shift-and-invert coupling operator.

    ``stable=False`` builds the affine [0, 1] normalization the naive way,
    straight from the weights.  That form is exact at moderate |sigma| and is
    used to cross-check the cancellation-free closed form the production code
    uses (T5).  ``stable=True`` mirrors production and is what the sigma sweep
    needs, since the naive form loses all significance for large |sigma|.
    """
    e, V = np.linalg.eigh(F_cand)
    w = 1.0 / (e - sigma)
    if not normalize:
        wn = w
    elif stable:
        wn = concentric_loc.resolvent_weights(e, sigma, normalize=True)
    else:
        wn = (w - np.min(w)) / (np.max(w) - np.min(w))
    M = (V * wn) @ V.T.conj()
    return 0.5 * (M + M.T.conj()), e, w


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------

def build_reference():
    mol = gto.M(
        atom='''
        C   0.0000   0.0000   0.7680
        C   0.0000   0.0000  -0.7680
        H   0.0000   1.0192   1.1573
        H  -0.8825  -0.5096   1.1573
        H   0.8825  -0.5096   1.1573
        H   0.0000  -1.0192  -1.1573
        H   0.8825   0.5096  -1.1573
        H  -0.8825   0.5096  -1.1573
        ''',
        basis='6-31g', symmetry=0, verbose=3)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, mf


def fresh_dmet(mf, imp_idx):
    dmet = ssdmet.SSDMET(mf, title=TITLE, imp_idx=imp_idx,
                         threshold=1e-12, es_natorb=False, verbose=3)
    dmet.build(save_chk=False)
    return dmet


def fv_block(dmet):
    """(cand_AO, nimp, nbath) for the frozen-virtual candidate block."""
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    cand_lo = dmet.lo_cloes[:, nimp + nbath + dmet.nfo:]
    return dmet.caolo @ cand_lo, nimp, nbath


def fock_ao_of(dmet):
    return dmet.mf_or_cas.get_fock(dm=dmet.mf_or_cas.make_rdm1())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def main():
    mol, mf = build_reference()
    imp_idx = mol.search_ao_label([r'^0\s+C\b'])
    print(f'\nfull-system RHF   E = {mf.e_tot:.12f} Ha')
    print(f'impurity AOs      = {list(imp_idx)}')

    dmet0 = fresh_dmet(mf, imp_idx)
    cand_AO, nimp, nbath = fv_block(dmet0)
    ncand = cand_AO.shape[1]
    fock = fock_ao_of(dmet0)
    ovlp = mol.intor_symmetric('int1e_ovlp')

    F_cand = cand_AO.T.conj() @ fock @ cand_AO
    F_cand = 0.5 * (F_cand + F_cand.T.conj())
    e_cand = np.linalg.eigvalsh(F_cand)
    X0, Xker0 = shell_zero(mol, cand_AO, ATOMS_A, PROJ_BAS)

    print(f'\nNImp = {nimp}, NBath = {nbath}, NFO = {dmet0.nfo}, '
          f'NFV = {ncand}')
    print(f'FV pseudo-canonical eps in [{e_cand[0]:.4f}, {e_cand[-1]:.4f}] Ha')
    print(f'shell 0: {X0.shape[1]} vectors, kernel: {Xker0.shape[1]}')

    e_homo = float(np.max(mf.mo_energy[mf.mo_occ > 0]))
    sigma_default = e_homo
    print(f'eps_HOMO = {e_homo:.6f} Ha  (default sigma, virtual side)')

    # --------------------------------------------------------------- T5, T4
    print('\n=== T5  AO-representation round trip ===')
    R_ao, sigma_used, e_from_prod = concentric_loc.build_resolvent_op(
        dmet0, fock, cand_AO, side='vir', sigma=None,
        sigma_offset=SIGMA_OFFSET)
    M_ref, e_ref, w_ref = resolvent_matrix(F_cand, sigma_used)
    # naive vs cancellation-free construction of the same affine map: this is
    # what makes the closed form in resolvent_weights() a checked claim
    M_naive, _, _ = resolvent_matrix(F_cand, sigma_used, stable=False)
    record('||M_closed_form - M_naive||',
           float(np.linalg.norm(M_ref - M_naive)), 1e-12)
    record('sigma matches eps_HOMO', abs(sigma_used - sigma_default), 1e-12)
    record('||eps_prod - eps_ref||', float(np.linalg.norm(e_from_prod - e_ref)),
           1e-12)
    M_prod = cand_AO.T.conj() @ R_ao @ cand_AO
    record('||C^dag R_ao C - M_ref||', float(np.linalg.norm(M_prod - M_ref)),
           1e-9)
    gram_err = float(np.linalg.norm(
        cand_AO.T.conj() @ ovlp @ cand_AO - np.eye(ncand)))
    record('||C^dag S C - 1||', gram_err, 1e-10)

    print('\n=== T4  definiteness and order reversal ===')
    record('min 1/(eps - sigma)', float(np.min(w_ref)), 0.0, mode='above')
    # eigh returns eps ascending, so with sigma below the spectrum the
    # resolvent weights must come out non-increasing: the resolvent ranks the
    # candidates in exactly the reverse Fock order.  Stated as a monotonicity
    # check rather than an argsort comparison, which would be ambiguous for
    # the degenerate manifolds that symmetric molecules always have.
    record('max d(1/(eps - sigma))', float(np.max(np.diff(w_ref))), 1e-14)

    # ------------------------------------------------------------------- T1
    print('\n=== T1  Krylov identity, span{C_0..C_n} = K_{n+1}(A, C_0) ===')
    for label, A in (('fock', F_cand), ('shift_invert', M_ref)):
        shells, _ = cl_recursion(A, X0, Xker0, 3)
        for n in range(1, 4):
            krylov = [X0]
            for _ in range(n):
                krylov.append(A @ krylov[-1])
            d = subspace_dist(cumulative(shells, n + 1), np.hstack(krylov))
            record(f'{label}: shells 0..{n} vs K_{n+1}', d, 1e-6)

    # ------------------------------------------------------------------- T2
    print('\n=== T2  exhausted Krylov space is operator-independent ===')
    sh_f, ker_f = cl_recursion(F_cand, X0, Xker0, ncand)
    sh_r, ker_r = cl_recursion(M_ref, X0, Xker0, ncand)
    cum_f, cum_r = cumulative(sh_f), cumulative(sh_r)
    print(f'  exhausted: fock -> {cum_f.shape[1]} orbitals '
          f'(kernel {ker_f.shape[1]}), '
          f'shift_invert -> {cum_r.shape[1]} (kernel {ker_r.shape[1]})')
    record('rank difference', float(abs(cum_f.shape[1] - cum_r.shape[1])), 0.5)
    record('||P_fock - P_shift_invert|| (exhausted)',
           subspace_dist(cum_f, cum_r), 1e-8)
    # and the ORDER must differ, otherwise the method does nothing new
    shell_overlaps = [subspace_overlap(sh_f[i], sh_r[i])
                      for i in range(1, min(4, len(sh_f)))
                      if sh_f[i].shape[1] and sh_r[i].shape[1]]
    print(f'  per-shell overlap fock vs shift_invert (shells 1..): '
          f'{np.array(shell_overlaps)}')
    if shell_overlaps:
        record('leading shells actually differ (1 - overlap)',
               1.0 - float(min(shell_overlaps)), 1e-6, mode='above')

    # ------------------------------------------------------------------- T3
    print('\n=== T3  sigma -> -inf recovers plain Fock CL ===')
    # The deviation is FIRST order in tau = -1/sigma: with the eigenvalue range
    # normalized to [0, 1], M -> (eps_max - F)/(eps_max - eps_min) + O(tau),
    # an affine function of F, which leaves the shell decomposition invariant.
    # So the right assertion is the RATE (log-log slope -1), not an absolute
    # floor -- reaching 1e-6 would need |sigma| ~ 1e6 for no good reason.
    n_probe = 3
    sh_fock, _ = cl_recursion(F_cand, X0, Xker0, n_probe)
    sigmas, dists = [], []
    for sigma in (-1e0, -1e1, -1e2, -1e3, -1e4, -1e5, -1e6):
        M_s, _, _ = resolvent_matrix(F_cand, sigma)
        sh_s, _ = cl_recursion(M_s, X0, Xker0, n_probe)
        sizes = [sh_s[i].shape[1] for i in range(n_probe + 1)]
        d = max(subspace_dist(sh_fock[i], sh_s[i])
                for i in range(1, n_probe + 1))
        sigmas.append(abs(sigma))
        dists.append(d)
        print(f'  sigma = {sigma:>9.1e}   max shell distance = {d:.3e}'
              f'   shell sizes = {sizes}')
    print(f'  Fock shell sizes = '
          f'{[sh_fock[i].shape[1] for i in range(n_probe + 1)]}')

    record('shell distance at sigma = -1e6', dists[-1], 1e-5)
    record('monotone decrease over the whole sweep',
           float(max(np.diff(dists))), 0.0)
    slope = float(np.polyfit(np.log10(sigmas), np.log10(dists), 1)[0])
    print(f'  log-log convergence slope = {slope:.3f}  (theory: -1)')
    record('|slope + 1| (first-order rate)', abs(slope + 1.0), 0.3)

    # ------------------------------------------------------------------ T10
    print('\n=== T10  negative control: sigma inside the spectrum ===')
    bad_sigma = float(0.5 * (e_cand[0] + e_cand[-1]))
    _, sigma_fixed, _ = concentric_loc.build_resolvent_op(
        dmet0, fock, cand_AO, side='vir', sigma=bad_sigma,
        sigma_offset=SIGMA_OFFSET)
    print(f'  requested sigma = {bad_sigma:.6f} (inside '
          f'[{e_cand[0]:.4f}, {e_cand[-1]:.4f}])  ->  relocated to '
          f'{sigma_fixed:.6f}')
    record('relocated sigma is below the spectrum',
           float(e_cand[0] - sigma_fixed), 0.0, mode='above')

    # --------------------------------------------------------- T6, T7, T8, T9
    print('\n=== T6/T7/T8  production run: exactness, basis, span ===')
    promoted_eps = {}
    for couple_op in ('fock', 'shift_invert'):
        A = F_cand if couple_op == 'fock' else M_ref
        for n_shell in (1, 2, 3):
            dmet = fresh_dmet(mf, imp_idx)
            nes_before = dmet.nes
            concentric_loc.concentric_localization(
                dmet, proj_bas=PROJ_BAS, n_shell=n_shell, atoms_A=ATOMS_A,
                couple_op=couple_op, threshold=THRESHOLD,
                sigma=None, sigma_offset=SIGMA_OFFSET)
            n_new = dmet.nes - nes_before
            tag = f'{couple_op}/n_shell={n_shell}'
            print(f'\n  --- {tag}: promoted {n_new} orbitals '
                  f'(NES {nes_before} -> {dmet.nes}) ---')

            # T6 HF-in-HF exactness -- must hold at EVERY shell count
            dev = abs(dmet.es_mf.e_tot + dmet.fo_ene - mf.e_tot)
            record(f'{tag}: |E_es + E_fo - E_HF|', float(dev), 1e-8)

            # T7 basis integrity
            n_lo = dmet.lo_cloes.shape[0]
            gram = dmet.lo_cloes.T.conj() @ dmet.lo_cloes
            record(f'{tag}: ||lo_cloes^dag lo_cloes - 1||',
                   float(np.linalg.norm(gram - np.eye(n_lo))), 1e-10)
            record(f'{tag}: NES + NFO + NFV - nao',
                   float(abs(dmet.nes + dmet.nfo + dmet.nfv - n_lo)), 0.5)

            # T8 promoted span vs the reference recursion
            prom_lo = dmet.lo_cloes[:, nimp + nbath: nimp + nbath + n_new]
            prom_cand = cand_AO.T.conj() @ ovlp @ (dmet.caolo @ prom_lo)
            shells, _ = cl_recursion(A, X0, Xker0, n_shell)
            record(f'{tag}: ||P_prod - P_ref||',
                   subspace_dist(prom_cand, cumulative(shells)), 1e-8)

            eps = np.sort(np.linalg.eigvalsh(
                prom_cand.T.conj() @ F_cand @ prom_cand))
            promoted_eps[(couple_op, n_shell)] = eps

    # ------------------------------------------------------------------- T9
    print('\n=== T9  eps distribution of the promoted orbitals ===')
    e_mid = 0.5 * (e_cand[0] + e_cand[-1])
    print(f'  FV spectrum [{e_cand[0]:.4f}, {e_cand[-1]:.4f}], '
          f'midpoint {e_mid:.4f} Ha')
    print(f'  {"case":24s} {"n":>3s} {"mean eps":>10s} '
          f'{"frac above mid":>15s}')
    for (couple_op, n_shell), eps in sorted(promoted_eps.items()):
        if eps.size == 0:
            continue
        frac = float(np.mean(eps > e_mid))
        print(f'  {couple_op + "/n=" + str(n_shell):24s} {eps.size:3d} '
              f'{np.mean(eps):10.4f} {frac:15.2f}')
    print('\n  prediction: Fock CL has a sizeable fraction above the midpoint')
    print('  (Lanczos converges on both spectral extremes); shift-invert CL')
    print('  should be near zero there.  This is the physics claim, not a')
    print('  correctness test -- no PASS/FAIL is asserted.')

    # ------------------------------------------------------------------ done
    n_fail = sum(1 for _, ok, _, _ in _RESULTS if not ok)
    print(f'\n{"=" * 68}')
    print(f'{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed')
    if n_fail:
        print('FAILED:')
        for name, ok, value, tol in _RESULTS:
            if not ok:
                print(f'  {name}: {value:.6e} (tol {tol:.1e})')
    print('=' * 68)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
