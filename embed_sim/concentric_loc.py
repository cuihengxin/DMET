import numpy as np
from pyscf import lib, gto

# ---------------------------------------------------------------------------
# Shift-and-invert ("resolvent") coupling operator
# ---------------------------------------------------------------------------
# The CL recursion only ever probes the coupling operator restricted to the
# candidate block: shell 0 and every kernel live inside the initial FV (or FO)
# space, so the cumulative shells span the block Krylov space
# K_n(P O P, C_0) with P the projector onto that block.
#
# With O = Fock the Krylov filter is a polynomial in the pseudo-canonical
# orbital energy eps, and Lanczos-type convergence reaches the EXTREMES of the
# spectrum first.  On the virtual side that means roughly half of the promoted
# slots are spent on the highest-lying, most diffuse candidates -- the ones
# with the largest MP2 energy denominators and hence the smallest correlation
# weight.  This is the "wasted slots" failure mode of a denominator-free
# (alpha = 0) selection criterion.
#
# Replacing the coupling operator by the resolvent of the same restricted Fock
# matrix,
#
#       R = (P F P - sigma)^{-1} ,
#
# turns the filter into a polynomial in 1/(eps - sigma), i.e. it puts an energy
# denominator into an otherwise denominator-free selection.  R shares its
# eigenvectors with P F P and has eigenvalues 1/(eps_a - sigma), so with sigma
# placed strictly OUTSIDE the candidate spectrum:
#
#   * side='vir': sigma below the spectrum (default: eps_HOMO) => all
#     1/(eps - sigma) > 0 and decreasing in eps, so shift-and-invert Lanczos
#     converges first on the LOW-lying virtuals, i.e. the small denominators.
#   * side='occ': sigma above the spectrum (default: eps_LUMO) => |1/(eps -
#     sigma)| grows with eps, so it converges first on the SHALLOW occupied
#     orbitals, again the small-denominator end.
#
# Two structural properties make this cheap to verify (see the test script):
#   (i)  R = g(P F P) with g injective on the spectrum, so the *exhausted*
#        Krylov space is identical to the Fock one -- only the ORDER in which
#        orbitals are promoted changes;
#   (ii) sigma -> -inf (virtual side) recovers the plain Fock CL shell by
#        shell, since (F - sigma)^{-1} = tau(1 + tau F + ...) with
#        tau = -1/sigma.  sigma is therefore a one-parameter family
#        interpolating between denominator-free CL and a denominator-weighted
#        selection, at CL cost (one small eigendecomposition, no correlated
#        calculation).

SI_ALIASES = ('shift_invert', 'shift-invert', 'resolvent', 'si')


def _get_ovlp(dmet):
    getter = getattr(dmet.mf_or_cas, 'get_ovlp', None)
    if getter is not None:
        return getter()
    return dmet.mol.intor_symmetric('int1e_ovlp')


def _reference_orbital_energies(mf_or_cas):
    """(eps_HOMO, eps_LUMO) of the DMET reference, or (None, None)."""
    mo_energy = getattr(mf_or_cas, 'mo_energy', None)
    mo_occ = getattr(mf_or_cas, 'mo_occ', None)
    if mo_energy is None or mo_occ is None:
        return None, None
    mo_energy = np.asarray(mo_energy)
    mo_occ = np.asarray(mo_occ)
    # spin-resolved (2, nmo) references are not handled here
    if mo_energy.ndim != 1 or mo_occ.shape != mo_energy.shape:
        return None, None
    occ = mo_energy[mo_occ > 0]
    vir = mo_energy[mo_occ == 0]
    e_homo = float(np.max(occ)) if occ.size else None
    e_lumo = float(np.min(vir)) if vir.size else None
    return e_homo, e_lumo


def _resolve_sigma(sigma, e_cand, side, mf_or_cas, sigma_offset, log):
    """Pick sigma and force it strictly outside the candidate spectrum."""
    e_min, e_max = float(np.min(e_cand)), float(np.max(e_cand))
    edge = e_min - sigma_offset if side == 'vir' else e_max + sigma_offset

    if sigma is None:
        sigma = 'auto'
    if isinstance(sigma, str):
        key = sigma.lower()
        if key == 'edge':
            sigma = edge
        elif key in ('auto', 'homo', 'lumo'):
            if key == 'auto':
                key = 'homo' if side == 'vir' else 'lumo'
            e_homo, e_lumo = _reference_orbital_energies(mf_or_cas)
            ref = e_homo if key == 'homo' else e_lumo
            if ref is None:
                log.warn('reference orbital energies unavailable; '
                         'using sigma="edge" instead of sigma="%s"', key)
                sigma = edge
            else:
                sigma = ref
        else:
            raise ValueError(
                f"unknown sigma specification {sigma!r}; expected a float or "
                "one of 'auto', 'homo', 'lumo', 'edge'")
    sigma = float(sigma)

    # A sigma inside the candidate spectrum makes the resolvent indefinite and
    # near-singular: one candidate dominates and the selection turns erratic
    # (the intruder-state pathology shift-and-invert is meant to avoid).
    if side == 'vir' and sigma >= e_min:
        log.warn('sigma = %.6f is not below the candidate spectrum '
                 '[%.6f, %.6f]; moving it to %.6f', sigma, e_min, e_max, edge)
        sigma = edge
    elif side == 'occ' and sigma <= e_max:
        log.warn('sigma = %.6f is not above the candidate spectrum '
                 '[%.6f, %.6f]; moving it to %.6f', sigma, e_min, e_max, edge)
        sigma = edge
    return sigma


def resolvent_weights(e_cand, sigma, normalize=True):
    """Eigenvalues of the shift-and-invert coupling operator.

    Unnormalized these are the plain resolvent weights ``1/(eps - sigma)``.
    Normalized they are the affine map of those weights onto ``[0, 1]``,

        (w_a - w_min) / (w_max - w_min)
            = (eps_max - eps_a)(eps_min - sigma)
              / [ (eps_a - sigma)(eps_max - eps_min) ] ,

    which is evaluated in the second, algebraically equivalent form on
    purpose.  The first form subtracts two numbers of size ``1/|sigma|`` whose
    difference is of size ``1/sigma^2``, so it loses all significance for large
    ``|sigma|``; in the second form sigma appears only inside well-conditioned
    factors and the expression is stable for arbitrarily large ``|sigma|``.

    The map has positive slope on both sides (for side='occ' all weights are
    negative and ``w_max = w(eps_min)`` still), so it is an ``alpha*w + beta``
    transform with ``alpha > 0`` and leaves the CL shell decomposition exactly
    invariant -- see the ``normalize`` note in :func:`build_resolvent_op`.
    """
    e_cand = np.asarray(e_cand, dtype=float)
    w = 1.0 / (e_cand - sigma)
    if not normalize:
        return w
    e_lo, e_hi = float(np.min(e_cand)), float(np.max(e_cand))
    span = e_hi - e_lo
    if span <= 1e-14 * max(1.0, abs(e_hi)):
        # Degenerate candidate spectrum: the restricted Fock matrix is a
        # multiple of the identity, so every coupling block vanishes and no
        # normalization can help.  Hand back the raw weights.
        return w
    return (e_hi - e_cand) * (e_lo - sigma) / ((e_cand - sigma) * span)


def build_resolvent_op(dmet, fock_ao, cand_AO, side, sigma=None,
                       sigma_offset=0.05, normalize=True, check=True):
    """AO-basis matrix of the shift-and-invert coupling operator.

    Returns ``(R_ao, sigma, e_cand)`` where ``R_ao`` satisfies

        C_i^dag R_ao C_j = [ scale * (F_cand - sigma)^{-1} ]_ij

    for any orbitals ``C_i``, ``C_j`` expanded in the candidate block
    ``cand_AO``.  The identity relies on ``cand_AO^dag S cand_AO = 1``, which
    is checked and reported.

    Parameters
    ----------
    cand_AO : (nao, ncand) array
        The FULL initial candidate block (FV for side='vir', FO for
        side='occ').  Passing the full block keeps the Krylov operator fixed
        over the whole recursion, as it is for couple_op='fock'.
    side : {'vir', 'occ'}
    sigma : float or {'auto', 'homo', 'lumo', 'edge'}
    sigma_offset : float
        Offset used by sigma='edge' and by the safety relocation, in Hartree.
    normalize : bool
        Affinely map the resolvent eigenvalues onto [0, 1].  The CL recursion
        is EXACTLY invariant under ``M -> alpha*M + beta*I`` with alpha > 0:
        the identity part drops out of every shell SVD because shell and
        kernel are mutually orthogonal, so only the singular VALUES are
        rescaled, never the singular vectors.  What the rescaling fixes is the
        comparison against the absolute rank threshold.  Scaling to unit
        spectral norm is NOT enough: as |sigma| grows all 1/(eps - sigma)
        become equal, M tends to the identity, and the informative part -- the
        deviation from the identity -- still collapses like 1/|sigma| until the
        threshold silently truncates whole shells.  Mapping the eigenvalue
        RANGE to [0, 1] keeps that informative part at O(1) for every sigma.
    """
    log = dmet.log
    if side not in ('vir', 'occ'):
        raise ValueError(f"side must be 'vir' or 'occ', got {side!r}")
    if cand_AO.shape[1] == 0:
        raise ValueError(f'empty {side} candidate space, nothing to expand')

    ovlp = _get_ovlp(dmet)

    gram = cand_AO.T.conj() @ ovlp @ cand_AO
    gram_err = np.linalg.norm(gram - np.eye(gram.shape[0]))
    if gram_err > 1e-8:
        log.warn('candidate block not orthonormal: ||C^dag S C - 1|| = %.3e',
                 gram_err)

    F_cand = cand_AO.T.conj() @ fock_ao @ cand_AO
    F_cand = 0.5 * (F_cand + F_cand.T.conj())
    e_cand, V_cand = np.linalg.eigh(F_cand)

    sigma = _resolve_sigma(sigma, e_cand, side, dmet.mf_or_cas,
                           sigma_offset, log)

    denom = e_cand - sigma
    weight = 1.0 / denom
    weight_used = resolvent_weights(e_cand, sigma, normalize=normalize)
    M = (V_cand * weight_used) @ V_cand.T.conj()
    M = 0.5 * (M + M.T.conj())

    SC = ovlp @ cand_AO
    R_ao = SC @ M @ SC.T.conj()

    log.info('')
    log.info('=== shift-and-invert coupling operator (side = %s) ===', side)
    log.info('candidate space: %d orbitals, eps in [%.6f, %.6f] Ha',
             len(e_cand), e_cand[0], e_cand[-1])
    log.info('sigma = %.6f Ha,  min|eps - sigma| = %.6f Ha',
             sigma, float(np.min(np.abs(denom))))
    log.info('filter dynamic range max|1/(eps-sigma)| / min = %.3f',
             float(np.max(np.abs(weight)) / np.min(np.abs(weight))))
    log.info('normalized weights in [%.4f, %.4f]',
             float(np.min(weight_used)), float(np.max(weight_used)))
    order = np.argsort(-np.abs(weight))
    log.info('top candidates by |1/(eps-sigma)|, their eps = %s',
             np.array2string(e_cand[order][:10], precision=4))
    if check:
        rep_err = np.linalg.norm(cand_AO.T.conj() @ R_ao @ cand_AO - M)
        log.info('AO representation error ||C^dag R C - M|| = %.3e', rep_err)
    log.info('')
    return R_ao, sigma, e_cand


def concentric_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6,
                            sigma=None, sigma_offset=0.05):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    # The body below rebinds the name `sigma` to SVD singular values, so keep
    # the shift-and-invert parameter under its own name before that happens.
    sigma_shift = sigma

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    # Build a fake molecule from selected atoms with a small projection basis.
    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    fake_mol.spin = 0 if fake_mol.nelectron % 2 == 0 else 1
    fake_mol.charge = 0 # here may be error for the sake of open shell systems
    fake_mol.build(False, False)
    
    slices = fake_mol.aoslice_by_atom()
    for ia in range(fake_mol.natm):
        ao_start = slices[ia][2]
        ao_end = slices[ia][3]
        num_ao = ao_end - ao_start  # AO number for this atom
    
        symbol = fake_mol.atom_symbol(ia)
        print(f" atom {ia} ({symbol}) with {num_ao:2d} basis functions from ({ao_start:2d} to {ao_end-1:2d})")

    print(f"Total basis functions (fake_mol.nao): {fake_mol.nao}")

    print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
    print(f"Fake molecule: {fake_mol.atom}")
    #s_wb = self.mol.intor_symmetric('int1e_ovlp')
    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol) 
    s_pb_inv = np.linalg.inv(s_pb)
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    fv_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]

    def svd(coeff, couple_op, coeff_ker):
        ovlp = coeff.T.conj() @ couple_op @ coeff_ker #  ovlp may use other types of coupling operators, not limited to h_core
        # MUST use full_matrices=True here to get the complete right singular vectors 

        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        dmet.log.info(f"Singular values for shell {i}: {sigma}")
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span  = Vh[:r, :].T.conj()
        V_ker = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1
    c_fv_prime = s_pb_inv @ s_cross @ fv_AO
    # space sizes
    dmet.log.info(f"fv_AO shape: {fv_AO.shape}")
    dmet.log.info(f"s_cross shape: {s_cross.shape}")
    dmet.log.info(f"c_fv_prime shape: {c_fv_prime.shape}")

    U, sigma, Vh = np.linalg.svd(c_fv_prime.T.conj() @ s_cross @ fv_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    dmet.log.info(f"SVD on projector rank r0: {r0}, sigma size: {len(sigma)}")
    dmet.log.info(f"Singular values: {sigma}")
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_0 = fv_AO @ V_span
    C_ker0 = fv_AO @ V_ker
    C_vir = []
    C_vir.append(C_0)
    C_ker = []
    C_ker.append(C_ker0)

    # pseudo-canonicalize the vir space by diagonalizing the Fock matrix in this subspace
    dm = dmet.mf_or_cas.make_rdm1()
    fock_ao = dmet.mf_or_cas.get_fock(dm=dm) 

    dmet.log.info(f"Shell 0: {C_0.shape[1]} vectors in vir space, {C_ker0.shape[1]} vectors in ker space.")

    # Built once, from the full initial FV block, so that the Krylov operator
    # stays fixed over the recursion (as it is for 'hcore' / 'fock').
    resolvent_ao, sigma_used = None, None
    if couple_op in SI_ALIASES:
        resolvent_ao, sigma_used, _ = build_resolvent_op(
            dmet, fock_ao, fv_AO, side='vir',
            sigma=sigma_shift, sigma_offset=sigma_offset)

    for i in range(n_shell):
        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
            dmet.log.info(f"Using Hcore as coupling operator for shell {i+1}")
        elif couple_op == 'fock':
            couple_matrix = fock_ao
            dmet.log.info(f"Using Fock matrix as coupling operator for shell {i+1}")
        elif couple_op in SI_ALIASES:
            couple_matrix = resolvent_ao
            dmet.log.info(f"Using shift-and-invert resolvent (sigma = {sigma_used:.6f} Ha) "
                          f"as coupling operator for shell {i+1}")
        else:
            raise ValueError(f"unknown couple_op {couple_op!r}; expected "
                             f"'hcore', 'fock' or one of {SI_ALIASES}")
        new_vir, new_ker = svd(C_vir[i], couple_matrix, C_ker[i])
        C_vir.append(new_vir)
        C_ker.append(new_ker)
        dmet.log.info(f"Shell {i+1}: {new_vir.shape[1]} new vectors added to vir space, {new_ker.shape[1]} vectors remain in ker space.")
        dmet.log.info(f"======Shell {i+1} concentric localized======")
    C_vir_matrix = np.hstack(C_vir)
    
    # Export density cube files for each shell. Noting that the density is calculated consider ing the orbs are occupied just for visualization.
    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_vir):
                if c_shell.shape[1] > 0:
                    dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                    cube_name = f"{dmet.title}_shell_{idx}_density.cube"
                    dmet.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    # Project Fock matrix into the vir subspace defined by C_vir_matrix
    # (N_vir, N_AO) @ (N_AO, N_AO) @ (N_AO, N_vir) -> (N_vir, N_vir)
    fock_sub = C_vir_matrix.T.conj() @ fock_ao @ C_vir_matrix

    # diagonalize the Fock matrix
    mo_energy, U = np.linalg.eigh(fock_sub)
    C_vir_canonical = C_vir_matrix @ U
    # Logged for the eps-distribution diagnostic: a denominator-free (Fock) CL
    # promotes both spectral extremes, a shift-and-invert CL concentrates on
    # the low-eps end.
    dmet.log.info('promoted FV pseudo-canonical energies (Ha): %s',
                  np.array2string(mo_energy, precision=4, max_line_width=100))

    C_fv_new = C_ker[-1]
    fock_fv = C_fv_new.T.conj() @ fock_ao @ C_fv_new
    mo_energy_fv, U_fv = np.linalg.eigh(fock_fv)
    C_fv_canonical = C_fv_new @ U_fv
    dmet.log.info('remaining FV pseudo-canonical energies (Ha): %s',
                  np.array2string(mo_energy_fv, precision=4, max_line_width=100))

    Q_emb = dmet.lo_cloes[:, :nimp+nbath]
    Q_fo  = dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
    
    lo2New_bath = dmet.cloao @ C_vir_canonical
    lo2New_fv   = dmet.cloao @ C_fv_canonical
    
    dmet.lo_cloes = np.hstack([Q_emb, lo2New_bath, Q_fo, lo2New_fv])
    
    n_shifted_fv = lo2New_bath.shape[1]
    dmet.nes += n_shifted_fv
    dmet.nfv -= n_shifted_fv
    
    dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
    dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes : dmet.nes+dmet.nfo])
    dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes+dmet.nfo :])

    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi'):
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
    dm_arg = dmet.dm_pair if (dmet.open_shell and dmet.dm_pair is not None) else dmet.dm
    dmet.es_dm = dmet.make_es_dm(dmet.open_shell, dmet.lo_cloes[:, :dmet.nes], dmet.cloao, dm_arg)

    dmet.es_mf = dmet.ROHF()
    dmet.calc_fo_ene()

    dmet.log.info(f"Concentric Shell appended. Added {n_shifted_fv} vir orbitals to bath.")
    dmet.log.info(f"New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; NImp={nimp}, NBATH = {dmet.nes - nimp}")
    return dmet
'''
    def concentric_occ_spade(self, atoms_A, threshold=1e-6):
        nimp = len(dmet.imp_idx)
        nbath = dmet.nes - nimp
        c_fo_lo = dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
        c_fo_ao = dmet.caolo @ c_fo_lo
        ao_indices_A = []
        for ia in atoms_A:
            atom_id, atom_symbol, start, end = dmet.mol.aoslice_by_atom()[ia]
            ao_indices_A.extend(range(start, end))
        Q_A = np.zeros((dmet.mol.nao, dmet.mol.nao))
        for idx in ao_indices_A:
            Q_A[idx, idx] = 1.0
        
        c_fo_lo_A = Q_A @ c_fo_lo
        u_A, sigma_A, vh_A = np.linalg.svd(c_fo_lo_A, full_matrices=True)
        print(f"Shape of c_fo_lo_A: {c_fo_lo_A.shape}")
        print(f"Singular values for c_fo_lo_A: {sigma_A}")
        print(f"U_A shape: {u_A.shape}, vh_A shape: {vh_A.shape}")
        print(f"Shape of V_A: {vh_A.T.conj().shape}")
        C_spade = c_fo_lo @ vh_A.T.conj()
        print(f"Shape of C_spade: {C_spade.shape}")
        
'''
def concentric_occ_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6,
                                sigma=None, sigma_offset=0.05):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    # See the note in concentric_localization: `sigma` is rebound to singular
    # values further down, so the shift is kept under a separate name.
    sigma_shift = sigma

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    # Build a fake molecule from selected atoms with a small projection basis.
    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    slices = fake_mol.aoslice_by_atom()
    for ia in range(fake_mol.natm):
        ao_start = slices[ia][2]
        ao_end = slices[ia][3]
        num_ao = ao_end - ao_start  # AO number for this atom
    
        symbol = fake_mol.atom_symbol(ia)
        print(f" atom {ia} ({symbol}) with {num_ao:2d} basis functions from ({ao_start:2d} to {ao_end-1:2d})")

    print(f"Total basis functions (fake_mol.nao): {fake_mol.nao}")
    fake_mol.spin = 0
    fake_mol.charge = 0 # here may be error for the sake of open shell systems
    if fake_mol.nelectron % 2 != 0:
        fake_mol.spin = 1 # a compromise but not affect the basis of fakemol
    fake_mol.build(False, False)
    print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
    
    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol) 
    s_pb_inv = np.linalg.inv(s_pb)
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    # NOTE: Here we operate on the frozen occupied (FO) orbitals instead of FV
    fo_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]

    def svd(coeff, couple_op, coeff_ker):
        ovlp = coeff.T.conj() @ couple_op @ coeff_ker
        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span  = Vh[:r, :].T.conj()
        V_ker = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1
        
    c_fo_prime = s_pb_inv @ s_cross @ fo_AO

    U, sigma, Vh = np.linalg.svd(c_fo_prime.T.conj() @ s_cross @ fo_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_0 = fo_AO @ V_span
    C_ker0 = fo_AO @ V_ker
    C_occ = [C_0]
    C_ker = [C_ker0]

    dm = dmet.mf_or_cas.make_rdm1()
    fock_ao = dmet.mf_or_cas.get_fock(dm=dm) 

    resolvent_ao, sigma_used = None, None
    if couple_op in SI_ALIASES:
        resolvent_ao, sigma_used, _ = build_resolvent_op(
            dmet, fock_ao, fo_AO, side='occ',
            sigma=sigma_shift, sigma_offset=sigma_offset)

    for i in range(n_shell):
        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
        elif couple_op == 'fock':
            couple_matrix = fock_ao
        elif couple_op in SI_ALIASES:
            couple_matrix = resolvent_ao
            dmet.log.info(f"Using shift-and-invert resolvent (sigma = {sigma_used:.6f} Ha) "
                          f"as coupling operator for shell {i+1}")
        else:
            raise ValueError(f"unknown couple_op {couple_op!r}; expected "
                             f"'hcore', 'fock' or one of {SI_ALIASES}")

        new_occ, new_ker = svd(C_occ[i], couple_matrix, C_ker[i])
        C_occ.append(new_occ)
        C_ker.append(new_ker)
        print(f"Shell {i+1}: {new_occ.shape[1]} new vectors added to occ space")
        
    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_occ):
                if c_shell.shape[1] > 0:
                    dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                    cube_name = f"{dmet.title}_shell_{idx}_density_occ.cube"
                    dmet.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    C_occ_matrix = np.hstack(C_occ)
    
    fock_sub = C_occ_matrix.T.conj() @ fock_ao @ C_occ_matrix
    mo_energy, U = np.linalg.eigh(fock_sub)
    C_occ_canonical = C_occ_matrix @ U
    dmet.log.info('promoted FO pseudo-canonical energies (Ha): %s',
                  np.array2string(mo_energy, precision=4, max_line_width=100))

    C_fo_new = C_ker[-1]
    fock_fo = C_fo_new.T.conj() @ fock_ao @ C_fo_new
    mo_energy_fo, U_fo = np.linalg.eigh(fock_fo)
    C_fo_canonical = C_fo_new @ U_fo
    dmet.log.info('remaining FO pseudo-canonical energies (Ha): %s',
                  np.array2string(mo_energy_fo, precision=4, max_line_width=100))

    Q_emb = dmet.lo_cloes[:, :nimp+nbath]
    Q_fv  = dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]
    
    lo2New_bath = dmet.cloao @ C_occ_canonical
    lo2New_fo   = dmet.cloao @ C_fo_canonical
    
    # Reassemble logic for FO
    # Sequence: [Emb, Target_FO (shifted to bath), Remaining_FO, FV]
    dmet.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_fo, Q_fv])
    
    n_shifted_fo = lo2New_bath.shape[1]
    dmet.nes += n_shifted_fo
    dmet.nfo -= n_shifted_fo
    
    dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
    dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes : dmet.nes+dmet.nfo])
    dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes+dmet.nfo :])

    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi'):
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
    dm_arg = dmet.dm_pair if (dmet.open_shell and dmet.dm_pair is not None) else dmet.dm
    dmet.es_dm = dmet.make_es_dm(dmet.open_shell, dmet.lo_cloes[:, :dmet.nes], dmet.cloao, dm_arg)
    
    dmet.es_mf = dmet.ROHF()
    dmet.calc_fo_ene() 
    
    dmet.log.info(f"Concentric FO Shell appended. Added {n_shifted_fo} occ orbitals to bath.")
    dmet.log.info(f"New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; NImp={nimp}, NBATH = {dmet.nes - nimp}")

    return dmet

def localize_environment_spaces(dmet, method='boys'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")            
    dmet.log.info(f"Performing {method.upper()} localization on Env subspaces (Bath, FO, FV)...")
    
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    # MO coeff in AO bases
    bath_AO = dmet.caolo @ dmet.lo_cloes[:, nimp : nimp+nbath]
    fo_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo]
    fv_AO   = dmet.caolo @ dmet.lo_cloes[:, nimp+nbath+dmet.nfo :]
    # note that, bath space may have some occ orbitals, may be some issues.
    from pyscf import lo
    def localize_subspace(coeff_AO, name):
        if coeff_AO.shape[1] <= 1:
            return coeff_AO  
        try:
            if method.lower() == 'boys':
                loc_obj = lo.Boys(dmet.mol, coeff_AO)
            elif method.lower() == 'pm':
                loc_obj = lo.PipekMezey(dmet.mol, coeff_AO)
            elif method.lower() == 'er':
                loc_obj = lo.EdmistonRuedenberg(dmet.mol, coeff_AO)
            else:
                dmet.log.warn(f"Unknown localization method {method}, skipping localization for {name}.")
                return coeff_AO
                
            loc_obj.verbose = 0
            return loc_obj.kernel()
        except Exception as e:
            dmet.log.warn(f"Localization failed for {name} subspace using {method}: {str(e)}")
            return coeff_AO
    
    bath_loc_AO = localize_subspace(bath_AO, "Bath")
    fo_loc_AO   = localize_subspace(fo_AO, "FO")
    fv_loc_AO   = localize_subspace(fv_AO, "FV")
    # from AO basis to LO basis
    dmet.lo_cloes[:, nimp : nimp+nbath] = dmet.cloao @ bath_loc_AO
    dmet.lo_cloes[:, nimp+nbath : nimp+nbath+dmet.nfo] = dmet.cloao @ fo_loc_AO
    dmet.lo_cloes[:, nimp+nbath+dmet.nfo :] = dmet.cloao @ fv_loc_AO
    # and refresh the fo fv orbitals in AO basis
    dmet.es_orb = dmet.caolo @ dmet.lo_cloes[:, :dmet.nes]
    dmet.fo_orb = fo_loc_AO
    dmet.fv_orb = fv_loc_AO
    
    dmet.log.info("Environment subspaces localized successfully.")
    return dmet

def eo_density(dmet):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before calculating density.")
    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    # Entangled space are selected
    c_es_lo = dmet.lo_cloes[:, :dmet.nes]
    c_es_ao = dmet.caolo @ c_es_lo
    dm_es = 2.0 * (c_es_ao @ c_es_ao.T.conj())
    from pyscf.tools import cubegen
    cube_name = f"{dmet.title}_es_density.cube"
    dmet.log.info(f"Exporting electron density of Embedded Space to {cube_name}")
    cubegen.density(dmet.mol, cube_name, dm_es)
    return dm_es