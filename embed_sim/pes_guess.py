"""
PES initial guess projection for DMET-embedded CASSCF calculations.

Projects converged CASSCF natural orbitals from one geometry to the next
in a potential energy surface scan. Only the active-space part is projected;
core and virtual orbitals come from the current geometry's SCF/AVAS,
eliminating bath-orbital projection errors that cause state-jumping.

Key insight:
  DMET bath orbitals are geometry-sensitive (derived from environment 1-RDM).
  Projecting the full embedded-space MO matrix drags bath projection artifacts
  into the new geometry and distorts the active orbital ordering.
  Instead, we project ONLY the active-space natural orbitals — which are
  atom-centered d/f orbitals that vary smoothly with geometry — while taking
  core and virtual orbitals fresh from the current geometry.

Pipeline for each active NO:
  prev embedded space --> prev full AO --> curr full AO --> curr embedded space
  (nes_prev, ncas)       (nao_prev,)      (nao_curr,)      (nes_curr, ncas)

Usage:
    from embed_sim.pes_guess import build_pes_guess, pes_scan_step

    # ── Method 1: build guess, then run CASSCF yourself ──
    mo_guess, quality = build_pes_guess(
        prev_cas, prev_dmet, mol_prev,
        curr_dmet, mol_curr,
        curr_ncas, curr_nelecas,
        aolabels=['Co 3d'],
        avas_kwargs={'minao': mol_prev._basis['Co'], 'threshold': 0.3,
                     'openshell_option': 2},
    )
    curr_cas.kernel(mo_guess)

    # ── Method 2: one-shot convenience (handles level-shift too) ──
    curr_cas = pes_scan_step(
        prev_cas, prev_dmet, mol_prev,
        curr_dmet, mol_curr,
        aolabels=['Co 3d'],
        avas_kwargs={'minao': ..., 'threshold': 0.3, 'openshell_option': 2},
        level_shift=0.3, ah_level_shift=1e-4,
        chkfile='point_02_sacasscf.chk',
    )
"""

import numpy as np
from pyscf import scf


# ──────────────────────────────────────────────────────────────────────
# Core building blocks
# ──────────────────────────────────────────────────────────────────────

def make_active_natorb(cas):
    """Extract active-space natural orbitals from a converged CASSCF.

    Diagonalizes the active-space 1-RDM and rotates the active MO columns
    into natural orbital order (descending occupation).  Natural orbitals
    are physically stable across geometries because occupation numbers
    vary smoothly — unlike canonical orbital energies which can cross.

    Args:
        cas: converged PySCF CASSCF object (must have .ci, .ncas, .nelecas,
             .ncore, .mo_coeff)

    Returns:
        occ_no: (ncas,)  natural orbital occupation numbers (descending)
        mo_active_no: (nmo_emb, ncas)  active NO coefficients in the
                      embedded MO basis
    """
    casdm1 = cas.fcisolver.make_rdm1(cas.ci, cas.ncas, cas.nelecas)
    occ_no, u_no = np.linalg.eigh(casdm1)
    # Sort descending by occupation
    idx = np.argsort(-occ_no)
    occ_no = occ_no[idx]
    u_no = u_no[:, idx]

    ncore = cas.ncore
    mo_active_no = cas.mo_coeff[:, ncore:ncore + cas.ncas] @ u_no
    return occ_no, mo_active_no


def project_active_mo(mo_active_emb, dmet_prev, mol_prev,
                       dmet_curr, mol_curr):
    """Project active-space MOs across two geometries via AO overlap.

    Three-step pipeline:
      1. prev embedded space → prev full AO  (via dmet_prev.es_orb)
      2. prev AO → curr AO                  (minimum-norm projection via
                                             scf.addons.project_mo_nr2nr)
      3. curr AO → curr embedded space      (via dmet_curr.es_orb^T @ S_curr)

    Args:
        mo_active_emb: (nes_prev, ncas)  active MOs in prev embedded basis
        dmet_prev: SSDMET object for previous geometry
        mol_prev: PySCF mol for previous geometry
        dmet_curr: SSDMET object for current geometry
        mol_curr: PySCF mol for current geometry

    Returns:
        mo_active_emb_curr: (nes_curr, ncas)  active MOs projected into
                            curr embedded basis
    """
    # Step 1: embedded → full AO (previous geometry)
    # es_orb: (nao_prev, nes_prev), mo_active_emb: (nes_prev, ncas)
    mo_active_ao_prev = dmet_prev.es_orb @ mo_active_emb

    # Step 2: project AO coefficients across geometries
    # project_mo_nr2nr uses overlap matrices internally to find
    # the minimum-norm projection of MOs from mol1 to mol2
    mo_active_ao_curr = scf.addons.project_mo_nr2nr(
        mol_prev, mo_active_ao_prev, mol_curr
    )

    # Step 3: full AO → embedded space (current geometry)
    S_curr = mol_curr.intor_symmetric('int1e_ovlp')
    # es_orb^T @ S maps from AO back to embedded space:
    #   <es_orb_i | psi_AO> = es_orb_i^T @ S @ C_AO
    mo_active_emb_curr = dmet_curr.es_orb.T @ S_curr @ mo_active_ao_curr

    return mo_active_emb_curr


def project_out(mo_target, mo_ref):
    """Remove reference-orbital components from target orbitals.

    For each reference orbital |j>, subtracts its projection from every
    target orbital:  |i'> = |i> - Σ_j |j><j|i>

    This ensures the target orbitals are orthogonal to the reference
    subspace without changing their relative orientation within the
    complementary subspace.

    Args:
        mo_target: (nbas, n)  orbitals to clean
        mo_ref: (nbas, m)     reference orbitals to project out

    Returns:
        mo_clean: (nbas, n)  target orbitals with reference components removed
    """
    mo_clean = mo_target.copy()
    for i in range(mo_ref.shape[1]):
        ovlp = mo_ref[:, i].conj().T @ mo_clean
        mo_clean -= np.outer(mo_ref[:, i], ovlp)
    return mo_clean


def lowdin_orth(mo, thr=1e-8):
    """Lowdin (symmetric) orthonormalization with linear-dep removal.

    X = S^{-1/2} = V @ diag(1/sqrt(w)) @ V^T, then mo_orth = mo @ X.

    Uses canonical orthogonalization: drops eigenvectors with eigenvalue
    below `thr` to handle near-linear dependencies from projection loss.

    Args:
        mo: (nbas, n)  orbitals to orthonormalize
        thr: eigenvalue threshold for linear dependency removal

    Returns:
        mo_orth: (nbas, n_keep)  orthonormal orbitals
        n_dropped: int  number of linear dependencies removed
    """
    S = mo.conj().T @ mo
    w, v = np.linalg.eigh(S)
    mask = w > thr
    n_dropped = mo.shape[1] - mask.sum()

    if n_dropped > 0:
        print(f"  [lowdin_orth] removed {n_dropped} linear dependencies "
              f"(min kept: {w[mask].min():.2e}, max dropped: "
              f"{'n/a' if n_dropped == mo.shape[1] else w[~mask].max():.2e})")

    # X = V @ diag(1/sqrt(w)) @ V^T
    X = v[:, mask] @ np.diag(1.0 / np.sqrt(w[mask])) @ v[:, mask].T.conj()
    mo_orth = mo @ X
    return mo_orth, n_dropped


# ──────────────────────────────────────────────────────────────────────
# Main public API
# ──────────────────────────────────────────────────────────────────────

def build_pes_guess(prev_cas, prev_dmet, mol_prev,
                    curr_dmet, mol_curr,
                    curr_ncas, curr_nelecas,
                    aolabels, avas_kwargs=None,
                    orth_thr=1e-8):
    """Build CASSCF initial guess by projecting active NOs across geometries.

    This is the core function.  It:

    1. Extracts active-space natural orbitals from prev_cas (occupation-ordered,
       geometry-stable ordering, no canonical-orbital swapping).
    2. Projects ONLY the active NOs through the AO bridge:
       prev_emb → prev_AO → curr_AO → curr_emb.
    3. Uses AVAS on the current geometry to get a correctly-structured
       MO template (core / active / virtual ordering).
    4. Replaces the active block of the template with the projected NOs,
       after orthogonalizing them against core and virtual.
    5. Runs diagnostic checks on the result quality.

    Args:
        prev_cas: converged CASSCF from previous geometry
        prev_dmet: SSDMET from previous geometry
        mol_prev: PySCF mol, previous geometry
        curr_dmet: SSDMET for current geometry (already built)
        mol_curr: PySCF mol, current geometry
        curr_ncas: active orbital count for current geometry
        curr_nelecas: active electron count (int or (nalpha, nbeta) tuple)
        aolabels: list of AO label patterns for AVAS, e.g. ['Co 3d']
        avas_kwargs: dict of extra kwargs forwarded to avas_export()
                     (minao, threshold, openshell_option, ...)
        orth_thr: linear dependency threshold for Lowdin orth

    Returns:
        mo_guess: (nes_curr, nes_curr)  initial-guess MO for curr CASSCF,
                  orthogonal in the curr_emb metric
        quality: dict with diagnostic keys:
            'occ_no_prev'       — prev geometry NO occupations
            'fallback'          — True if AVAS fallback was used
            'max_nonorth'       — max |mo_guess^T mo_guess - I|
            'det_ovlap_avas'    — |det(active overlap with pure AVAS guess)|
            'state_jump_warning'— True if overlap with AVAS is suspiciously low
            'n_active_dropped'  — number of linear dependencies removed
    """
    if avas_kwargs is None:
        avas_kwargs = {}

    quality = {}

    # ── 1. Extract active natural orbitals from previous CASSCF ──
    occ_no_prev, mo_active_no_prev = make_active_natorb(prev_cas)
    print(f"[pes_guess] Prev geometry NO occupations: {np.round(occ_no_prev, 4)}")
    quality['occ_no_prev'] = occ_no_prev

    ncas_prev = mo_active_no_prev.shape[1]
    if ncas_prev != curr_ncas:
        print(f"[pes_guess] NOTE: prev ncas={ncas_prev}, curr ncas={curr_ncas}")
        if ncas_prev > curr_ncas:
            # Truncate: keep most-occupied NOs
            mo_active_no_prev = mo_active_no_prev[:, :curr_ncas]
            print(f"[pes_guess]   truncated to {curr_ncas} (most occupied)")
        else:
            # We'll supplement from AVAS — the AVAS template already
            # has correctly-identified active orbitals
            print(f"[pes_guess]   expanding from {ncas_prev} to {curr_ncas} "
                  f"(extra from AVAS)")

    # ── 2. Project active NOs across geometries ──
    mo_active_proj = project_active_mo(
        mo_active_no_prev, prev_dmet, mol_prev, curr_dmet, mol_curr
    )

    # ── 3. Get AVAS template for current geometry ──
    # AVAS gives us properly-ordered MOs where active orbitals live at
    # columns ncore:ncore+ncas; we use this as the structural template.
    _ncas_chk, _nelec_chk, es_mo_avas = curr_dmet.avas_export(
        aolabels, **avas_kwargs
    )

    # ── 4. Determine ncore from a temporary CASSCF object ──
    # CASSCF computes ncore from mol.nelectron and nelecas; this is
    # the only reliable way to know where the active block starts.
    from embed_sim import sacasscf_mixer
    cas_temp = sacasscf_mixer.sacasscf_mixer(
        curr_dmet.es_mf, curr_ncas, curr_nelecas,
        statelis=sacasscf_mixer.read_statelis(prev_cas),
        weights=getattr(prev_cas, 'weights', None)
    )
    ncore = cas_temp.ncore
    print(f"[pes_guess] ncore={ncore}, ncas={curr_ncas}, "
          f"nes={curr_dmet.nes}")

    # ── 5. Orthogonalize projected NOs against core and virtual ──
    mo_core = es_mo_avas[:, :ncore]
    mo_virt = es_mo_avas[:, ncore + curr_ncas:]

    mo_active_clean = project_out(mo_active_proj, mo_core)
    mo_active_clean = project_out(mo_active_clean, mo_virt)

    # Symmetric orthonormalization (handles projection-quality losses)
    mo_active_orth, n_dropped = lowdin_orth(mo_active_clean, thr=orth_thr)
    quality['n_active_dropped'] = n_dropped

    if mo_active_orth.shape[1] < curr_ncas:
        print(f"[pes_guess] WARNING: {n_dropped} orbitals lost in "
              f"orthogonalization — falling back to AVAS")
        quality['fallback'] = True
        # Build diagnostic info anyway
        quality['max_nonorth'] = np.max(np.abs(
            es_mo_avas.conj().T @ es_mo_avas - np.eye(es_mo_avas.shape[1])))
        quality['det_ovlap_avas'] = 1.0
        quality['state_jump_warning'] = False
        return es_mo_avas, quality

    quality['fallback'] = False

    # ── 6. Assemble final guess ──
    mo_guess = es_mo_avas.copy()
    mo_guess[:, ncore:ncore + curr_ncas] = mo_active_orth

    # ── 7. Diagnostics ──
    # Orthogonality check
    max_nonorth = np.max(np.abs(
        mo_guess.conj().T @ mo_guess - np.eye(mo_guess.shape[1])))
    print(f"[pes_guess] Max orthogonality deviation: {max_nonorth:.2e}")
    quality['max_nonorth'] = float(max_nonorth)

    # Overlap between projected active NOs and AVAS active MOs
    # Low determinant → projected guess diverged significantly from AVAS
    # (could mean state jump, or could just mean AVAS is in a different gauge)
    ovlp = (mo_active_orth.conj().T @
            es_mo_avas[:, ncore:ncore + curr_ncas])
    det_ovlp = abs(np.linalg.det(ovlp))
    print(f"[pes_guess] |det(overlap with AVAS active)|: {det_ovlp:.4f}")
    quality['det_ovlap_avas'] = float(det_ovlp)

    if det_ovlp < 0.5:
        print(f"[pes_guess] ⚠ WARNING: low overlap with AVAS ({det_ovlp:.4f})")
        print(f"[pes_guess]   Projected active space may have rotated "
              f"significantly.")
        print(f"[pes_guess]   Run and check NO occupations after CASSCF "
              f"convergence.")
        quality['state_jump_warning'] = True
    else:
        quality['state_jump_warning'] = False

    return mo_guess, quality


def pes_scan_step(prev_cas, prev_dmet, mol_prev,
                  curr_dmet, mol_curr,
                  aolabels,
                  statelis=None,
                  avas_kwargs=None,
                  level_shift=0.3,
                  ah_level_shift=1e-4,
                  fine_tune=True,
                  chkfile=None):
    """One full PES step: build guess + run CASSCF with level-shift annealing.

    Convenience wrapper around build_pes_guess + CASSCF kernel.
    Runs a two-stage CASSCF:
      1. Coarse convergence with level_shift (suppresses orbital oscillations
         when the initial guess is imperfect)
      2. Fine convergence without level_shift (gets exact gradient)

    After convergence, compares NO occupations against the previous
    geometry to verify state continuity on the same PES.

    Args:
        prev_cas: converged CASSCF from previous geometry
        prev_dmet: SSDMET from previous geometry
        mol_prev: PySCF mol, previous geometry
        curr_dmet: SSDMET for current geometry (already built)
        mol_curr: PySCF mol, current geometry
        aolabels: AO label patterns for AVAS, e.g. ['Co 3d']
        statelis: state list for sacasscf_mixer (default: from prev_cas)
        avas_kwargs: extra kwargs for avas_export
        level_shift: level shift for coarse stage (default 0.3)
        ah_level_shift: AH level shift (default 1e-4)
        fine_tune: if True, run second pass without level shift
        chkfile: checkpoint file path for the new CASSCF

    Returns:
        curr_cas: converged CASSCF for current geometry
        quality: dict with diagnostic info (see build_pes_guess)
    """
    from embed_sim import sacasscf_mixer

    if avas_kwargs is None:
        avas_kwargs = {}
    if statelis is None:
        statelis = sacasscf_mixer.read_statelis(prev_cas)

    curr_ncas = prev_cas.ncas
    curr_nelecas = prev_cas.nelecas

    # ── Build projected initial guess ──
    mo_guess, quality = build_pes_guess(
        prev_cas, prev_dmet, mol_prev,
        curr_dmet, mol_curr,
        curr_ncas, curr_nelecas,
        aolabels, avas_kwargs
    )

    # ── Create CASSCF solver ──
    curr_cas = sacasscf_mixer.sacasscf_mixer(
        curr_dmet.es_mf, curr_ncas, curr_nelecas,
        statelis=statelis,
        weights=getattr(prev_cas, 'weights', None)
    )
    if chkfile is not None:
        curr_cas.chkfile = chkfile

    # ── Stage 1: coarse convergence with level shift ──
    if level_shift > 0 or ah_level_shift > 0:
        print(f"[pes_scan_step] Stage 1: level_shift={level_shift}, "
              f"ah_level_shift={ah_level_shift}")
        curr_cas.level_shift = level_shift
        curr_cas.ah_level_shift = ah_level_shift
        curr_cas.max_cycle_macro = getattr(curr_cas, 'max_cycle_macro', 50)
        curr_cas.kernel(mo_guess)

    # ── Stage 2: fine convergence without level shift ──
    if fine_tune and (level_shift > 0 or ah_level_shift > 0):
        print(f"[pes_scan_step] Stage 2: fine convergence (no level shift)")
        curr_cas.level_shift = 0.0
        curr_cas.ah_level_shift = 0.0
        curr_cas.max_cycle_macro = getattr(curr_cas, 'max_cycle_macro', 200)
        curr_cas.kernel(curr_cas.mo_coeff)
    elif not fine_tune:
        curr_cas.kernel(mo_guess)

    # ── Verify state continuity ──
    occ_curr, _ = make_active_natorb(curr_cas)
    occ_prev = quality['occ_no_prev']
    print(f"[pes_scan_step] Prev NO occupations: {np.round(occ_prev, 4)}")
    print(f"[pes_scan_step] Curr NO occupations: {np.round(occ_curr, 4)}")

    max_occ_diff = np.max(np.abs(occ_curr - occ_prev[:len(occ_curr)]))
    print(f"[pes_scan_step] Max occupation diff: {max_occ_diff:.4f}")
    if max_occ_diff > 0.3:
        print(f"[pes_scan_step] ⚠  WARNING: Large NO occupation change "
              f"({max_occ_diff:.3f}) — possible state jump!")

    quality['occ_no_curr'] = occ_curr
    quality['max_occ_diff'] = float(max_occ_diff)

    return curr_cas, quality
