

from pyscf import gto
import numpy as np
def concentric_occ_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore',
                                    spin='alpha', ele_density=False, threshold=1e-6):
    """
    Concentric FO localization for UHF: shift strongly coupled FO orbitals
    into the bath for a specific spin channel.

    Parameters
    ----------
    proj_bas : str
        Basis set name for the projection (fake) molecule on atoms_A.
    n_shell : int
        Number of concentric shells to expand.
    atoms_A : list of int
        Atom indices (0-based) defining the target region.
    couple_op : str
        Coupling operator: 'hcore' or 'fock'.
    spin : str
        Which spin channel to localize: 'alpha' or 'beta'.
    ele_density : bool
        If True, export cube files for each shell.
    threshold : float
        SVD singular value threshold.
    """
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    if spin == 'alpha':
        s = 0
    elif spin == 'beta':
        s = 1
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin}")

    nimp = len(dmet.imp_idx)
    nbath = dmet.nes[s] - nimp
    nfo_s = dmet.nfo[s]

    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    fake_mol.spin = 0
    fake_mol.charge = 0
    if fake_mol.nelectron % 2 != 0:
        fake_mol.spin = 1
    fake_mol.build(False, False)

    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol)
    s_pb_inv = np.linalg.inv(s_pb)

    # FO orbitals for this spin in AO basis
    fo_AO = dmet.caolo @ dmet.lo_cloes[s][:, nimp+nbath : nimp+nbath+nfo_s]

    dmet.log.info(f"[{spin}] fo_AO shape: {fo_AO.shape}, nfo={nfo_s}")

    def svd_step(coeff, couple_matrix, coeff_ker):
        if coeff.shape[1] == 0 or coeff_ker.shape[1] == 0:
            return np.zeros((coeff.shape[0], 0)), coeff_ker
        ovlp = coeff.T.conj() @ couple_matrix @ coeff_ker
        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span = Vh[:r, :].T.conj()
        V_ker  = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1

    c_fo_prime = s_pb_inv @ s_cross @ fo_AO
    U, sigma, Vh = np.linalg.svd(c_fo_prime.T.conj() @ s_cross @ fo_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    dmet.log.info(f"[{spin}] Initial SVD rank r0: {r0}, sigma: {sigma[:min(10, len(sigma))]}...")
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_occ = [fo_AO @ V_span]
    C_ker = [fo_AO @ V_ker]

    if couple_op == 'fock':
        dm = dmet.mf_or_cas.make_rdm1()
        fock = dmet.mf_or_cas.get_fock(dm=dm)
        if isinstance(fock, tuple):
            fock_ao = fock[s]
        elif isinstance(fock, np.ndarray) and fock.ndim == 3:
            fock_ao = fock[s]
        else:
            fock_ao = fock
        dmet.log.info(f"[{spin}] fock_ao shape: {fock_ao.shape}")

    dmet.log.info(f"[{spin}] C_occ[0] shape={C_occ[0].shape}, C_ker[0] shape={C_ker[0].shape}")

    for i in range(n_shell):
        if C_ker[i].shape[1] == 0:
            dmet.log.info(f"[{spin}] Shell {i}: ker is empty, stopping iteration")
            break

        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
        elif couple_op == 'fock':
            couple_matrix = fock_ao
        else:
            raise ValueError(f"Unknown couple_op: {couple_op}")

        dmet.log.info(f"[{spin}] Shell {i}: coeff shape={C_occ[i].shape}, "
                        f"couple_matrix shape={couple_matrix.shape}, "
                        f"coeff_ker shape={C_ker[i].shape}")

        new_occ, new_ker = svd_step(C_occ[i], couple_matrix, C_ker[i])
        C_occ.append(new_occ)
        C_ker.append(new_ker)
        dmet.log.info(f"[{spin}] Shell {i+1}: {new_occ.shape[1]} new vectors, "
                        f"{new_ker.shape[1]} remaining in ker")

    # Canonicalize the localized FO orbitals in Fock subspace
    if couple_op == 'fock':
        fock_for_canon = fock_ao
    else:
        fock = dmet.mf_or_cas.get_fock()
        if isinstance(fock, tuple):
            fock_for_canon = fock[s]
        elif isinstance(fock, np.ndarray) and fock.ndim == 3:
            fock_for_canon = fock[s]
        else:
            fock_for_canon = fock

    C_occ_matrix = np.hstack(C_occ)
    fock_sub = C_occ_matrix.T.conj() @ fock_for_canon @ C_occ_matrix
    mo_energy_occ, U = np.linalg.eigh(fock_sub)
    C_occ_canonical = C_occ_matrix @ U

    C_fo_new = C_ker[-1]
    fock_fo = C_fo_new.T.conj() @ fock_for_canon @ C_fo_new
    mo_energy_fo, U_fo = np.linalg.eigh(fock_fo)
    C_fo_canonical = C_fo_new @ U_fo

    lo_cloes_s = dmet.lo_cloes[s]
    Q_emb = lo_cloes_s[:, :nimp+nbath]               
    Q_fv  = lo_cloes_s[:, nimp+nbath+nfo_s:] # frozen virtual

    # Transform canonical FO to LO basis 
    lo2New_bath = dmet.cloao @ C_occ_canonical
    lo2New_fo   = dmet.cloao @ C_fo_canonical

    n_shifted = lo2New_bath.shape[1]
    dmet.log.info(f"[{spin}] Shifting {n_shifted} FO orbitals into bath")

    new_lo_cloes_s = np.hstack([Q_emb, lo2New_bath, lo2New_fo, Q_fv])

    # Update internal state for this spin
    new_nes_s = dmet.nes[s] + n_shifted
    new_nfo_s = dmet.nfo[s] - n_shifted

    # updated lo_cloes
    lo_cloes_list = list(dmet.lo_cloes)
    lo_cloes_list[s] = new_lo_cloes_s
    dmet.lo_cloes = tuple(lo_cloes_list)

    nes_list = list(dmet.nes)
    nfo_list = list(dmet.nfo)
    nes_list[s] = new_nes_s
    nfo_list[s] = new_nfo_s
    dmet.nes = tuple(nes_list)
    dmet.nfo = tuple(nfo_list)

    # Rebuild AO-basis coefficients
    dmet.caoes = (dmet.caolo @ dmet.lo_cloes[0], dmet.caolo @ dmet.lo_cloes[1])
    dmet.es_orb = (dmet.caoes[0][:, :dmet.nes[0]], dmet.caoes[1][:, :dmet.nes[1]])
    dmet.fo_orb = (dmet.caoes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]],
                    dmet.caoes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]])
    dmet.fv_orb = (dmet.caoes[0][:, dmet.nes[0]+dmet.nfo[0]:],
                    dmet.caoes[1][:, dmet.nes[1]+dmet.nfo[1]:])

    # Rebuild embedded 1e/2e integrals and embedded DM
    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi') and getattr(dmet, 'es_cderi', None) is not None:
        dmet.log.info("[%s] Rebuilding DF 3-index integrals (es_cderi) ...", spin)
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()

    # Rebuild es_dm: keep original ES block, add FO→bath occupations
    old_nes_s = dmet.nes[s] - n_shifted  # ES size before this concentric step
    old_es_dm_s = dmet.es_dm[s]
    new_es_dm_s = np.zeros((dmet.nes[s], dmet.nes[s]))
    new_es_dm_s[:old_nes_s, :old_nes_s] = old_es_dm_s
    if n_shifted > 0:
        # FO orbitals moving to bath: project their AO DM to get occupation
        ao_dm = dmet.mf_or_cas.make_rdm1()
        S = dmet.mf_or_cas.get_ovlp()
        new_bath_ao = dmet.es_orb[s][:, old_nes_s:]  # new bath AO coefficients
        new_bath_dm = new_bath_ao.T @ S @ ao_dm[s] @ S @ new_bath_ao
        new_es_dm_s[old_nes_s:, old_nes_s:] = new_bath_dm

    es_dm_list = list(dmet.es_dm)
    es_dm_list[s] = new_es_dm_s
    dmet.es_dm = tuple(es_dm_list)

    # Rebuild embedded mean-field
    dmet.es_mf = dmet.UHF()
    dmet.calc_fo_ene()

    dmet.log.info(f"[{spin}] Concentric FO done. \n"
                    f"NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; \n"
                    f"nimp={nimp}, nbath=({dmet.nes[0]-nimp}, {dmet.nes[1]-nimp})\n")

    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_occ):
                if c_shell.shape[1] > 0:
                    dm_shell = c_shell @ c_shell.T.conj()
                    cube_name = f"{dmet.title}_{spin}_shell_{idx}_density_occ.cube"
                    dmet.log.info(f"Exporting {cube_name}\n")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    return dmet
def concentric_vir_localization(dmet, proj_bas, n_shell, atoms_A, couple_op='hcore',
                                    spin='alpha', ele_density=False, threshold=1e-6):

    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")
    if atoms_A is None or len(atoms_A) == 0:
        raise ValueError("atoms_A must be a non-empty list of atom indices")

    atoms_A = [int(i) for i in atoms_A]
    natm = dmet.mol.natm
    if any(i < 0 or i >= natm for i in atoms_A):
        raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

    if spin == 'alpha':
        s = 0
    elif spin == 'beta':
        s = 1
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin}")

    nimp = len(dmet.imp_idx)
    nbath = dmet.nes[s] - nimp
    nfo_s = dmet.nfo[s]
    nfv_s = dmet.nfv[s]

    # Fake molecule for projection
    fake_mol = gto.Mole()
    fake_mol.verbose = dmet.verbose
    fake_mol.unit = 'Bohr'
    fake_mol.symmetry = False
    fake_mol.atom = [(dmet.mol.atom_symbol(i), dmet.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
    fake_mol.basis = proj_bas
    fake_mol.spin = 0
    fake_mol.charge = 0
    if fake_mol.nelectron % 2 != 0:
        fake_mol.spin = 1
    fake_mol.build(False, False)

    s_pb = fake_mol.intor_symmetric('int1e_ovlp')
    s_cross = gto.intor_cross('int1e_ovlp', fake_mol, dmet.mol)
    s_pb_inv = np.linalg.inv(s_pb)

    # FV orbitals for this spin in AO basis (after FO block)
    fv_start = nimp + nbath + nfo_s
    fv_end   = fv_start + nfv_s
    fv_AO = dmet.caolo @ dmet.lo_cloes[s][:, fv_start:fv_end]

    dmet.log.info(f"[{spin}] fv_AO shape: {fv_AO.shape}, nfv={nfv_s}")

    def svd_step(coeff, couple_matrix, coeff_ker):
        if coeff.shape[1] == 0 or coeff_ker.shape[1] == 0:
            return np.zeros((coeff.shape[0], 0)), coeff_ker
        ovlp = coeff.T.conj() @ couple_matrix @ coeff_ker
        U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
        r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span = Vh[:r, :].T.conj()
        V_ker  = Vh[r:, :].T.conj()
        coeff_n1 = coeff_ker @ V_span
        coeff_ker1 = coeff_ker @ V_ker
        return coeff_n1, coeff_ker1

    c_fv_prime = s_pb_inv @ s_cross @ fv_AO
    U, sigma, Vh = np.linalg.svd(c_fv_prime.T.conj() @ s_cross @ fv_AO, full_matrices=True)
    r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
    dmet.log.info(f"[{spin}] Initial SVD rank r0: {r0}, sigma: {sigma[:min(10, len(sigma))]}...")
    V_span = Vh[:r0, :].T.conj()
    V_ker  = Vh[r0:, :].T.conj()
    C_vir = [fv_AO @ V_span]
    C_ker = [fv_AO @ V_ker]

    if couple_op == 'fock':
        dm = dmet.mf_or_cas.make_rdm1()
        fock = dmet.mf_or_cas.get_fock(dm=dm)
        if isinstance(fock, tuple):
            fock_ao = fock[s]
        elif isinstance(fock, np.ndarray) and fock.ndim == 3:
            fock_ao = fock[s]
        else:
            fock_ao = fock
        dmet.log.info(f"[{spin}] fock_ao shape: {fock_ao.shape}")

    dmet.log.info(f"[{spin}] C_vir[0] shape={C_vir[0].shape}, C_ker[0] shape={C_ker[0].shape}")

    for i in range(n_shell):
        if C_ker[i].shape[1] == 0:
            dmet.log.info(f"[{spin}] Shell {i}: ker is empty, stopping iteration")
            break

        if couple_op == 'hcore':
            couple_matrix = dmet.mf_or_cas.get_hcore()
        elif couple_op == 'fock':
            couple_matrix = fock_ao
        else:
            raise ValueError(f"Unknown couple_op: {couple_op}")

        dmet.log.info(f"[{spin}] Shell {i}: coeff shape={C_vir[i].shape}, "
                        f"couple_matrix shape={couple_matrix.shape}, "
                        f"coeff_ker shape={C_ker[i].shape}")

        new_vir, new_ker = svd_step(C_vir[i], couple_matrix, C_ker[i])
        C_vir.append(new_vir)
        C_ker.append(new_ker)
        dmet.log.info(f"[{spin}] Shell {i+1}: {new_vir.shape[1]} new vectors, "
                        f"{new_ker.shape[1]} remaining in ker")

    # Canonicalize the localized FV orbitals in Fock subspace
    if couple_op == 'fock':
        fock_for_canon = fock_ao
    else:
        fock = dmet.mf_or_cas.get_fock()
        if isinstance(fock, tuple):
            fock_for_canon = fock[s]
        elif isinstance(fock, np.ndarray) and fock.ndim == 3:
            fock_for_canon = fock[s]
        else:
            fock_for_canon = fock

    C_vir_matrix = np.hstack(C_vir)
    fock_sub = C_vir_matrix.T.conj() @ fock_for_canon @ C_vir_matrix
    mo_energy_vir, U = np.linalg.eigh(fock_sub)
    C_vir_canonical = C_vir_matrix @ U

    C_fv_new = C_ker[-1]
    fock_fv = C_fv_new.T.conj() @ fock_for_canon @ C_fv_new
    mo_energy_fv, U_fv = np.linalg.eigh(fock_fv)
    C_fv_canonical = C_fv_new @ U_fv

    lo_cloes_s = dmet.lo_cloes[s]
    Q_emb = lo_cloes_s[:, :nimp+nbath]                          # existing embedding
    Q_fo  = lo_cloes_s[:, nimp+nbath : nimp+nbath+nfo_s]        # frozen occupied (unchanged)

    # Transform canonical FV to LO basis and insert after bath
    lo2New_bath = dmet.cloao @ C_vir_canonical
    lo2New_fv   = dmet.cloao @ C_fv_canonical

    n_shifted = lo2New_bath.shape[1]
    dmet.log.info(f"[{spin}] Shifting {n_shifted} FV orbitals into bath")

    # Reassemble: [Emb, Target_FV_as_bath, FO, Remaining_FV]
    new_lo_cloes_s = np.hstack([Q_emb, lo2New_bath, Q_fo, lo2New_fv])

    # Update internal state for this spin
    new_nes_s = dmet.nes[s] + n_shifted
    new_nfv_s = dmet.nfv[s] - n_shifted

    # updated lo_cloes
    lo_cloes_list = list(dmet.lo_cloes)
    lo_cloes_list[s] = new_lo_cloes_s
    dmet.lo_cloes = tuple(lo_cloes_list)

    nes_list = list(dmet.nes)
    nfv_list = list(dmet.nfv)
    nes_list[s] = new_nes_s
    nfv_list[s] = new_nfv_s
    dmet.nes = tuple(nes_list)
    dmet.nfv = tuple(nfv_list)

    # Rebuild AO-basis coefficients
    dmet.caoes = (dmet.caolo @ dmet.lo_cloes[0], dmet.caolo @ dmet.lo_cloes[1])
    dmet.es_orb = (dmet.caoes[0][:, :dmet.nes[0]], dmet.caoes[1][:, :dmet.nes[1]])
    dmet.fo_orb = (dmet.caoes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]],
                    dmet.caoes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]])
    dmet.fv_orb = (dmet.caoes[0][:, dmet.nes[0]+dmet.nfo[0]:],
                    dmet.caoes[1][:, dmet.nes[1]+dmet.nfo[1]:])

    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi') and getattr(dmet, 'es_cderi', None) is not None:
        dmet.log.info("[%s] Rebuilding DF 3-index integrals (es_cderi) ...", spin)
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
        
    # Rebuild es_dm: keep original ES block, zero-pad for new bath (FV→bath)
    # New bath orbitals come from FV space and should have zero occupation
    old_nes_s = dmet.nes[s] - n_shifted  # ES size before this concentric step
    old_es_dm_s = dmet.es_dm[s]  # current es_dm for this spin (before padding)
    new_es_dm_s = np.zeros((dmet.nes[s], dmet.nes[s]))
    new_es_dm_s[:old_nes_s, :old_nes_s] = old_es_dm_s
    # new bath block (old_nes_s:) stays zero → FV has zero occupation

    es_dm_list = list(dmet.es_dm)
    es_dm_list[s] = new_es_dm_s
    dmet.es_dm = tuple(es_dm_list)

    dmet.es_mf = dmet.UHF()
    dmet.calc_fo_ene()

    dmet.log.info(f"[{spin}] Concentric FV done.\n"
                    f"NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; \n"
                    f"nimp={nimp}, nbath=({dmet.nes[0]-nimp}, {dmet.nes[1]-nimp})\n")

    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_vir):
                if c_shell.shape[1] > 0:
                    dm_shell = c_shell @ c_shell.T.conj()
                    cube_name = f"{dmet.title}_{spin}_vir_shell_{idx}_density.cube"
                    dmet.log.info(f"Exporting {cube_name}\n")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    return dmet


def localize_spaces(dmet, method='boys', spin='alpha'):
    if spin == 'alpha':
        s = 0
    elif spin == 'beta':
        s = 1
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin}")
    dmet.log.info("[%s] ====== Orthogonality check before %s localization ======", spin, method.upper())
    _export_molden_orbital(dmet, spin=spin, suffix="before_loc")
    def check_orthogonal():
        S = dmet.mf_or_cas.get_ovlp()

        def _max_offdiag(M, label):
            """Return max absolute off-diagonal element of M^T @ S @ M."""
            if M.shape[1] <= 1:
                return 0.0
            ovlp = M.T.conj() @ S @ M
            diag = np.diag(np.diag(ovlp))
            offdiag = np.max(np.abs(ovlp - diag))
            dmet.log.info("[%s] %s: max|offdiag| = %.2e, max|diag-1| = %.2e",
                        spin, label, offdiag, np.max(np.abs(np.diag(ovlp) - 1.0)))
            return offdiag

        def _inter_block_orth(M1, M2, label1, label2):
            """Check orthogonality between two blocks: M1^T @ S @ M2."""
            if M1.shape[1] == 0 or M2.shape[1] == 0:
                return 0.0
            ovlp = M1.T.conj() @ S @ M2
            maxel = np.max(np.abs(ovlp))
            dmet.log.info("[%s] Inter-block <%s|%s>: max|ovlp| = %.2e", spin, label1, label2, maxel)
            return maxel
        nimp = len(dmet.imp_idx)
        bath_ao = dmet.es_orb[s][:, nimp:]
        fo_ao = dmet.fo_orb[s]
        fv_ao = dmet.fv_orb[s]



        # Intra-block: bath-AO (excluding impurity), FO, FV
        _max_offdiag(bath_ao, 'bath (intra)')
        _max_offdiag(fo_ao, 'fo (intra)')
        _max_offdiag(fv_ao, 'fv (intra)')

        # Full embedding space (ES = impurity + bath)
        es_ao = dmet.es_orb[s]
        _max_offdiag(es_ao, 'ES (full, intra)')

        # Inter-block: bath vs FO, bath vs FV, FO vs FV
        _inter_block_orth(bath_ao, fo_ao, 'bath', 'fo')
        _inter_block_orth(bath_ao, fv_ao, 'bath', 'fv')
        _inter_block_orth(fo_ao, fv_ao, 'fo', 'fv')

        # Impurity vs bath (impurity orthogonality to new bath)
        imp_ao = dmet.es_orb[s][:, :nimp]
        _inter_block_orth(imp_ao, bath_ao, 'imp', 'bath')

        # Full orbital set (all lo_cloes columns)
        full_lo = dmet.lo_cloes[s]
        # Check LO-basis orthogonality: lo_cloes^T @ cloao^T @ S @ caolo @ lo_cloes = I
        # Since caolo^T @ S @ caolo = I (LO basis is orthonormal), this reduces to
        # lo_cloes^T @ lo_cloes, but we use AO metric for safety:
        full_ao = dmet.caolo @ full_lo
        _max_offdiag(full_ao, 'FULL lo_cloes (intra)')

        dmet.log.info("[%s] ====== Orthogonality check done ======", spin)


    def localize_subspace(coeff_AO, name):
        from pyscf import lo
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
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first before localization.")            
    dmet.log.info(f"Performing {method.upper()} localization on Env subspaces (Bath, FO, FV)...")

    nimp = len(dmet.imp_idx)
    nbath = dmet.nes[s] - nimp
    nfo_s = dmet.nfo[s]
    nfv_s = dmet.nfv[s]
   
    bath_AO = dmet.caolo @ dmet.lo_cloes[s][:, nimp:nimp+nbath]


    # FV orbitals for this spin in AO basis (after FO block)
    fv_start = nimp + nbath + nfo_s
    fv_end   = fv_start + nfv_s
    fv_AO = dmet.caolo @ dmet.lo_cloes[s][:, fv_start:fv_end]

    dmet.log.info(f"[{spin}] fv_AO shape: {fv_AO.shape}, nfv={nfv_s}")
    
    ## fo 
    fo_AO = dmet.caolo @ dmet.lo_cloes[s][:, nimp+nbath : nimp+nbath+nfo_s]

    print(f"[{spin}] Before localization: \nbath_AO shape: {bath_AO.shape}, nbath={nbath}; \n")

    dmet.log.info(f"[{spin}] fo_AO shape: {fo_AO.shape}, nfo={nfo_s}")

    bath_loc_AO = localize_subspace(bath_AO, 'bath')
    fo_loc_AO = localize_subspace(fo_AO, 'fo')
    fv_loc_AO = localize_subspace(fv_AO, 'fv')

    # from AO to LO
    dmet.lo_cloes[s][:, nimp:nimp+nbath] = dmet.cloao @ bath_loc_AO
    dmet.lo_cloes[s][:, nimp+nbath:nimp+nbath+nfo_s] = dmet.cloao @ fo_loc_AO
    dmet.lo_cloes[s][:, fv_start:fv_end] = dmet.cloao @ fv_loc_AO

    dmet.caoes = (dmet.caolo @ dmet.lo_cloes[0], dmet.caolo @ dmet.lo_cloes[1])
    dmet.es_orb = (dmet.caoes[0][:, :dmet.nes[0]], dmet.caoes[1][:, :dmet.nes[1]])
    dmet.fo_orb = (dmet.caoes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]], dmet.caoes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]])
    dmet.fv_orb = (dmet.caoes[0][:, dmet.nes[0]+dmet.nfo[0]:dmet.nes[0]+dmet.nfo[0]+dmet.nfv[0]], dmet.caoes[1][:, dmet.nes[1]+dmet.nfo[1]:dmet.nes[1]+dmet.nfo[1]+dmet.nfv[1]])


    dmet.log.info("[%s] ====== Orthogonality check after %s localization ======", spin, method.upper())

    check_orthogonal()

    # Export localized orbitals to Molden file (keep imp/bath/fo/fv order)
    _export_molden_orbital(dmet, spin=spin, suffix=f"localized_{method}")

    dmet.log.info("Environment subspaces localized successfully.")
    return dmet


def _export_molden_orbital(dmet, spin='alpha', suffix=None):
    """
    Export the full orbital set (imp + bath + FO + FV) for a given spin
    to a Molden file.  Orbitals are written in their natural order:
    imp_0 ... imp_{nimp-1}, bath_0 ..., fo_0 ..., fv_0 ...
    The orbital label and index are printed into the log.
    """
    if spin == 'alpha':
        s = 0
    elif spin == 'beta':
        s = 1
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin}")

    try:
        from pyscf.tools import molden

        nimp = len(dmet.imp_idx)
        nbath_s = dmet.nes[s] - nimp
        nfo_s = dmet.nfo[s]
        nfv_s = dmet.nfv[s]

        # Build labels in original order (no sorting)
        mo_labels = (
            [f'imp_{i}'   for i in range(nimp)] +
            [f'bath_{i}'  for i in range(nbath_s)] +
            [f'fo_{i}'    for i in range(nfo_s)] +
            [f'fv_{i}'    for i in range(nfv_s)]
        )

        # Build MO coefficients in original order
        mo_coeff = np.hstack([dmet.es_orb[s], dmet.fo_orb[s], dmet.fv_orb[s]])

        # Occupations from embedded DM (ES only), diagonalize for natural occ
        mo_occ = np.zeros(mo_coeff.shape[1])
        occ_loc, _ = np.linalg.eigh(dmet.es_dm[s])
        mo_occ[:dmet.nes[s]] = occ_loc[::-1]  # descending order

        mo_ene = np.zeros(mo_coeff.shape[1])

        if suffix is None: 
            suffix_str = ""
        else:
            suffix_str = f"_{suffix}"

        molden_filename = f"{dmet.title}{suffix_str}_{spin}.molden"
        with open(molden_filename, 'w') as f:
            molden.header(dmet.mol, f)
            molden.orbital_coeff(dmet.mol, f, mo_coeff,
                                 ene=mo_ene, occ=mo_occ,
                                 ignore_h=True)

        dmet.log.info("[%s] Orbitals written to %s", spin, molden_filename)

        # Print orbital mapping table
        dmet.log.info("[%s] Orbital index —— label mapping (total %d orbitals):",
                      spin, len(mo_labels))
        for idx, lbl in enumerate(mo_labels):
            # Also show occupation if available (non-zero)
            occ_val = mo_occ[idx]
            if abs(occ_val) > 1e-10:
                dmet.log.info("[%s]   %3d  %s   occ=%.4f", spin, idx, lbl, occ_val)
            else:
                dmet.log.info("[%s]   %3d  %s", spin, idx, lbl)

    except Exception as e:
        dmet.log.warn("[%s] Failed to export Molden file: %s", spin, str(e))


def get_UMP2_bath(dmet, ao2eo=None, ao2core=None, ao2vir=None, lo2core=None, lo2vir=None, eta=1e-2, verbose=None):
    """
    Construct UMP2 bath orbitals for UHF DMET and update dmet in place.

    When ao2eo/ao2core/ao2vir/lo2core/lo2vir are None, they are derived from
    dmet attributes (es_orb, fo_orb, fv_orb, lo_cloes) on a per-spin basis.
    The function computes per-spin bath projections and updates dmet.lo_cloes,
    dmet.nes, dmet.nfo/dmet.nfv, dmet.es_orb, dmet.fo_orb, dmet.fv_orb,
    dmet.es_mf, etc. simultaneously for both alpha and beta spins.

    Returns
    -------
    dmet : SSDMET_uhf
    """
    from pyscf.lib import logger
    from pyscf import lib, df, ao2mo
    from pyscf.mp.dfmp2 import _DFINCOREERIS as _RDFINCOREERIS
    from pyscf.mp.dfmp2 import _DFOUTCOREERIS as _RDFOUTCOREERIS
    from pyscf.mp.dfump2 import _DFINCOREERIS as _UDFINCOREERIS
    from pyscf.mp.dfump2 import _DFOUTCOREERIS as _UDFOUTCOREERIS
    from functools import reduce
    import numpy as np
    import ctypes

    nimp = len(dmet.imp_idx)

    # ---- Derive per-spin transformation matrices from dmet if not provided ----
    if ao2eo is None:
        # es_orb is (coeff_a, coeff_b): from AO to embedding space
        ao2eo = dmet.es_orb
    if ao2core is None:
        ao2core = dmet.fo_orb
    if ao2vir is None:
        ao2vir = dmet.fv_orb
    if lo2core is None:
        # lo_cloes[s][:, nimp+nbath : nimp+nbath+nfo] -> LO-basis FO coeffs
        lo2core = (
            dmet.lo_cloes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]],
            dmet.lo_cloes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]],
        )
    if lo2vir is None:
        lo2vir = (
            dmet.lo_cloes[0][:, dmet.nes[0]+dmet.nfo[0]:],
            dmet.lo_cloes[1][:, dmet.nes[1]+dmet.nfo[1]:],
        )

    mf = dmet.mf_or_cas.to_uhf()
    mf.max_memory = dmet.mf_or_cas.max_memory
    es_mf = dmet.es_mf.to_uhf() if dmet.es_mf is not None else dmet.UHF().to_uhf()

    log = logger.new_logger(mf, verbose=verbose)
    log.info('')
    log.info('constructing UMP2 bath for UHF DMET')
    
    nocc = [(mf.mo_occ[i]>0).sum() for i in range(2)]
    nvir = [(mf.mo_occ[i]==0).sum() for i in range(2)]
    
    occ_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]>0] for i in range(2)]
    vir_coeff = [mf.mo_coeff[i][:,mf.mo_occ[i]==0] for i in range(2)]
    es_occ_coeff = [lib.dot(ao2eo[i], es_mf.mo_coeff[i][:,es_mf.mo_occ[i]>0]) for i in range(2)]
    es_vir_coeff = [lib.dot(ao2eo[i], es_mf.mo_coeff[i][:,es_mf.mo_occ[i]==0]) for i in range(2)]
    
    occ_energy = [mf.mo_energy[i][mf.mo_occ[i]>0] for i in range(2)]
    vir_energy = [mf.mo_energy[i][mf.mo_occ[i]==0] for i in range(2)]
    es_occ_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]>0] for i in range(2)]
    es_vir_energy = [es_mf.mo_energy[i][es_mf.mo_occ[i]==0] for i in range(2)]
    
    def _make_df_eris(mf, occ_coeff=None, vir_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
        log = logger.new_logger(mf, verbose)
    
        with_df = getattr(mf, 'with_df', None)
        assert( with_df is not None )
    
        if with_df._cderi is None:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
        else:
            log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
    
        assert (occ_coeff is not None and vir_coeff is not None)
    
        # determine incore or outcore
        nocc = np.asarray([occ_coeff[i].shape[1] for i in range(2)])
        nvir = np.asarray([vir_coeff[i].shape[1] for i in range(2)])
        naux = with_df.get_naoaux()
    
        if ovL is not None:
            if isinstance(ovL, np.ndarray):
                outcore = False
            elif isinstance(ovL, str):
                outcore = True
            else:
                log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                          type(ovL))
                raise TypeError
        else:
            mem_now = mf.max_memory - lib.current_memory()[0]
            mem_df = sum(nocc*nvir)*8/1024**2.
            log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
            
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
        log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')
    
        if outcore:
            eris = _UDFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                  ovL=ovL, ovL_to_save=ovL_to_save,
                                  verbose=log.verbose, stdout=log.stdout)
        else:
            eris = _UDFINCOREERIS(with_df, occ_coeff, vir_coeff, mf.max_memory,
                                 ovL=ovL,
                                 verbose=log.verbose, stdout=log.stdout)
        eris.build()
    
        return eris
    
    def get_t2(mf, occ_energy=None, vir_energy=None, eris=None, with_t2=True, verbose=None):
    
        log = logger.new_logger(mf, verbose)
    
        assert (ao2mo is not None)
    
        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
        nvirmax = max(nvir)
        assert (occ_energy is not None and vir_energy is not None)
    
        mem_avail = mf.max_memory - lib.current_memory()[0]
    
        if with_t2:
            t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
                  np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
                  np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
            t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
            mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
        else:
            t2 = None
            t2_ptr = [lib.c_null_ptr()] * 3
    
        if mem_avail < 0:
            log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
            raise MemoryError
    
        libmp = lib.load_library('libmp')
        drv = libmp.MP2_contract_d
    
        # determine occ blksize
        if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
            occ_blksize = nocc
        else:   # outcore ovL
            # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
            occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
            occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]
    
        log.debug('occ blksize for %s loop: %d/%d %d/%d', mf.__class__.__name__,
                  occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])
    
        cput1 = (logger.process_clock(), logger.perf_counter())
        # for different spin
        for s in [0,1]:
            s_t2 = 0 if s == 0 else 2
            moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
            for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                nocci = i1-i0
                iaL = eris.get_occ_blk(s,i0,i1)
                for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                    noccj = j1-j0
                    if ibatch == jbatch:
                        jbL = iaL
                    else:
                        jbL = eris.get_occ_blk(s,j0,j1)
    
                    ed = np.zeros(1, dtype=np.float64)
                    ex = np.zeros(1, dtype=np.float64)
                    moeoo_block = np.asarray(
                        occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                    s2symm = 1
                    t2_ex = True
                    drv(
                        ed.ctypes.data_as(ctypes.c_void_p),
                        ex.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(s2symm),
                        iaL.ctypes.data_as(ctypes.c_void_p),
                        jbL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(i0), ctypes.c_int(j0),
                        ctypes.c_int(nocci), ctypes.c_int(noccj),
                        ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                        moeoo_block.ctypes.data_as(ctypes.c_void_p),
                        moevv.ctypes.data_as(ctypes.c_void_p),
                        t2_ptr[s_t2], ctypes.c_int(t2_ex)
                    )
    
                    jbL = None
                iaL = None
    
                cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                         *cput1)
                
        # opposite spin
        sa, sb = 0, 1
        drv = libmp.MP2_OS_contract_d
        moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
            nocci = i1-i0
            iaL = eris.get_occ_blk(sa,i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
                noccj = j1-j0
                jbL = eris.get_occ_blk(sb,j0,j1)
    
                ed = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                    ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                    ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr[1]
                )
    
                jbL = None
            iaL = None
    
            cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                     *cput1)
    
        return t2
    
    def _gamma1_intermediates(mf, t2=None, eris=None):
        assert (t2 is not None)
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2[1].shape
        
        dooa  = lib.einsum('imef,jmef->ij', t2aa, t2aa) *-.5
        dooa -= lib.einsum('imef,jmef->ij', t2ab, t2ab)
        doob  = lib.einsum('imef,jmef->ij', t2bb, t2bb) *-.5
        doob -= lib.einsum('mief,mjef->ij', t2ab, t2ab)
    
        dvva  = lib.einsum('mnae,mnbe->ba', t2aa, t2aa) * .5
        dvva += lib.einsum('mnae,mnbe->ba', t2ab, t2ab)
        dvvb  = lib.einsum('mnae,mnbe->ba', t2bb, t2bb) * .5
        dvvb += lib.einsum('mnea,mneb->ba', t2ab, t2ab)
        
        dooa += dooa.T
        doob += doob.T
        dvva += dvva.T
        dvvb += dvvb.T
        dooa *= 0.5
        doob *= 0.5
        dvva *= 0.5
        dvvb *= 0.5
        dooa[np.diag_indices(nocca)] += 1
        doob[np.diag_indices(noccb)] += 1
        
        dm1occ = [dooa,doob]
        dm1vir = [dvva,dvvb]
        return dm1occ, dm1vir
    
    eris_Ov = _make_df_eris(mf, occ_coeff, es_vir_coeff, verbose=verbose)
    eris_oV = _make_df_eris(mf, es_occ_coeff, vir_coeff, verbose=verbose)
    
    t_IJab = get_t2(mf, occ_energy, es_vir_energy, eris_Ov, verbose=verbose, with_t2=True)
    t_ijAB = get_t2(mf, es_occ_energy, vir_energy, eris_oV, verbose=verbose, with_t2=True)
    
    D_IJ = _gamma1_intermediates(mf, t_IJab, eris_Ov)[0]
    D_AB = _gamma1_intermediates(mf, t_ijAB, eris_oV)[1]
    
    S = mf.get_ovlp()
    D_IJ_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', occ_coeff[i], D_IJ[i], occ_coeff[i]) for i in range(2)])
    D_AB_ao = reduce(np.add, [lib.einsum('pi,ij,qj->pq', vir_coeff[i], D_AB[i], vir_coeff[i]) for i in range(2)])

    bins = np.array([10**-x for x in range(0,11)][::-1])

    # BNO construction for deferent spins
    lo2MP2_bath_list = [None, None]
    lo2MP2_core_list = [None, None]
    lo2MP2_vir_list  = [None, None]
    nbath_new_spin   = [0, 0]
    nbath_new_core_spin = [0, 0]
    nbath_new_vir_spin  = [0, 0]

    for s in [0, 1]:
        spin_label = 'alpha' if s == 0 else 'beta'

        D_MP2_core_s = reduce(lib.dot, (ao2core[s].T, S, D_IJ_ao, S.T, ao2core[s]))
        D_MP2_vir_s  = reduce(lib.dot, (ao2vir[s].T, S, D_AB_ao, S.T, ao2vir[s]))

        eigvals_core, eigvecs_core = np.linalg.eigh(D_MP2_core_s)
        histogram_core = make_histogram(2 - eigvals_core, bins, labels=True, show_number=True)
        log.info('[%s] Occupied BNO histogram', spin_label)
        log.info('%s', histogram_core)
        log.info('')

        eigvals_vir, eigvecs_vir = np.linalg.eigh(D_MP2_vir_s)
        histogram_vir = make_histogram(eigvals_vir, bins, labels=True, show_number=True)
        log.info('[%s] Virtual BNO histogram', spin_label)
        log.info('%s', histogram_vir)
        log.info('')

        MP2_bath_core = (eigvals_core < 2 - eta)
        MP2_bath_vir  = (eigvals_vir > eta)
        lo2MP2_bath_core = lib.dot(lo2core[s], eigvecs_core[:, MP2_bath_core])
        lo2MP2_bath_vir  = lib.dot(lo2vir[s],  eigvecs_vir[:,  MP2_bath_vir])
        lo2MP2_bath_s = np.hstack((lo2MP2_bath_core, lo2MP2_bath_vir))
        lo2MP2_core_s = lib.dot(lo2core[s], eigvecs_core[:, ~MP2_bath_core])
        lo2MP2_vir_s  = lib.dot(lo2vir[s],  eigvecs_vir[:,  ~MP2_bath_vir])

        nbath_new_core_spin[s] = MP2_bath_core.sum()
        nbath_new_vir_spin[s]  = MP2_bath_vir.sum()
        nbath_new_spin[s] = nbath_new_core_spin[s] + nbath_new_vir_spin[s]

        lo2MP2_bath_list[s] = lo2MP2_bath_s
        lo2MP2_core_list[s] = lo2MP2_core_s
        lo2MP2_vir_list[s]  = lo2MP2_vir_s

        log.info('[%s] Number of newly added bath orbitals = %d (%d from core, %d from virtual)',
                 spin_label, nbath_new_spin[s], nbath_new_core_spin[s], nbath_new_vir_spin[s])
        log.info('')
    # then update the DMET object which is defferent from the ssdmet within each spin
    new_lo_cloes_list = [None, None]
    for s in [0, 1]:
        nbath_old = dmet.nes[s] - nimp

        Q_emb = dmet.lo_cloes[s][:, :nimp + nbath_old]

        new_lo_cloes_s = np.hstack([
            Q_emb,
            lo2MP2_bath_list[s],
            lo2MP2_core_list[s],
            lo2MP2_vir_list[s],
        ])
        new_lo_cloes_list[s] = new_lo_cloes_s

        new_nes_s = dmet.nes[s] + nbath_new_spin[s]
        new_nfo_s = lo2MP2_core_list[s].shape[1]
        new_nfv_s = lo2MP2_vir_list[s].shape[1]

        nes_list = list(dmet.nes)
        nfo_list = list(dmet.nfo)
        nfv_list = list(dmet.nfv)
        nes_list[s] = new_nes_s
        nfo_list[s] = new_nfo_s
        nfv_list[s] = new_nfv_s
        dmet.nes = tuple(nes_list)
        dmet.nfo = tuple(nfo_list)
        dmet.nfv = tuple(nfv_list)

    dmet.lo_cloes = tuple(new_lo_cloes_list)
    if dmet.restore_imp:
        dmet.caoes = (dmet.caolo[0] @ dmet.lo_cloes[0], dmet.caolo[1] @ dmet.lo_cloes[1])
    else:
    # AO-basis coefficients
        dmet.caoes = (dmet.caolo @ dmet.lo_cloes[0], dmet.caolo @ dmet.lo_cloes[1])
    dmet.es_orb = (dmet.caoes[0][:, :dmet.nes[0]], dmet.caoes[1][:, :dmet.nes[1]])
    dmet.fo_orb = (dmet.caoes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]],
                    dmet.caoes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]])
    dmet.fv_orb = (dmet.caoes[0][:, dmet.nes[0]+dmet.nfo[0]:],
                    dmet.caoes[1][:, dmet.nes[1]+dmet.nfo[1]:])

    # embedded integrals and mean-field
    dmet.es_int1e = dmet.make_es_int1e()
    if hasattr(dmet, 'es_cderi') and getattr(dmet, 'es_cderi', None) is not None:
        dmet.log.info("Rebuilding DF 3-index integrals (es_cderi) ...")
        dmet.es_cderi = dmet.make_es_cderi()
    else:
        dmet.es_int2e = dmet.make_es_int2e()
    # 0503
        # Rebuild es_dm: keep old blocks, project new bath from AO DM
    '''    ao_dm = dmet.mf_or_cas.make_rdm1()
    S = dmet.mf_or_cas.get_ovlp()
    new_es_dm_list = [None, None]
    for ss in [0, 1]:
        old_nes = dmet.nes[ss] - nbath_new_spin[ss]
        old_dm = dmet.es_dm[ss]
        new_dm = np.zeros((dmet.nes[ss], dmet.nes[ss]))
        new_dm[:old_nes, :old_nes] = old_dm
        if nbath_new_spin[ss] > 0:
            new_bath_ao = dmet.es_orb[ss][:, old_nes:]
            new_dm[old_nes:, old_nes:] = new_bath_ao.T @ S @ ao_dm[ss] @ S @ new_bath_ao
        new_es_dm_list[ss] = new_dm
    dmet.es_dm = tuple(new_es_dm_list)
    '''
    # 0429 original version but not sultable for large system, where it does not differentiate the core and vir orbs.
    '''    ldm0, ldm1, _, _ = dmet.lowdin_orth()
    dmet.es_dm = dmet.make_es_dm(
        (dmet.lo_cloes[0][:, :dmet.nes[0]], dmet.lo_cloes[1][:, :dmet.nes[1]]),
        (ldm0, ldm1)
    )'''

    # Rebuild es_dm: keep old ES block.
    # For new bath from core BNO (occupied) : project from AO DM.
    # For new bath from virtual BNO (empty) : keep zero. which matches the logic in concentric_occ_localization (for FO→bath) and concentric_vir_localization (for FV→bath).
    ao_dm = dmet.mf_or_cas.make_rdm1()
    S = dmet.mf_or_cas.get_ovlp()
    new_es_dm_list = [None, None]
    for ss in [0, 1]:
        old_nes = dmet.nes[ss] - nbath_new_spin[ss]
        n_core_bath = nbath_new_core_spin[ss]
        n_virt_bath = nbath_new_vir_spin[ss]
        old_dm = dmet.es_dm[ss]
        new_dm = np.zeros((dmet.nes[ss], dmet.nes[ss]))
        new_dm[:old_nes, :old_nes] = old_dm
        # Core BNO bath block: project occupation from AO DM
        if n_core_bath > 0:
            core_bath_ao = dmet.es_orb[ss][:, old_nes:old_nes + n_core_bath]
            new_dm[old_nes:old_nes + n_core_bath,
                   old_nes:old_nes + n_core_bath] = \
                core_bath_ao.T @ S @ ao_dm[ss] @ S @ core_bath_ao
        # Virtual BNO bath block (old_nes+n_core_bath : old_nes+nbath_new)
        # stays zero — these orbitals are empty
        new_es_dm_list[ss] = new_dm
    dmet.es_dm = tuple(new_es_dm_list)



    dmet.es_mf = dmet.UHF()
    dmet.calc_fo_ene()

    dmet.log.info("UMP2 bath done. NES=%s, NFO=%s, NFV=%s; nimp=%d, nbath=(%d, %d)",
                   dmet.nes, dmet.nfo, dmet.nfv,
                   nimp, dmet.nes[0]-nimp, dmet.nes[1]-nimp)

    return dmet


def make_histogram(values, bins, labels=True, binwidth=6, height=10, fill=":", show_number=False, invertx=True, rstrip=True):
    '''
    Modified from https://github.com/BoothGroup/Vayesta/blob/master/vayesta/core/bath/helper.py
    Original author: Max Nusspickel & Charles J. C. Scott
    '''
    hist = np.histogram(values, bins)[0]
    if invertx:
        bins, hist = bins[::-1], hist[::-1]
    hmax = hist.max()
    
    binwidths = [len(str(hval))-2 for hval in hist]

    width = binwidth * len(hist) + sum(binwidths)
    plot = np.zeros((height + show_number, width), dtype=str)
    plot[:] = " "
    if hmax > 0:
        for i, hval in enumerate(hist):
            colstart = i * binwidth + sum(binwidths[:i])
            colend = (i + 1) * binwidth + sum(binwidths[:(i+1)])
            barheight = int(np.rint(height * hval / hmax))
            if barheight == 0:
                continue
            # Top
            plot[-barheight, colstart + 1 : colend - 1] = "_"
            if show_number:
                number = " {:^{w}s}".format("%d" % hval, w=binwidth - 1 + binwidths[i])
                for idx, i in enumerate(range(colstart, colend)):
                    plot[-barheight - 1, i] = number[idx]

            if barheight == 1:
                continue
            # Fill
            if fill:
                plot[-barheight + 1 :, colstart + 1 : colend] = fill
            # Left/right border
            plot[-barheight + 1 :, colstart] = "|"
            plot[-barheight + 1 :, colend - 1] = "|"

    lines = ["".join(plot[r, :].tolist()) for r in range(height)]
    # Baseline
    lines.append("+" + ((width - 2) * "-") + "+")
    
    labelwides = np.hstack([6+np.array(binwidths)[1:],np.array([6])])
    if labels:
        lines += ["{:<{w}}".format("E-0", w=4) + "".join(["{:<{w}}".format("E-%d" % d, w=labelwides[i]) for i,d in enumerate(range(1, 11))])]

    if rstrip:
        lines = [line.rstrip() for line in lines]
    txt = "\n".join(lines)
    return txt