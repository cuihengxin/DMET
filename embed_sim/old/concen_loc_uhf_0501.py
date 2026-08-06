

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

    ldm0, ldm1, _, _ = dmet.lowdin_orth()
    dmet.es_dm = dmet.make_es_dm(
        (dmet.lo_cloes[0][:, :dmet.nes[0]], dmet.lo_cloes[1][:, :dmet.nes[1]]),
        (ldm0, ldm1)
    )

    # Rebuild embedded mean-field
    dmet.es_mf = dmet.UHF()
    dmet.calc_fo_ene()

    dmet.log.info(f"[{spin}] Concentric FO done. "
                    f"NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; "
                    f"nimp={nimp}, nbath=({dmet.nes[0]-nimp}, {dmet.nes[1]-nimp})")

    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_occ):
                if c_shell.shape[1] > 0:
                    dm_shell = c_shell @ c_shell.T.conj()
                    cube_name = f"{dmet.title}_{spin}_shell_{idx}_density_occ.cube"
                    dmet.log.info(f"Exporting {cube_name}")
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
        
    ldm0, ldm1, _, _ = dmet.lowdin_orth()
    dmet.es_dm = dmet.make_es_dm(
        (dmet.lo_cloes[0][:, :dmet.nes[0]], dmet.lo_cloes[1][:, :dmet.nes[1]]),
        (ldm0, ldm1)
    )

    dmet.es_mf = dmet.UHF()
    dmet.calc_fo_ene()

    dmet.log.info(f"[{spin}] Concentric FV done. "
                    f"NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}; "
                    f"nimp={nimp}, nbath=({dmet.nes[0]-nimp}, {dmet.nes[1]-nimp})")

    if ele_density:
        try:
            from pyscf.tools import cubegen
            for idx, c_shell in enumerate(C_vir):
                if c_shell.shape[1] > 0:
                    dm_shell = c_shell @ c_shell.T.conj()
                    cube_name = f"{dmet.title}_{spin}_vir_shell_{idx}_density.cube"
                    dmet.log.info(f"Exporting {cube_name}")
                    cubegen.density(dmet.mol, cube_name, dm_shell)
        except Exception as e:
            dmet.log.warn(f"Failed to export cube files: {e}")

    return dmet