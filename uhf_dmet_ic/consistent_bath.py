import numpy as np
from pyscf import lib, gto
from pyscf.lo.orth import lowdin
from functools import reduce
def lowdin_orth(mol, ovlp=None):
    # lowdin orthonormalize
    if ovlp is None:
        s = mol.intor_symmetric('int1e_ovlp')
    else:
        s = ovlp
    caolo, cloao = lowdin(s), lowdin(s) @ s # caolo=lowdin(s)=s^-1/2, cloao=lowdin(s)@s=s^1/2
    return caolo, cloao

def append_bath_by_env_idx(dmet, env_idx, spin='alpha'):
    """
    append the bath orbital into the EO space from idx
    """
    if dmet.lo_cloes is None or getattr(dmet, 'caolo', None) is None or getattr(dmet, 'cloao', None) is None:
        raise RuntimeError("Embedded subspace not built or transforms not cached. Run build() first.")

    is_uhf = isinstance(dmet.nes, tuple)
    if is_uhf:
        s = 0 if spin == 'alpha' else 1
        nimp = len(dmet.imp_idx)
        nbath = dmet.nes[s] - nimp
        lo_cloes_target = dmet.lo_cloes[s]
        nfo_target = dmet.nfo[s]
    else:
        nimp = len(dmet.imp_idx)
        nbath = dmet.nes - nimp
        lo_cloes_target = dmet.lo_cloes
        nfo_target = dmet.nfo

    indices_to_move = [] 
    
    for idx in env_idx:
        if idx < nbath:
            dmet.log.warn(f"[{spin}] Index {idx} is already in Bath (current nbath={nbath}), skipping.")
        else:
            indices_to_move.append(idx - nbath)
    
    if not indices_to_move:
        dmet.log.warn(f"[{spin}] No valid Frozen orbitals selected to append.")
        return dmet

    Q_emb = lo_cloes_target[:, :nimp+nbath]
    env_block = lo_cloes_target[:, nimp+nbath:] 

    n_shifted_fo = 0
    n_shifted_fv = 0
    
    for local_idx in indices_to_move:
        if local_idx < nfo_target:
            n_shifted_fo += 1
        else:
            n_shifted_fv += 1
    
    dmet.log.info(f"[{spin}] Appending Bath: Shifted {n_shifted_fo} from FO, {n_shifted_fv} from FV")
    # fo+fv index
    mask_move = np.zeros(env_block.shape[1], dtype=bool)
    mask_move[indices_to_move] = True
    
    B_new_candidates = env_block[:, mask_move] 
    
    lo2New_bath, _ = np.linalg.qr(B_new_candidates)
    
    indices_all = np.arange(env_block.shape[1])
    indices_remain = indices_all[~mask_move]
    
    idx_remain_fo = [i for i in indices_remain if i < nfo_target]
    idx_remain_fv = [i for i in indices_remain if i >= nfo_target]
    
    lo2New_core = env_block[:, idx_remain_fo]
    lo2New_vir  = env_block[:, idx_remain_fv]
    
    new_lo_cloes_s = np.hstack([Q_emb, lo2New_bath, lo2New_core, lo2New_vir])
    
    if is_uhf:
        new_nes_s = dmet.nes[s] + lo2New_bath.shape[1] 
        new_nfo_s = dmet.nfo[s] - n_shifted_fo
        new_nfv_s = dmet.nfv[s] - n_shifted_fv
        
        # updated lo_cloes
        lo_cloes_list = list(dmet.lo_cloes)
        lo_cloes_list[s] = new_lo_cloes_s
        dmet.lo_cloes = tuple(lo_cloes_list)

        nes_list = list(dmet.nes)
        nfo_list = list(dmet.nfo)
        nfv_list = list(dmet.nfv)
        nes_list[s] = new_nes_s
        nfo_list[s] = new_nfo_s
        nfv_list[s] = new_nfv_s
        dmet.nes = tuple(nes_list)
        dmet.nfo = tuple(nfo_list)
        dmet.nfv = tuple(nfv_list)

        # Rebuild AO-basis coefficients
        dmet.caoes = (dmet.caolo[0] @ dmet.lo_cloes[0], dmet.caolo[1] @ dmet.lo_cloes[1])
        dmet.es_orb = (dmet.caoes[0][:, :dmet.nes[0]], dmet.caoes[1][:, :dmet.nes[1]])
        dmet.fo_orb = (dmet.caoes[0][:, dmet.nes[0]:dmet.nes[0]+dmet.nfo[0]],
                        dmet.caoes[1][:, dmet.nes[1]:dmet.nes[1]+dmet.nfo[1]])
        dmet.fv_orb = (dmet.caoes[0][:, dmet.nes[0]+dmet.nfo[0]:],
                        dmet.caoes[1][:, dmet.nes[1]+dmet.nfo[1]:])

        # Rebuild embedded 1e/2e integrals
        dmet.es_int1e = dmet.make_es_int1e()
        if hasattr(dmet, 'es_cderi') and getattr(dmet, 'es_cderi', None) is not None:
            dmet.log.info(f"[{spin}] Rebuilding DF 3-index integrals (es_cderi) ...")
            dmet.es_cderi = dmet.make_es_cderi()
        else:
            dmet.es_int2e = dmet.make_es_int2e()

        # Rebuild es_dm: keep original ES block, add FO->bath occupations
        n_shifted = lo2New_bath.shape[1]
        old_nes_s = dmet.nes[s] - n_shifted
        old_es_dm_s = dmet.es_dm[s]
        new_es_dm_s = np.zeros((dmet.nes[s], dmet.nes[s]))
        new_es_dm_s[:old_nes_s, :old_nes_s] = old_es_dm_s
        if n_shifted > 0:
            ao_dm = dmet.mf_or_cas.make_rdm1()
            S = dmet.mf_or_cas.get_ovlp()
            new_bath_ao = dmet.es_orb[s][:, old_nes_s:]
            new_bath_dm = new_bath_ao.T @ S @ ao_dm[s] @ S @ new_bath_ao
            new_es_dm_s[old_nes_s:, old_nes_s:] = new_bath_dm

        es_dm_list = list(dmet.es_dm)
        es_dm_list[s] = new_es_dm_s
        dmet.es_dm = tuple(es_dm_list)

        # Rebuild embedded mean-field
        dmet.es_mf = dmet.UHF()
        if hasattr(dmet, 'calc_fo_ene'):
            dmet.calc_fo_ene()

        dmet.log.info(f"[{spin}] Bath appended. New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}")
        if hasattr(dmet, 'fo_ene') and not callable(getattr(dmet, 'fo_ene')):
            dmet.log.info(f"[{spin}] Frozen Energy updated: {dmet.fo_ene:.6f}")
        dmet.log.info(f'[{spin}] number of impurity orbitals = {nimp}')
        dmet.log.info(f'[{spin}] number of bath orbitals = {dmet.nes[s] - nimp}')
        dmet.log.info(f'[{spin}] number of embedded cluster orbitals = {dmet.nes[s]}')
        dmet.log.info(f'[{spin}] percentage of embedded cluster orbitals = {((dmet.nes[s])/dmet.mol.nao)*100:.2f}%%')
        dmet.log.info(f'[{spin}] percentage of frozen orbitals = {((dmet.nfo[s]+dmet.nfv[s])/dmet.mol.nao)*100:.2f}%%')

    else:
        dmet.lo_cloes = new_lo_cloes_s
        dmet.nes  = nimp + nbath + lo2New_bath.shape[1] 
        dmet.nfo -= n_shifted_fo
        dmet.nfv -= n_shifted_fv
        
        dmet.es_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, :dmet.nes])
        dmet.fo_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes : dmet.nes+dmet.nfo])
        dmet.fv_orb = lib.dot(dmet.caolo, dmet.lo_cloes[:, dmet.nes+dmet.nfo :])

        dmet.es_int1e = dmet.make_es_int1e()
        if hasattr(dmet, 'es_cderi'):
            dmet.es_cderi = dmet.make_es_cderi()
        else:
            dmet.es_int2e = dmet.make_es_int2e()

        dm_arg = getattr(dmet, 'dm_pair', None) if getattr(dmet, 'open_shell', False) and getattr(dmet, 'dm_pair', None) is not None else getattr(dmet, 'dm', None)
        dmet.es_dm = dmet.make_es_dm(getattr(dmet, 'open_shell', False), dmet.lo_cloes[:, :dmet.nes], dmet.cloao, dm_arg)
        
        dmet.es_mf = dmet.ROHF()
        if hasattr(dmet, 'calc_fo_ene'):
            dmet.calc_fo_ene() 
        elif hasattr(dmet, 'fo_ene') and callable(getattr(dmet, 'fo_ene')):
            dmet.fo_ene()
        
        dmet.log.info(f"Bath appended. New sizes: NES={dmet.nes}, NFO={dmet.nfo}, NFV={dmet.nfv}")
        if hasattr(dmet, 'fo_ene') and not callable(getattr(dmet, 'fo_ene')):
            dmet.log.info(f"Frozen Energy updated: {dmet.fo_ene:.6f}")
        dmet.log.info(f'number of impurity orbitals = {nimp}')
        dmet.log.info(f'number of bath orbitals = {dmet.nes - nimp}')
        dmet.log.info(f'number of embedded cluster orbitals = {dmet.nes}')
        dmet.log.info(f'percentage of embedded cluster orbitals = {((dmet.nes)/dmet.mol.nao)*100:.2f}%%')
        dmet.log.info(f'percentage of frozen orbitals = {((dmet.nfo+dmet.nfv)/dmet.mol.nao)*100:.2f}%%')
    return dmet

def analyze_bath_composition(dmet, threshold=0.1, spin='alpha'):
    if dmet.es_orb is None:
        dmet.log.warn("Embedded subspace not built.")
        return

    is_uhf = isinstance(dmet.nes, tuple)
    s_idx = 0 if spin == 'alpha' else 1
    
    nimp = len(dmet.imp_idx)
    if is_uhf:
        nbath = dmet.nes[s_idx] - nimp
        bath_orb_coeff = dmet.es_orb[s_idx][:, nimp:nimp+nbath]
    else:
        nbath = dmet.nes - nimp
        bath_orb_coeff = dmet.es_orb[:, nimp:nimp+nbath]
    
    S = dmet.mol.intor_symmetric('int1e_ovlp')
    
    dmet.log.info(f"[{spin}] {'='*20} Bath Orbital Composition Analysis {'='*20}")
    
    total_atoms = [dmet.mol.atom_symbol(i) for i in range(dmet.mol.natm)]
    
    ao_labels_str = dmet.mol.ao_labels() 
    ao_labels_fmt = dmet.mol.ao_labels(fmt=None) 

    for ib in range(nbath):
        C = bath_orb_coeff[:, ib]
        SC = np.dot(S, C)
        pop = C * SC 
        
        atom_pops = np.zeros(dmet.mol.natm)
        for iao, label in enumerate(ao_labels_fmt):
            atom_id = label[0]
            atom_pops[atom_id] += pop[iao]
        
        sorted_indices = np.argsort(np.abs(atom_pops))[::-1]
        
        comp_str = []
        for idx in sorted_indices:
            val = atom_pops[idx]
            if abs(val) > threshold:
                comp_str.append(f"{total_atoms[idx]}{idx}({val:.2f})")
        
        dmet.log.info(f"[{spin}] Bath {ib+1} [Atom]: {', '.join(comp_str)}")

        sorted_ao_idx = np.argsort(np.abs(pop))[::-1]
        orb_details = []
        
        detail_threshold = threshold 
        
        for idx in sorted_ao_idx:
            val = pop[idx]
            if abs(val) > detail_threshold:
                lbl = ao_labels_str[idx].strip()
                orb_details.append(f"{lbl}({val:.2f})")
        
        if orb_details:
            dmet.log.info(f"[{spin}]         [Detail]: {', '.join(orb_details)}")

    dmet.log.info("="*65)


def mapping_ao(dmet, ref_coeff, ref_mol, threshold=0.4, spin='alpha'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    is_uhf = isinstance(dmet.nes, tuple)
    s_idx = 0 if spin == 'alpha' else 1
    nimp = len(dmet.imp_idx)
    
    if is_uhf:
        nbath_current = dmet.nes[s_idx] - nimp
        lo_cloes_target = dmet.lo_cloes[s_idx]
        nfo_target = dmet.nfo[s_idx]
        nfv_target = dmet.nfv[s_idx]
    else:
        nbath_current = dmet.nes - nimp
        lo_cloes_target = dmet.lo_cloes
        nfo_target = dmet.nfo
        nfv_target = dmet.nfv
    
    n_ref_orb = ref_coeff.shape[1]
    
    dmet.log.info(f"[{spin}] {'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"[{spin}] Reference Orbital Size: {n_ref_orb} | Current Bath Size: {nbath_current}")
    
    env_loc_indices = slice(nimp, nimp + nbath_current + nfo_target + nfv_target)
    env_orb_AO = dmet.caolo @ lo_cloes_target[:, env_loc_indices]
    S_ref = ref_mol.intor_symmetric('int1e_ovlp')
    S_ref_inv = np.linalg.inv(S_ref)
    ao_ovlp = gto.mole.intor_cross('int1e_ovlp', ref_mol, dmet.mol)
    env_orb_refAO = S_ref_inv @ ao_ovlp @ env_orb_AO
    ovlp_mat = env_orb_refAO.T.conj() @ S_ref @ ref_coeff
    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
        plt.title(f"Overlap Matrix between Current Bath+Env and Reference Bath ({spin})")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_{spin}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"[{spin}] Saved overlap matrix heatmap to {dmet.title}_{spin}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn(f"[{spin}] matplotlib or seaborn not installed, skipping heatmap.")

    dmet.log.info(f"[{spin}] ==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_orb) 

    dmet.log.info(f"[{spin}] Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        if s < threshold:
            dmet.log.debug(f"[{spin}] Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
            continue
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"[{spin}] SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + nfo_target:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"[{spin}] Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"[{spin}] Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd(dmet, ref_bath_coeff, ref_mol, threshold=0.1, spin='alpha'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    is_uhf = isinstance(dmet.nes, tuple)
    s_idx = 0 if spin == 'alpha' else 1
    nimp = len(dmet.imp_idx)
    
    if is_uhf:
        nbath_current = dmet.nes[s_idx] - nimp
        lo_cloes_target = dmet.lo_cloes[s_idx]
        nfo_target = dmet.nfo[s_idx]
        nfv_target = dmet.nfv[s_idx]
    else:
        nbath_current = dmet.nes - nimp
        lo_cloes_target = dmet.lo_cloes
        nfo_target = dmet.nfo
        nfv_target = dmet.nfv
    
    n_ref_bath = ref_bath_coeff.shape[1]
    
    dmet.log.info(f"[{spin}] {'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"[{spin}] Reference Bath Size: {n_ref_bath} | Current Bath Size: {nbath_current}")
    env_loc_indices = slice(nimp, nimp + nbath_current + nfo_target + nfv_target)
    env_orb_AO = dmet.caolo[s_idx] @ lo_cloes_target[:, env_loc_indices]
    _, S_new_half = lowdin_orth(dmet.mol)
    _, S_old_half = lowdin_orth(ref_mol)
    #_, S_new_half = lowdin(dmet.mol.intor_symmetric('int1e_ovlp')), lowdin(dmet.mol.intor_symmetric('int1e_ovlp')) @ dmet.mol.intor_symmetric('int1e_ovlp')
    #_, S_old_half = lowdin(ref_mol.intor_symmetric('int1e_ovlp')), lowdin(ref_mol.intor_symmetric('int1e_ovlp')) @ ref_mol.intor_symmetric('int1e_ovlp')
    
    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_bath_coeff))
    '''    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
        plt.title(f"Overlap Matrix between Current Bath+Env and Reference Bath ({spin})")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_{spin}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"[{spin}] Saved overlap matrix heatmap to {dmet.title}_{spin}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn(f"[{spin}] matplotlib or seaborn not installed, skipping heatmap.")'''

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"[{spin}] ==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_bath) 

    dmet.log.info(f"[{spin}] Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        if s < threshold:
            dmet.log.debug(f"[{spin}] Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
            continue
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"[{spin}] SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + nfo_target:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"[{spin}] Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"[{spin}] Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd2(dmet, ref_coeff, ref_mol, spin='alpha'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    is_uhf = isinstance(dmet.nes, tuple)
    s_idx = 0 if spin == 'alpha' else 1
    nimp_current = len(dmet.imp_idx)

    if is_uhf:
        nbath_current = dmet.nes[s_idx] - nimp_current
        lo_cloes_target = dmet.lo_cloes[s_idx]
        nfo_target = dmet.nfo[s_idx]
        nfv_target = dmet.nfv[s_idx]
    else:
        nbath_current = dmet.nes - nimp_current
        lo_cloes_target = dmet.lo_cloes
        nfo_target = dmet.nfo
        nfv_target = dmet.nfv

    n_ref = ref_coeff.shape[1]
    target_nbath = n_ref - nimp_current
    n_needed = target_nbath - nbath_current
    
    dmet.log.info(f"[{spin}] {'='*20} SVD Consistent Bath Search (Full EO Match) {'='*20}")
    dmet.log.info(f"[{spin}] Reference Target Size: {n_ref} | Current Imp Size: {nimp_current}")
    dmet.log.info(f"[{spin}] Current Bath Size:   {nbath_current} | Target Bath Size: {target_nbath}")
    
    if n_needed <= 0:
        dmet.log.info(f"[{spin}] Current Bath size ({nbath_current}) >= Target ({target_nbath}). No extension needed based on size.")
        dmet.log.info("="*65)
        return []
        
    dmet.log.info(f"[{spin}] Target: Recover {n_needed} orbitals from Frozen space to match reference total size.")

    env_loc_indices = slice(nimp_current, nimp_current + nbath_current + nfo_target + nfv_target)
    env_orb_AO = dmet.caolo[s_idx] @ lo_cloes_target[:, env_loc_indices]

    _, S_new_half = lowdin_orth(dmet.mol)
    _, S_old_half = lowdin_orth(ref_mol)
    print("env_orb_AO.T shape:", env_orb_AO.T.conj().shape)
    print("S_new_half shape:", S_new_half.shape)
    print("S_old_half shape:", S_old_half.shape)
    print("ref_bath_coeff shape:", ref_coeff.shape)

    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_coeff))

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"[{spin}] ==> All Singular Values (sigma) for Reference Full EO SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = len(sigma)

    for i in range(num_modes_to_check):
        if len(recommended_indices) >= n_needed:
            break 

        s = sigma[i]
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"[{spin}] SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + nfo_target:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"[{spin}] Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"[{spin}] Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd3(dmet, ref_eo_coeff, ref_mol, threshold=0.4, spin='alpha'):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    is_uhf = isinstance(dmet.nes, tuple)
    s_idx = 0 if spin == 'alpha' else 1
    nimp = len(dmet.imp_idx)

    if is_uhf:
        nbath_current = dmet.nes[s_idx] - nimp
        lo_cloes_target = dmet.lo_cloes[s_idx]
        nfo_target = dmet.nfo[s_idx]
        nfv_target = dmet.nfv[s_idx]
    else:
        nbath_current = dmet.nes - nimp
        lo_cloes_target = dmet.lo_cloes
        nfo_target = dmet.nfo
        nfv_target = dmet.nfv
    
    n_ref_eo = ref_eo_coeff.shape[1]
    
    dmet.log.info(f"[{spin}] {'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"[{spin}] Reference EO Size: {n_ref_eo} | Reference BATH Size: {n_ref_eo-nimp} | Current Bath Size: {nbath_current}")
    
    env_loc_indices = slice(nimp, nimp + nbath_current + nfo_target + nfv_target)
    env_orb_AO = dmet.caolo[s_idx] @ lo_cloes_target[:, env_loc_indices]

    _, S_new_half = lowdin_orth(dmet.mol)
    _, S_old_half = lowdin_orth(ref_mol)

    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_eo_coeff))
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
        plt.title(f"Overlap Matrix between Current Bath+Env and Reference Bath ({spin})")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_{spin}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"Saved overlap matrix heatmap to {dmet.title}_{spin}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn(f"[{spin}] matplotlib or seaborn not installed, skipping heatmap.")

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"[{spin}] ==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_eo - nimp) 

    dmet.log.info(f"[{spin}] Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        if s < threshold:
            dmet.log.debug(f"[{spin}] Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
            continue
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"[{spin}] SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + nfo_target:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"[{spin}] Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"[{spin}] Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def iao_analysis(dmet):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    nimp = len(dmet.imp_idx)
    nbath_current = dmet.nes - nimp
    
    dmet.log.info(f" Current Bath Size: {nbath_current}")
    caoes = getattr(dmet, 'caolo') @ dmet.lo_cloes
    
    bath_orb = caoes[:, nimp:nimp+nbath_current]
    fo_orb = caoes[:, nimp+nbath_current : nimp+nbath_current+dmet.nfo]
    fv_orb = caoes[:, nimp+nbath_current+dmet.nfo : nimp+nbath_current+dmet.nfo+dmet.nfv]
    imp_orb = caoes[:, :nimp]
    
    mf_or_cas = getattr(dmet, 'mf_or_cas')
    
    mo_occ = mf_or_cas.mo_coeff[:, mf_or_cas.mo_occ>1e-3]
    a = lo.iao.iao(dmet.mol, mo_occ)
    a = lo.vec_lowdin(a, dmet.mol.intor_symmetric('int1e_ovlp'))
    mo_occ_iao = reduce(np.dot, (a.T, dmet.mol.intor_symmetric('int1e_ovlp'), mo_occ))
    
    dm = np.dot(mo_occ_iao, mo_occ_iao.T) * 2
    assert(abs(dm.trace() - dmet.mol.nelectron) < 1e-13)
    
    pmol = dmet.mol.copy()
    pmol.build(False, False, basis='minao')
    mf_or_cas.mulliken_pop(pmol, dm, s=np.eye(pmol.nao_nr()))
    
    t_ao = dmet.mol.intor('int1e_kin')
    
    def get_orb_kinetic_energy(idx, orb_set):
        c_i = orb_set[:, idx]
        t_mo = c_i.T @ t_ao @ c_i
        print(f"Kinetic Energy for MO index {idx} : {t_mo:.8f} Hartree")
        return t_mo

    def analyze_orb_iao(env_orb_AO, idx, a, mf, pmol):
        target_env = env_orb_AO[:, idx:idx+1]
        target_env_iao = reduce(np.dot, (a.T, mf.get_ovlp(), target_env))
        dm_target = np.dot(target_env_iao, target_env_iao.T)
        
        print(f"\n--- IAO Population for Environment/Bath Orbital index {idx} ---")
        pop, charge = mf.mulliken_pop(pmol, dm_target, s=np.eye(pmol.nao_nr()))
        return pop    

    kinetic_energy = []
    iao_pop = []
    
    print("\nImpurity orbitals:")
    for idx in range(imp_orb.shape[1]):
        iao_pop.append(analyze_orb_iao(imp_orb, idx, a, mf_or_cas, pmol))
        kinetic_energy.append(get_orb_kinetic_energy(idx, imp_orb))
        
    print(f"\nBath orbitals:")
    for idx in range(bath_orb.shape[1]):
        iao_pop.append(analyze_orb_iao(bath_orb, idx, a, mf_or_cas, pmol))
        kinetic_energy.append(get_orb_kinetic_energy(idx, bath_orb))
        
    print(f"\nFrozen occupied orbitals:")
    for idx in range(fo_orb.shape[1]):
        iao_pop.append(analyze_orb_iao(fo_orb, idx, a, mf_or_cas, pmol))
        kinetic_energy.append(get_orb_kinetic_energy(idx, fo_orb))
        
    print("\nFrozen virtual orbitals:")
    for idx in range(fv_orb.shape[1]):
        iao_pop.append(analyze_orb_iao(fv_orb, idx, a, mf_or_cas, pmol))
        kinetic_energy.append(get_orb_kinetic_energy(idx, fv_orb))
        
    return np.array(iao_pop), np.array(kinetic_energy)