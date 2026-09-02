import numpy as np
from pyscf import lib, gto
from pyscf.lo.orth import lowdin
from functools import reduce

def append_bath_by_env_idx(dmet, env_idx):
    """
    append the bath orbital into the EO space from idx
    """
    if dmet.lo_cloes is None or getattr(dmet, 'caolo', None) is None or getattr(dmet, 'cloao', None) is None:
        raise RuntimeError("Embedded subspace not built or transforms not cached. Run build() first.")

    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    indices_to_move = [] 
    
    for idx in env_idx:
        if idx < nbath:
            dmet.log.warn(f"Index {idx} is already in Bath (current nbath={nbath}), skipping.")
        else:
            indices_to_move.append(idx - nbath)
    
    if not indices_to_move:
        dmet.log.warn("No valid Frozen orbitals selected to append.")
        return

    Q_emb = dmet.lo_cloes[:, :nimp+nbath]
    env_block = dmet.lo_cloes[:, nimp+nbath:] 

    n_shifted_fo = 0
    n_shifted_fv = 0
    
    for local_idx in indices_to_move:
        if local_idx < dmet.nfo:
            n_shifted_fo += 1
        else:
            n_shifted_fv += 1
    
    dmet.log.info(f"Appending Bath: Shifted {n_shifted_fo} from FO, {n_shifted_fv} from FV")

    mask_move = np.zeros(env_block.shape[1], dtype=bool)
    mask_move[indices_to_move] = True
    
    B_new_candidates = env_block[:, mask_move] 
    
    lo2New_bath, _ = np.linalg.qr(B_new_candidates)
    
    indices_all = np.arange(env_block.shape[1])
    indices_remain = indices_all[~mask_move]
    
    idx_remain_fo = [i for i in indices_remain if i < dmet.nfo]
    idx_remain_fv = [i for i in indices_remain if i >= dmet.nfo]
    
    lo2New_core = env_block[:, idx_remain_fo]
    lo2New_vir  = env_block[:, idx_remain_fv]
    
    dmet.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_core, lo2New_vir])
    
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


def analyze_bath_composition(dmet, threshold=0.1):
    if dmet.es_orb is None:
        dmet.log.warn("Embedded subspace not built.")
        return

    nimp = len(dmet.imp_idx)
    nbath = dmet.nes - nimp
    
    bath_orb_coeff = dmet.es_orb[:, nimp:nimp+nbath]
    
    S = dmet.mol.intor_symmetric('int1e_ovlp')
    
    dmet.log.info(f"{'='*20} Bath Orbital Composition Analysis {'='*20}")
    
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
        
        dmet.log.info(f"Bath {ib+1} [Atom]: {', '.join(comp_str)}")

        sorted_ao_idx = np.argsort(np.abs(pop))[::-1]
        orb_details = []
        
        detail_threshold = threshold 
        
        for idx in sorted_ao_idx:
            val = pop[idx]
            if abs(val) > detail_threshold:
                lbl = ao_labels_str[idx].strip()
                orb_details.append(f"{lbl}({val:.2f})")
        
        if orb_details:
            dmet.log.info(f"        [Detail]: {', '.join(orb_details)}")

    dmet.log.info("="*65)


def mapping_ao(dmet, ref_coeff, ref_mol, threshold=0.4):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    nimp = len(dmet.imp_idx)
    nbath_current = dmet.nes - nimp
    
    n_ref_orb = ref_coeff.shape[1]
    
    dmet.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"Reference Orbital Size: {n_ref_orb} | Current Bath Size: {nbath_current}")
    
    env_loc_indices = slice(nimp, nimp + nbath_current + dmet.nfo + dmet.nfv)
    env_orb_AO = dmet.caolo @ dmet.lo_cloes[:, env_loc_indices]
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
        plt.title("Overlap Matrix between Current Bath+Env and Reference Bath")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"Saved overlap matrix heatmap to {dmet.title}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

    dmet.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_orb) 

    dmet.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        if s < threshold:
            dmet.log.debug(f"Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
            continue
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + dmet.nfo:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd(dmet, ref_bath_coeff, ref_mol):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    nimp = len(dmet.imp_idx)
    nbath_current = dmet.nes - nimp
    
    n_ref_bath = ref_bath_coeff.shape[1]
    
    dmet.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"Reference Bath Size: {n_ref_bath} | Current Bath Size: {nbath_current}")
    env_loc_indices = slice(nimp, nimp + nbath_current + dmet.nfo + dmet.nfv)
    env_orb_AO = dmet.caolo @ dmet.lo_cloes[:, env_loc_indices]

    _, S_new_half = lowdin(dmet.mol.intor_symmetric('int1e_ovlp')), lowdin(dmet.mol.intor_symmetric('int1e_ovlp')) @ dmet.mol.intor_symmetric('int1e_ovlp')
    _, S_old_half = lowdin(ref_mol.intor_symmetric('int1e_ovlp')), lowdin(ref_mol.intor_symmetric('int1e_ovlp')) @ ref_mol.intor_symmetric('int1e_ovlp')
    
    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_bath_coeff))
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
        plt.title("Overlap Matrix between Current Bath+Env and Reference Bath")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"Saved overlap matrix heatmap to {dmet.title}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_bath) 

    dmet.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + dmet.nfo:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd2(dmet, ref_coeff, ref_mol):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    nimp_current = len(dmet.imp_idx)
    nbath_current = dmet.nes - nimp_current
    
    n_ref = ref_coeff.shape[1]
    target_nbath = n_ref - nimp_current
    n_needed = target_nbath - nbath_current
    
    dmet.log.info(f"{'='*20} SVD Consistent Bath Search (Full EO Match) {'='*20}")
    dmet.log.info(f"Reference Target Size: {n_ref} | Current Imp Size: {nimp_current}")
    dmet.log.info(f"Current Bath Size:   {nbath_current} | Target Bath Size: {target_nbath}")
    
    if n_needed <= 0:
        dmet.log.info(f"Current Bath size ({nbath_current}) >= Target ({target_nbath}). No extension needed based on size.")
        dmet.log.info("="*65)
        return []
        
    dmet.log.info(f"Target: Recover {n_needed} orbitals from Frozen space to match reference total size.")

    env_loc_indices = slice(nimp_current, nimp_current + nbath_current + dmet.nfo + dmet.nfv)
    env_orb_AO = dmet.caolo @ dmet.lo_cloes[:, env_loc_indices]

    _, S_new_half = lowdin(dmet.mol.intor_symmetric('int1e_ovlp')), lowdin(dmet.mol.intor_symmetric('int1e_ovlp')) @ dmet.mol.intor_symmetric('int1e_ovlp')
    _, S_old_half = lowdin(ref_mol.intor_symmetric('int1e_ovlp')), lowdin(ref_mol.intor_symmetric('int1e_ovlp')) @ ref_mol.intor_symmetric('int1e_ovlp')
    
    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_coeff))

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"==> All Singular Values (sigma) for Reference Full EO SVD: {np.round(sigma, 4)}")

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
            dmet.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + dmet.nfo:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices


def find_bath_indices_from_reference_svd3(dmet, ref_eo_coeff, ref_mol, threshold=0.4):
    if dmet.lo_cloes is None:
        raise RuntimeError("Run build() first.")

    nimp = len(dmet.imp_idx)
    nbath_current = dmet.nes - nimp
    
    n_ref_eo = ref_eo_coeff.shape[1]
    
    dmet.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
    dmet.log.info(f"Reference EO Size: {n_ref_eo} | Reference BATH Size: {n_ref_eo-nimp} | Current Bath Size: {nbath_current}")
    
    env_loc_indices = slice(nimp, nimp + nbath_current + dmet.nfo + dmet.nfv)
    env_orb_AO = dmet.caolo @ dmet.lo_cloes[:, env_loc_indices]
    
    _, S_new_half = lowdin(dmet.mol.intor_symmetric('int1e_ovlp')), lowdin(dmet.mol.intor_symmetric('int1e_ovlp')) @ dmet.mol.intor_symmetric('int1e_ovlp')
    _, S_old_half = lowdin(ref_mol.intor_symmetric('int1e_ovlp')), lowdin(ref_mol.intor_symmetric('int1e_ovlp')) @ ref_mol.intor_symmetric('int1e_ovlp')
    
    ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_eo_coeff))
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
        plt.title("Overlap Matrix between Current Bath+Env and Reference Bath")
        plt.xlabel("Reference Bath Index")
        plt.ylabel("Current Bath+Env Index")
        plt.savefig(f"{dmet.title}_overlap_matrix.png")
        plt.close()
        dmet.log.info(f"Saved overlap matrix heatmap to {dmet.title}_overlap_matrix.png")
    except ImportError:
        dmet.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

    U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    dmet.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

    recommended_indices = []
    used_env = set()
    num_modes_to_check = min(len(sigma), n_ref_eo - nimp) 

    dmet.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

    for i in range(num_modes_to_check):
        s = sigma[i]
        if s < threshold:
            dmet.log.debug(f"Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
            continue
        weights = np.abs(U[:, i])
        sorted_env = np.argsort(weights)[::-1]
        
        best_env_idx = -1
        for idx in sorted_env:
            if idx not in used_env:
                best_env_idx = idx
                break
        
        if best_env_idx == -1:
            dmet.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
            continue

        used_env.add(best_env_idx)
        
        status = ""
        if best_env_idx < nbath_current:
            status = "Match Current Bath (Skipped)"
        elif best_env_idx < nbath_current + dmet.nfo:
            status = f"Recover FO (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        else:
            status = f"Recover FV (idx {best_env_idx})"
            recommended_indices.append(best_env_idx)
        
        dmet.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

    dmet.log.info(f"Indices recovered from frozen space: {recommended_indices}")
    dmet.log.info("="*65)
    return recommended_indices