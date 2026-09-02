import numpy as np

from embed_sim.consistent_bath import (
    analyze_bath_composition,
    find_bath_indices_from_reference_svd,
    find_bath_indices_from_reference_svd2,
    find_bath_indices_from_reference_svd3,
    mapping_ao,
    append_bath_by_env_idx
)
from embed_sim.env_analysis_utils import iao_analysis

# to sweep the curve, please choose 3 ways. sweep method can be used combined with localization of virtual orbitals, called by concentric_loc.localize_environment_spaces
# one is to use the first structure as the reference, and keep the bath number consistent with the first one. that is the "consistent" way. the other is to use the previous structure as the reference, and append the bath according to  the previous one. that is the "sweep" way.


def from_ref_get_newbath4new_geom2(mydmet_ref, mydmet_new):
    nimp_ref = len(mydmet_ref.imp_idx)
    nbath_ref = mydmet_ref.nes - nimp_ref
    ref_bath_ao = mydmet_ref.es_orb[:, nimp_ref:nimp_ref+nbath_ref] 
    print(f"NUMBER of BATH ORBITALS in the old system: {nbath_ref}")
    print("="*50)
    print("Analisis of the OLD SYSTEM:")
    analyze_bath_composition(mydmet_ref)
    print("\n" + "="*50 + "\n")
    idxs_to_recover = find_bath_indices_from_reference_svd(mydmet_new, ref_bath_ao, ref_mol=mydmet_ref.mol)
    print(f"Restoring {len(idxs_to_recover)} bath orbitals that were lost to maintain consistency...")
    append_bath_by_env_idx(mydmet_new, idxs_to_recover)
    print("========= DMET APPEND =========")
    print("NUMBER of BATH ORBITALS", mydmet_new.nes - len(mydmet_new.imp_idx))
    print("NUMBER of EO space orbitals after appending:", mydmet_new.nes)
    print("Indices of the new bath orbitals added to maintain consistency:", idxs_to_recover)
    print("="*65)
    print("Analisis of the NEW BATH SYSTEM after restoration:")
    analyze_bath_composition(mydmet_new)
    return mydmet_new

def from_ref_get_newbath4new_geom3(mydmet_ref, mydmet_new):
    nimp_ref = len(mydmet_ref.imp_idx)
    nbath_ref = mydmet_ref.nes - nimp_ref
    ref_eo_ao = mydmet_ref.es_orb # 这里可能是来回扫的问题所在，用es_orb还是 bath
    print(f"NUMBER of BATH ORBITALS in the old system: {nbath_ref}")
    print("="*50)
    print("Analisis of the OLD SYSTEM:")
    analyze_bath_composition(mydmet_ref)
    print("\n" + "="*50 + "\n")
    idxs_to_recover = find_bath_indices_from_reference_svd3(mydmet_new, ref_eo_ao, ref_mol=mydmet_ref.mol)
    print(f"Restoring {len(idxs_to_recover)} bath orbitals that were lost to maintain consistency...")
    append_bath_by_env_idx(mydmet_new, idxs_to_recover)
    print("========= DMET APPEND =========")
    print("NUMBER of BATH ORBITALS", mydmet_new.nes - len(mydmet_new.imp_idx))
    print("NUMBER of EO space orbitals after appending:", mydmet_new.nes)
    print("Indices of the new bath orbitals added to maintain consistency:", idxs_to_recover)
    print("="*65)
    print("Analisis of the NEW BATH SYSTEM after restoration:")
    analyze_bath_composition(mydmet_new)
    return mydmet_new
def from_ref_get_newbath4new_geom4(mydmet_ref, mydmet_new):
    nimp_ref = len(mydmet_ref.imp_idx)
    nbath_ref = mydmet_ref.nes - nimp_ref
    #ref_ao = mydmet_ref.es_orb[:, nimp_ref:nimp_ref+nbath_ref]
    ref_ao = mydmet_ref.es_orb  # 这里可能是来回扫的问题所在，用es_orb还是 bath 测完发现还是用参考结构的es_orb和新结构的bath env比较
    #ref_ao = np.hstack((mydmet_ref.es_orb, mydmet_ref.fo_orb, mydmet_ref.fv_orb))
    print(f"NUMBER of BATH ORBITALS in the old system: {nbath_ref}")
    print("="*50)
    print("Analisis of the OLD SYSTEM:")
    analyze_bath_composition(mydmet_ref)
    print("\n" + "="*50 + "\n")
    idxs_to_recover = mapping_ao(mydmet_new, ref_ao, ref_mol=mydmet_ref.mol)
    print(f"Restoring {len(idxs_to_recover)} bath orbitals that were lost to maintain consistency...")
    append_bath_by_env_idx(mydmet_new, idxs_to_recover)
    print("========= DMET APPEND =========")
    print("NUMBER of BATH ORBITALS", mydmet_new.nes - len(mydmet_new.imp_idx))
    print("NUMBER of EO space orbitals after appending:", mydmet_new.nes)
    print("Indices of the new bath orbitals added to maintain consistency:", idxs_to_recover)
    print("="*65)
    print("Analisis of the NEW BATH SYSTEM after restoration:")
    analyze_bath_composition(mydmet_new)
    return mydmet_new
def from_ref_get_newbath4new_geom(mydmet_ref, mydmet_new):
    nimp_ref = len(mydmet_ref.imp_idx)
    nbath_ref = mydmet_ref.nes - nimp_ref
    ref_bath_ao = mydmet_ref.es_orb
    print(f"NUMBER of BATH ORBITALS in the old system: {nbath_ref}")
    print("="*50)
    print("Analisis of the OLD SYSTEM:")
    analyze_bath_composition(mydmet_ref)
    print("\n" + "="*50 + "\n")
    idxs_to_recover = find_bath_indices_from_reference_svd2(mydmet_new, ref_bath_ao, ref_mol=mydmet_ref.mol)
    print(f"Restoring {len(idxs_to_recover)} bath orbitals that were lost to maintain consistency...")
    append_bath_by_env_idx(mydmet_new, idxs_to_recover)
    print("========= DMET APPEND =========")
    print("NUMBER of BATH ORBITALS", mydmet_new.nes - len(mydmet_new.imp_idx))
    print("NUMBER of EO space orbitals after appending:", mydmet_new.nes)
    print("Indices of the new bath orbitals added to maintain consistency:", idxs_to_recover)
    print("="*65)
    print("Analisis of the NEW BATH SYSTEM after restoration:")
    analyze_bath_composition(mydmet_new)
    return mydmet_new
def map_orbitals(ref_dmet, new_dmet, name1, name2, iao_threshold=0.1, kinetic_threshold=0.1):
    ref_iao, ref_kinetic = iao_analysis(ref_dmet)
    new_iao, new_kinetic = iao_analysis(new_dmet)
    # need to calculate the orbital mapping between two sets of orbitals
    # firstly we can compare the orbitals of imp
    # ref_iao follows the order of imp bath fo fv
    ref_nimp = len(ref_dmet.imp_idx)
    new_nimp = len(new_dmet.imp_idx)
    ref_imp_iao = ref_iao[:ref_nimp]
    new_imp_iao = new_iao[:new_nimp]

    ref_kinetic_imp = ref_kinetic[:ref_nimp]
    new_kinetic_imp = new_kinetic[:new_nimp]

    ref_iao_bath = ref_iao[ref_nimp : ref_dmet.nes]
    new_iao_bath = new_iao[new_nimp : new_dmet.nes]
    ref_kinetic_bath = ref_kinetic[ref_nimp : ref_dmet.nes]
    new_kinetic_bath = new_kinetic[new_nimp : new_dmet.nes]
    bath_imp_list = []
    bath_kinetic_list = []
    # comparing the bath orbitals
    for i, ii in enumerate(ref_iao_bath):
        for j, jj in enumerate(new_iao_bath):
            print(f"Comparing IAO population of ref bath orbital {i} with new bath orbital {j}:")
            pop_diff = np.sum(np.abs(ii - jj))
            print(f"Population difference: {pop_diff}")
            print("-" * 30)
            bath_imp_list.append((i, j, pop_diff))
    print(" shape of iao_bath_list: ", np.array(bath_imp_list).shape)
    print("\n" + "=" * 50 + "\n")
    for i, k_ref in enumerate(ref_kinetic_bath):
        for j, k_new in enumerate(new_kinetic_bath):
            kine_diff = np.abs(k_ref - k_new)
            print(f"Comparing kinetic energy of ref bath orbital {i} with new bath orbital {j}:")
            print(f"Kinetic energy difference: {kine_diff}")
            bath_kinetic_list.append((i, j, kine_diff))
    print(" shape of kinetic_bath_list: ", np.array(bath_kinetic_list).shape)
    print("\n" + "=" * 50 + "\n")
    diff_bath = np.zeros((len(ref_iao_bath), len(new_iao_bath)))
    for i, j, pop_diff in bath_imp_list:
        diff_bath[i, j] = pop_diff
    diff_bath_kinetic = np.zeros((len(ref_kinetic_bath), len(new_kinetic_bath)))
    for i, j, kine_diff in bath_kinetic_list:
        diff_bath_kinetic[i, j] = kine_diff
    bath_map_list = []
    for i in range(diff_bath.shape[0]):
        for j in range(diff_bath.shape[1]):
            if diff_bath[i, j] < iao_threshold and diff_bath_kinetic[i, j] < kinetic_threshold:
                print(f"Orbital {i} in ref bath matches with orbital {j} in new bath with population difference {diff_bath[i, j]:.4f} and kinetic energy difference {diff_bath_kinetic[i, j]:.4f}")
                bath_map_list.append((i, j , diff_bath[i, j], diff_bath_kinetic[i, j]))



    # Bath + Frozen Occ + Frozen Vir 轨道的起始与结束 slice
    ref_bath_env_slice = slice(ref_nimp, ref_dmet.nes + ref_dmet.nfo + ref_dmet.nfv)
    new_bath_env_slice = slice(new_nimp, new_dmet.nes + new_dmet.nfo + new_dmet.nfv)
    
    ref_iao_bath_env = ref_iao[ref_bath_env_slice]
    new_iao_bath_env = new_iao[new_bath_env_slice]

    ref_kinetic_bath_env = ref_kinetic[ref_bath_env_slice]
    new_kinetic_bath_env = new_kinetic[new_bath_env_slice]

    n_ref_bath_env_orb = len(ref_iao_bath_env)
    n_new_bath_env_orb = len(new_iao_bath_env)

    ref_iao_bath_env_list = []
    ref_kinetic_bath_env_list = []
    
    for i, ii in enumerate(ref_iao_bath_env):
        for j, jj in enumerate(new_iao_bath_env):
            print(f"Comparing IAO population of ref bath+env orbital {i} with new bath+env orbital {j}:")
            pop_diff = np.sum(np.abs(ii - jj))
            print(f"Population difference: {pop_diff}")
            print("-" * 30)
            ref_iao_bath_env_list.append((i, j, pop_diff))
    print(" shape of iao_bath_env_list: ", np.array(ref_iao_bath_env_list).shape)
    print("\n" + "=" * 50 + "\n")
    
    for i, k_ref in enumerate(ref_kinetic_bath_env):
        for j, k_new in enumerate(new_kinetic_bath_env):
            kine_diff = np.abs(k_ref - k_new)
            print(f"Comparing kinetic energy of ref bath+env orbital {i} with new bath+env orbital {j}:")
            print(f"Kinetic energy difference: {kine_diff}")
            ref_kinetic_bath_env_list.append((i, j, kine_diff))
            
    diff_bath_env = np.zeros((n_ref_bath_env_orb, n_new_bath_env_orb))
    for i, j, pop_diff in ref_iao_bath_env_list:
        diff_bath_env[i, j] = pop_diff
        
    diff_bath_env_kinetic = np.zeros((n_ref_bath_env_orb, n_new_bath_env_orb))
    for i, j, kine_diff in ref_kinetic_bath_env_list:
        diff_bath_env_kinetic[i, j] = kine_diff
    bath_env_map_list = []
        
    for i in range(diff_bath_env.shape[0]):
        for j in range(diff_bath_env.shape[1]):
            if diff_bath_env[i, j] < iao_threshold and diff_bath_env_kinetic[i, j] < kinetic_threshold:
                print(f"Orbital {i} in ref bath+env matches with orbital {j} in new bath+env with population difference {diff_bath_env[i, j]:.4f} and kinetic energy difference {diff_bath_env_kinetic[i, j]:.4f}")
                bath_env_map_list.append((i, j , diff_bath_env[i, j], diff_bath_env_kinetic[i, j]))
    print(ref_kinetic.shape)
    print("bath env ")
    print(diff_bath_env)
    print("bath env kinetic")
    print(diff_bath_env_kinetic)
    
    
    
    # then we can compare the imp iao population to find the mapping
    ref_iao_imp_list = []
    ref_kinetic_imp_list = []
    for i, ii in enumerate(ref_imp_iao):
        for j, jj in enumerate(new_imp_iao):
            print(f"Comparing IAO population of ref imp orbital {i} with new imp orbital {j}:")
            #print("Ref IAO population:", ii)
            #print("New IAO population:", jj)
            # sum over the IAO pop difference to find the best match
            pop_diff = np.sum(np.abs(ii - jj))
            print(f"Population difference: {pop_diff}")
            print("-"*30)
            ref_iao_imp_list.append((i, j, pop_diff))
    print("\n" + "="*50 + "\n")
    #print(" shape of iao_imp_list: ", np.array(ref_iao_imp_list).shape)
    for i, ii in enumerate(ref_kinetic_imp):
        for j, jj in enumerate(new_kinetic_imp):
            kine_diff = np.abs(ii - jj)
            print(f"Comparing kinetic energy of ref orbital {i} with new orbital {j}:")
            print(f"Kinetic energy difference: {kine_diff}")
            ref_kinetic_imp_list.append((i, j, kine_diff))
    print(" shape of iao_imp_list: ", np.array(ref_iao_imp_list).shape)
    print(" shape of kinetic_imp_list: ", np.array(ref_kinetic_imp_list).shape)
    diff_imp = np.zeros((len(ref_imp_iao), len(new_imp_iao)))
    for i, j, pop_diff in ref_iao_imp_list:
        diff_imp[i, j] = pop_diff
    diff_imp_kinetic = np.zeros((len(ref_kinetic_imp), len(new_kinetic_imp)))
    for i, j, kine_diff in ref_kinetic_imp_list:
        diff_imp_kinetic[i, j] = kine_diff
    imp_map_list = []
    for i in range(diff_imp.shape[0]):
        for j in range(diff_imp.shape[1]):
            if diff_imp[i,j] < iao_threshold and diff_imp_kinetic[i,j] < kinetic_threshold:
                print(f"Orbital {i} in ref matches with orbital {j} in new with population difference {diff_imp[i,j]:.4f} and kinetic energy difference {diff_imp_kinetic[i,j]:.4f}")
                imp_map_list.append((i, j , diff_imp[i,j], diff_imp_kinetic[i,j]))
                
    def visualize_mapping(diff_matrix, matched_pairs, title, filename, val_fmt="{:.2f}", colorbar_label="Absolute Difference", ref_start_idx=0, new_start_idx=0):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        num_ref, num_new = diff_matrix.shape
        plt.figure(figsize=(max(8, num_new * 0.5), max(6, num_ref * 0.5)))
        plt.imshow(diff_matrix, cmap='viridis_r', aspect='auto')
        
        font_size = 8 if max(num_ref, num_new) > 20 else 10
        mid_val = diff_matrix.max() / 2 if diff_matrix.size > 0 else 1.0
        
        for i in range(num_ref):
            for j in range(num_new):
                plt.text(j, i, val_fmt.format(diff_matrix[i, j]), 
                         ha="center", va="center", fontsize=font_size,
                         color="w" if diff_matrix[i, j] < mid_val else "black")

        if matched_pairs is not None:
            for pair in matched_pairs:
                i, j = pair[0], pair[1]
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)

        plt.colorbar(label=colorbar_label)
        
        step_y = 1 if num_ref <= 20 else max(1, num_ref // 10)
        step_x = 1 if num_new <= 20 else max(1, num_new // 10)
        
        plt.yticks(np.arange(0, num_ref, step_y), [f"Ref {i + ref_start_idx}" for i in range(0, num_ref, step_y)])
        plt.xticks(np.arange(0, num_new, step_x), [f"New {j + new_start_idx}" for j in range(0, num_new, step_x)], rotation=45)
        
        plt.ylabel("Reference Orbital Index")
        plt.xlabel("New Orbital Index")
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[{title}] mapping heatmap saved as {filename}")

    def get_orbitals_centroids(dmet):
        C = dmet.caolo @ dmet.lo_cloes
        with dmet.mol.with_common_orig((0,0,0)):
            r_ao = dmet.mol.intor('int1e_r', comp=3)
        centroids = np.einsum('xab, ai, bi -> ix', r_ao, C, C)
        return centroids

    ref_centroids = get_orbitals_centroids(ref_dmet)
    new_centroids = get_orbitals_centroids(new_dmet)

    ref_bath_env_centroids = ref_centroids[ref_bath_env_slice]
    new_bath_env_centroids = new_centroids[new_bath_env_slice]
    
    diff_bath_env_centroid = np.zeros((n_ref_bath_env_orb, n_new_bath_env_orb))
    for i in range(n_ref_bath_env_orb):
        for j in range(n_new_bath_env_orb):
            diff_bath_env_centroid[i, j] = np.linalg.norm(ref_bath_env_centroids[i] - new_bath_env_centroids[j])

    centroid_map_list = []
    for i in range(n_ref_bath_env_orb):
        j_match = np.argmin(diff_bath_env_centroid[i, :])
        if diff_bath_env_centroid[i, j_match] < 2.0: # threshold for distance
            centroid_map_list.append((i, j_match))

    visualize_mapping(diff_imp, imp_map_list, 
                      "Impurity IAO Population Mapping", f"imp_iao_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.2f}", ref_start_idx=0, new_start_idx=0)
    visualize_mapping(diff_imp_kinetic, imp_map_list, 
                      "Impurity Kinetic Energy Mapping", f"imp_kinetic_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.3f}", ref_start_idx=0, new_start_idx=0)
                      
    # Bath/Bath+Env 
    visualize_mapping(diff_bath_env, bath_env_map_list, 
                      "Bath+Env IAO Population Mapping", f"bath_env_iao_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.2f}", ref_start_idx=ref_nimp, new_start_idx=new_nimp)
    visualize_mapping(diff_bath_env_kinetic, bath_env_map_list, 
                      "Bath+Env Kinetic Energy Mapping", f"bath_env_kinetic_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.3f}", ref_start_idx=ref_nimp, new_start_idx=new_nimp)

    visualize_mapping(diff_bath_env_centroid, centroid_map_list, 
                      "Bath+Env Centroid Distance Mapping", f"bath_env_centroid_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.2f}", colorbar_label="Distance (Bohr)", ref_start_idx=ref_nimp, new_start_idx=new_nimp)

    visualize_mapping(diff_bath, bath_map_list, "Bath IAO Population Mapping", f"bath_iao_mapping_{name1}_to_{name2}.png", 
                      val_fmt="{:.2f}", ref_start_idx=ref_nimp, new_start_idx=new_nimp)


def sweep(dmet_list, max_iter=20, method="sweep"):
    if callable(method):
        updater = method
    else:
        method_map = {
            "sweep": from_ref_get_newbath4new_geom2,
            "sweep2": from_ref_get_newbath4new_geom3,
            "consistent": from_ref_get_newbath4new_geom,
            "map": from_ref_get_newbath4new_geom4,
        }
        try:
            updater = method_map[method]
        except KeyError:
            raise ValueError("Unknown method: choose 'sweep', 'consistent', or pass a callable")
    eo_sizes = [dmet.nes for dmet in dmet_list]
    print(f"\n[Sweep Start] 初始 EO 大小列表: {eo_sizes}")
    
    for iteration in range(max_iter):
        old_eo_sizes = list(eo_sizes) 
        print(f"\n---====== 第 {iteration+1} 次循环扫描 ======---")
        
        print(">>> 正向扫描 (Forward)")
        for i in range(1, len(dmet_list)):
            print(f"--> 使用体系 {i-1} 作为参考，更新体系 {i}")
            dmet_list[i] = updater(dmet_list[i-1], dmet_list[i])
            eo_sizes[i] = dmet_list[i].nes
            
        print("<<< 反向扫描 (Backward)")
        for i in range(len(dmet_list)-1, 0, -1):
            print(f"<-- 使用体系 {i} 作为参考，更新体系 {i-1}")
            dmet_list[i-1] = updater(dmet_list[i], dmet_list[i-1])
            eo_sizes[i-1] = dmet_list[i-1].nes
            
        print(f"第 {iteration+1} 次循环完成，当前 EO 大小同步列表为: {eo_sizes}")
        
        if eo_sizes == old_eo_sizes:
            print(f"\n=> Surprise! 在第 {iteration+1} 次循环后，所有的包含 Bath 在内的 EO 大小不再变化，达到了完全一致！")
            break
    else:
        print(f"\n=> 警告：达到了最大迭代次数 {max_iter}，NOT CONVERGED。")
        
    return dmet_list, eo_sizes
