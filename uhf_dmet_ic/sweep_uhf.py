import numpy as np

from uhf_dmet_ic.consistent_bath import (
    analyze_bath_composition,
    find_bath_indices_from_reference_svd,
    find_bath_indices_from_reference_svd2,
    find_bath_indices_from_reference_svd3,
    mapping_ao,
    append_bath_by_env_idx
)

def from_ref_get_newbath4new_geom(mydmet_ref, mydmet_new):
    for spin, s in zip(["alpha", "beta"], [0, 1]):
        print(f"Analyzing {spin} spin:")
        nimp_ref = len(mydmet_ref.imp_idx)
        nbath_ref = mydmet_ref.nes[s] - nimp_ref
        ref_bath_ao = mydmet_ref.es_orb[s]
        print(f"[{spin}] NUMBER of BATH ORBITALS in the old system: {nbath_ref}")
        print("="*50)
        print(f"[{spin}] Analisis of the OLD SYSTEM:")
        #analyze_bath_composition(mydmet_ref, spin=spin)
        print("\n" + "="*50 + "\n")

        idxs_to_recover = find_bath_indices_from_reference_svd(mydmet_new, ref_bath_ao, ref_mol=mydmet_ref.mol, spin=spin)
        e1 = mydmet_new.es_e + mydmet_new.fo_ene
        print(f"*** CHECK Mean Field Energy (initial): {e1}")
        
        print(f"[{spin}] Restoring {len(idxs_to_recover)} bath orbitals that were lost to maintain consistency...")
        mydmet_new = append_bath_by_env_idx(mydmet_new, idxs_to_recover, spin=spin)
        e2 = mydmet_new.es_e + mydmet_new.fo_ene
        print(f"*** CHECK Mean Field Energy: {e2}", f"Deviation from the not appended system: {e2 - e1}")
        print(f"========= DMET APPEND for {spin} spin =========")
        print(f"[{spin}] NUMBER of BATH ORBITALS", mydmet_new.nes[s] - len(mydmet_new.imp_idx))
        print(f"[{spin}] NUMBER of EO space orbitals after appending:", mydmet_new.nes[s])
        print(f"[{spin}] Indices of the new bath orbitals added to maintain consistency:", idxs_to_recover)
        print("="*65)
        print(f"[{spin}] Analisis of the NEW BATH SYSTEM after restoration:")
        #analyze_bath_composition(mydmet_new, spin=spin)
    return mydmet_new

    '''
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
    return mydmet_new'''
def sweep(dmet_list, max_iter=20, method="sweep"):
    if callable(method):
        updater = method
    else:
        method_map = {
            #"sweep": from_ref_get_newbath4new_geom2,
            #"sweep2": from_ref_get_newbath4new_geom3,
            "consistent": from_ref_get_newbath4new_geom,
            #"map": from_ref_get_newbath4new_geom4,
        }
        try:
            updater = method_map[method]
        except KeyError:
            raise ValueError("Unknown method: choose 'sweep', 'consistent', or pass a callable")
    eo_sizes = [dmet.nes for dmet in dmet_list]
    print(f"\n[Sweep Start] Initial EO sizes: {eo_sizes}")
    
    for iteration in range(max_iter):
        old_eo_sizes = list(eo_sizes) 
        print(f"\n---====== The {iteration+1} iteration ======---")
        
        print(">>> Forward scanning")
        for i in range(1, len(dmet_list)):
            print(f"--> Using system {i-1} as reference, updating system {i}")
            dmet_list[i] = updater(dmet_list[i-1], dmet_list[i])
            eo_sizes[i] = dmet_list[i].nes
            
        print("<<< Backward scanning")
        for i in range(len(dmet_list)-1, 0, -1):
            print(f"<-- Using system {i} as reference, updating system {i-1}")
            dmet_list[i-1] = updater(dmet_list[i], dmet_list[i-1])
            eo_sizes[i-1] = dmet_list[i-1].nes
            
        print(f"The {iteration+1} iteration completed, current EO sizes are: {eo_sizes}")
        
        if eo_sizes == old_eo_sizes:
            print(f"\n=> Surprise! In the {iteration+1} iteration, all EO sizes within the Bath are no longer changing, reaching complete consistency!")
            break
    else:
        print(f"\n=> Warning: Reached maximum iteration number {max_iter}, NOT CONVERGED.")
        
    return dmet_list, eo_sizes
