import numpy as np
from pyscf import lib, lo
from functools import reduce

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