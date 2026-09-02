from embed_sim import uhf_tool, ssdmet
import numpy as np
from embed_sim import concen_loc_uhf

def sie_ccsd_t(mol, imp_idx, title, full_cc=False, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.uhf.UHF(mol).x2c().density_fit()
    else:
        mf = scf.uhf.UHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 0.0
    mf.conv_tol = 1e-6
    mf.max_memory = 100000
    mf.chkfile = 'cluster{0}_uhf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mymp2 = mp.MP2(mf).run()
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf).run()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3 = lib.logger.perf_counter()
    mydmet = uhf_tool.SSDMET_uhf(mf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    t4 = lib.logger.perf_counter()
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene

def sie_rohf_uccsd_t(mol, imp_idx, title, full_cc=False, x2c=True, frozen=False):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-8
    mf.max_memory = 100000
    mf.max_cycle = 250
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        if frozen:
            mycc.set_frozen()
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)




    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene

def sie_rohf_uccsd_t_cl(mol, imp_idx, title, full_cc=False, shell=None, atoms_A=None, proj_bas=None, x2c=True, loc = False):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-6
    mf.max_memory = 100000
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    if loc :
        mydmet = concen_loc_uhf.localize_spaces(mydmet, spin='alpha')
        mydmet = concen_loc_uhf.localize_spaces(mydmet, spin='beta')

    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy1: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)
    print(f"Mean field energy2: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy3: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)

    print(f"Mean field energy4: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))



    
    t3 = lib.logger.perf_counter()
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t3 - t2:.2f} seconds")
    print(f"Cluster {title} DMET-MP2 time: {t4 - t3:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t5 - t4:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    #print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t3 - t2:.2f} seconds")
    print(f"DMET-MP2 time: {t4 - t3:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t5 - t4:.2f} seconds")
    print("=============================================")
        ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t6 - t5:.2f} seconds")

    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene

def sie_rohf_uccsd_t_mp2(mol, imp_idx, title, full_cc=False, eta=1e-2, x2c=True, loc=False):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-8
    mf.max_memory =getattr(mol, 'max_memory', 100000)
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    if loc :
        mydmet = concen_loc_uhf.localize_spaces(mydmet, spin='alpha')
        mydmet = concen_loc_uhf.localize_spaces(mydmet, spin='beta')

    print(f"Mean field energy before mp2 bath expansion: {mydmet.es_e + mydmet.fo_ene}")
    mydmet = concen_loc_uhf.get_UMP2_bath(mydmet, eta=eta)
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")

    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene


def sie_rccsd_t(mol, imp_idx, title, full_cc=False, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rhf.RHF(mol).x2c().density_fit()
    else:
        mf = scf.rhf.RHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)

    mf.init_guess = 'atom'
    mf.level_shift = 0.0
    mf.conv_tol = 1e-6
    mf.max_memory = 100000
    mf.chkfile = 'cluster{0}_rhf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mymp2 = mp.MP2(mf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf).run()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3  = lib.logger.perf_counter()
    mydmet = ssdmet.SSDMET(mf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12, es_natorb = False).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    t4= lib.logger.perf_counter()
    dmet_mp_etot, dmet_mp_ecorr = mydmet.mp2_solver()

    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_etot}")
    t5 = lib.logger.perf_counter()
    dmet_cc, dmet_cc_etot, dmet_cc_ecorr = mydmet.ccsdt_solver()
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_etot}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp_ecorr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc_ecorr}")
    print(f"Mean field energy: {mydmet.mf_or_cas.e_tot}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc_ecorr + mydmet.mf_or_cas.e_tot - dmet_mp_ecorr:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_etot, dmet_cc_etot, mymp2.e_corr + dmet_cc_ecorr + mydmet.mf_or_cas.e_tot - dmet_mp_ecorr



def sie_rohf_uccsd_t_cl_check(mol, imp_idx, title, full_cc=False, shell=None, atoms_A=None, proj_bas=None, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    from embed_sim import concen_loc_uhf
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-6
    mf.max_memory = 100000
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")

    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy1: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)
    print(f"Mean field energy2: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy3: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)

    print(f"Mean field energy4: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))


    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")







def sie_uhf_uccsd_t(mol, imp_idx, title, full_cc=False, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 1.0
    mf.conv_tol = 1e-8
    mf.max_memory = 150000
    mf.max_cycle = 200
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 0.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mf_uhf.kernel()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)




    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene

def sie_uhf_uccsd_t_cl(mol, imp_idx, title, full_cc=False, shell=None, atoms_A=None, proj_bas=None, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-6
    mf.max_memory = 150000
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mf_uhf.kernel()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")

    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy1: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_occ_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)
    print(f"Mean field energy2: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='alpha', ele_density=True)
    print(f"Mean field energy3: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))
    mydmet = concen_loc_uhf.concentric_vir_localization(mydmet, proj_bas = proj_bas, n_shell=shell, atoms_A=atoms_A, couple_op='fock', spin='beta', ele_density=True)

    print(f"Mean field energy4: {mydmet.es_e + mydmet.fo_ene}")
    print(f"es_e: {mydmet.es_e}, fo_ene: {mydmet.fo_ene}")
    print(f"es_dm trace:", np.trace(mydmet.es_dm[0]), np.trace(mydmet.es_dm[1]))



    
    t3 = lib.logger.perf_counter()
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t3 - t2:.2f} seconds")
    print(f"Cluster {title} DMET-MP2 time: {t4 - t3:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t5 - t4:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    #print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t3 - t2:.2f} seconds")
    print(f"DMET-MP2 time: {t4 - t3:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t5 - t4:.2f} seconds")
    print("=============================================")
        ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t6 - t5:.2f} seconds")

    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene

def sie_uhf_uccsd_t_mp2(mol, imp_idx, title, full_cc=False, eta=1e-2, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-8
    mf.max_memory =getattr(mol, 'max_memory', 150000)
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mf_uhf.kernel()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD(T) energy: {mycc.e_tot + et}")
        print(f"Cluster {title} CCSD(T) correlation energy: {mycc.e_corr + et}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD(T) time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, threshold = 1e-12).density_fit()
    mydmet.build(restore_imp=True, save_chk= False)
    print(f"Mean field energy before mp2 bath expansion: {mydmet.es_e + mydmet.fo_ene}")
    mydmet = concen_loc_uhf.get_UMP2_bath(mydmet, eta=eta)
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")

    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + dmet_et + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD(T) energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD(T) correlation energy: {dmet_cc.e_corr + dmet_et}")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"SIE-CCSD(T) energy: {mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD(T) time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr + dmet_et - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene



#################### UHF-CCSD no pertubation as tests ####################
def sie_uhf_uccsd(mol, imp_idx, title, full_cc=False, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-8
    mf.max_memory = 150000
    mf.max_cycle = 250
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mf_uhf.kernel()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        #et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD energy: {mycc.e_tot}")
        print(f"Cluster {title} CCSD correlation energy: {mycc.e_corr}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, es_method = 'eig', threshold = 1e-12).density_fit()
    mydmet.build(restore_imp = True, save_chk = False)




    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    #dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"Full MP2 total energy: {mymp2.e_tot}")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET-MP2 total energy: {dmet_mp_ene}")
    print(f"DMET-CCSD total energy: {dmet_cc_ene}")
    print("*****************************")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD correlation energy: {dmet_cc.e_corr }")
    print(f"SIE-CCSD energy: {mymp2.e_corr + dmet_cc.e_corr - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr  - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene




def sie_uhf_uccsd_ao(mol, imp_idx, title, full_cc=False, x2c=True):
    from pyscf import cc, mp, lib, scf, df
    if x2c == True:
        mf = scf.rohf.ROHF(mol).x2c().density_fit()
    else:
        mf = scf.rohf.ROHF(mol).density_fit()
    mf.with_df.auxbasis = df.make_auxbasis(mol)
    mf.init_guess = 'atom'
    mf.level_shift = 2.0
    mf.conv_tol = 1e-8
    mf.max_memory = 150000
    mf.max_cycle = 250
    mf.chkfile = 'cluster{0}_rohf.chk'.format(title)
    t0 = lib.logger.perf_counter()
    mf.kernel()
    dm = mf.make_rdm1()

    mf.level_shift = 1.0
    mf.kernel(dm)
    dm = mf.make_rdm1()
    mf.level_shift = 0.0

    mf.kernel(dm)
    t1 = lib.logger.perf_counter()
    print(f"Cluster {title} HF time: {t1 - t0:.2f} seconds")
    ### MP2
    mf_uhf = mf.to_uhf()
    mf_uhf.kernel()
    mymp2 = mp.MP2(mf_uhf).run()
    print(f"Cluster {title} MP2 energy: {mymp2.e_tot}")
    t2 = lib.logger.perf_counter()
    print(f"Cluster {title} MP2 time: {t2 - t1:.2f} seconds")
    ### CCSD(T)
    if full_cc:
        mycc = cc.CCSD(mf_uhf)
        mycc.verbose = 4
        mycc.conv_tol_normt = 1e-5
        mycc.kernel()
        #et = mycc.ccsd_t()
        print(f"Cluster {title} CCSD energy: {mycc.e_tot}")
        print(f"Cluster {title} CCSD correlation energy: {mycc.e_corr}")
    t3 = lib.logger.perf_counter()
    print(f"Cluster {title} CCSD time: {t3 - t2:.2f} seconds")
    mydmet = uhf_tool.SSDMET_uhf(mf_uhf, title = f'DMET_{title}',  imp_idx = imp_idx, es_method = 'eig', threshold = 1e-12).density_fit()
    mydmet.build(restore_imp = False, aodmet = True, save_chk = False)




    t4 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET build time: {t4 - t3:.2f} seconds")
    dmet_mp = mydmet.ump2()
    dmet_mp.verbose = 4
    dmet_mp.kernel(with_t2 = False)
    t5 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-MP2 time: {t5 - t4:.2f} seconds")
    dmet_mp_ene = dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
    print(f"Cluster {title} DMET-MP2 energy: {dmet_mp_ene}")
    dmet_cc = mydmet.uccsd()
    dmet_cc.verbose = 4
    dmet_cc.kernel()
    #dmet_et = dmet_cc.ccsd_t()
    dmet_cc_ene = dmet_cc.e_corr + mydmet.es_e + mydmet.fo_ene
    t6 = lib.logger.perf_counter()
    print(f"Cluster {title} DMET-CCSD time: {t6 - t5:.2f} seconds")
    print(f"Cluster {title} DMET-CCSD energy: {dmet_cc_ene}")
    print("================Cluster Results=================")
    print(f"Mean field energy: {mydmet.es_e + mydmet.fo_ene}")
    print(f"Full MP2 total energy: {mymp2.e_tot}")
    print(f"Full MP2 correlation energy: {mymp2.e_corr}")
    print(f"DMET-MP2 total energy: {dmet_mp_ene}")
    print(f"DMET-CCSD total energy: {dmet_cc_ene}")
    print("*****************************")
    print(f"DMET MP2 correlation energy: {dmet_mp.e_corr}")
    print(f"DMET CCSD correlation energy: {dmet_cc.e_corr }")
    print(f"SIE-CCSD energy: {mymp2.e_corr + dmet_cc.e_corr - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene:.8f} Hartree")
    print("================Time Summary===============")
    print(f"HF time: {t1 - t0:.2f} seconds")
    print(f"MP2 time: {t2 - t1:.2f} seconds")
    print(f"CCSD(T) time: {t3 - t2:.2f} seconds")
    print(f"DMET build time: {t4 - t3:.2f} seconds")
    print(f"DMET-MP2 time: {t5 - t4:.2f} seconds")
    print(f"DMET-CCSD time: {t6 - t5:.2f} seconds")
    print("=============================================")
    return mymp2.e_tot, dmet_mp_ene, dmet_cc_ene, mymp2.e_corr + dmet_cc.e_corr  - dmet_mp.e_corr + mydmet.es_e + mydmet.fo_ene
