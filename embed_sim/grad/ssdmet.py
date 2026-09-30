from pyscf import ao2mo, lib
from pyscf.lib import logger
from functools import reduce
import numpy as np
import scipy
from embed_sim import ssdmet
from embed_sim.grad import solvers

from functools import partial
from numpy import einsum
einsum = partial(einsum, optimize=True)

def get_es_imds(es_mf, solver_option, log):
    solver = solver_option['solver']
    
    if solver == 'ROHF':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.ROHF(es_mf, log)
    elif solver == 'SA-CASSCF':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.SA_CASSCF(es_mf, solver_option, log)
    elif solver == 'SA-HCISCF':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.SA_HCISCF(es_mf, solver_option, log)
    elif solver == 'alpha-CASSCF':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.alpha_CASSCF(es_mf, solver_option, log)
    elif solver == 'CASSCF':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.CASSCF(es_mf, solver_option, log)
    elif solver =='MP2':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.MP2(es_mf, log)
    elif solver == 'CCSD':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.CCSD(es_mf, solver_option, log)
    elif solver == 'DSRG-MRPT2':
        es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = solvers.DSRG_MRPT2(es_mf, solver_option, log)
    else:
        raise NotImplementedError
    
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def mol_slice(atm_id, mol):
    _, _, p0, p1 = mol.aoslice_by_atom()[atm_id]
    return slice(p0, p1)

def gen_get_indep_pair_slice(mf):
    mo_occ = mf.mo_occ
    ndo = (mo_occ==2).sum()
    nho = (mo_occ==1).sum()
    nvo = (mo_occ==0).sum()
    sd = slice(0,ndo)
    sh = slice(ndo,ndo+nho)
    sv = slice(ndo+nho,ndo+nho+nvo)
    n_indep_pair = nho*ndo + nvo*ndo + nvo*nho
    def get_indep_pair_slice(A, axis=-1):
        if axis == -1:
            A_indep_pair = np.zeros([x for x in A.shape[:-2]]+[n_indep_pair])
            A_indep_pair[...,:nho*ndo] = A[...,sh,sd].reshape(*A.shape[:-2],-1)
            A_indep_pair[...,nho*ndo:nho*ndo+nvo*ndo] = A[...,sv,sd].reshape(*A.shape[:-2],-1)
            A_indep_pair[...,nho*ndo+nvo*ndo:] = A[...,sv,sh].reshape(*A.shape[:-2],-1)
        elif axis == 0:
            A_indep_pair = np.zeros([n_indep_pair]+[x for x in A.shape[2:]])
            A_indep_pair[:nho*ndo] = A[sh,sd].reshape(-1,*A.shape[2:])
            A_indep_pair[nho*ndo:nho*ndo+nvo*ndo] = A[sv,sd].reshape(-1,*A.shape[2:])
            A_indep_pair[nho*ndo+nvo*ndo:] = A[sv,sh].reshape(-1,*A.shape[2:])
        else:
            raise IndexError
        return A_indep_pair
    return get_indep_pair_slice

def get_S_1(mf, atmlst, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mol = mf.mol
    nao = mol.nao
    mo_coeff = mf.mo_coeff
    natm = mol.natm
    int1e_ipovlp = mol.intor('int1e_ipovlp')
    S_1_ao = np.zeros((len(atmlst), 3, nao, nao))
    for k, ia in enumerate(atmlst):
        sA = mol_slice(ia, mol)
        S_1_ao[k, :, sA, :] = - int1e_ipovlp[:, sA, :]
    S_1_ao += S_1_ao.swapaxes(-1, -2)
    S_1_mo = einsum('Atuv,up,vq->Atpq', S_1_ao, mo_coeff, mo_coeff)
    
    t0 = log.timer('S_1', *t0)
    return S_1_ao, S_1_mo

def get_H_1(mf, atmlst, log):
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    mol = mf.mol
    nao = mol.nao
    mo_coeff = mf.mo_coeff
    natm = mol.natm
    hcore_deriv = mf.Gradients().hcore_generator(mol)
    H_1_ao = np.zeros((len(atmlst), 3, nao, nao))
    for k, ia in enumerate(atmlst):
        H_1_ao[k] = hcore_deriv(ia)
    H_1_mo = einsum('Atuv,up,vq->Atpq', H_1_ao, mo_coeff, mo_coeff)
    
    t0 = log.timer('H_1', *t0)
    return H_1_ao, H_1_mo

def group_indexes_by_diff(arr, threshold=1e-15):
    if len(arr) == 0:
        return []
    
    result = [[]]
    current_group = [0]

    for i in range(1, len(arr)):
        diff = abs(arr[i] - arr[i-1])
        
        if diff > threshold:
            if len(current_group) == 1:
                result[0].extend(current_group)
            else:
                result.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if len(current_group) == 1:
        result[0].extend(current_group)
    else:
        result.append(current_group)

    return result[0], result[1:]

# def get_lowdin_1(mf, S_1_ao, log):
#     from pyscf.lo.orth import lowdin
    
#     t0 = (logger.process_clock(), logger.perf_counter())
    
#     nao = mf.mol.nao
#     S = mf.get_ovlp()
#     sval,svec = np.linalg.eigh(S)
#     Ws = sval[:,None] - sval[None,:]
#     Ws[range(nao), range(nao)] = np.inf
#     if (abs(Ws) < 1e-12).sum() > 0:
#         if (abs(Ws) < 1e-15).sum() > 0:
#             log.warn(f'''
#     {(abs(Ws) < 1e-15).sum()} degenerate eigenvalues of the overlap matrix are found,
#     the gradients may be unreliable!
#     ''')
#         else:
#             log.warn(f'''
#     {(abs(Ws) < 1e-12).sum()} nearly degenerate eigenvalues of the overlap matrix are found,
#     the gradients may suffer from numerical instability.
#     ''')
#     Ws[abs(Ws) < 1e-15] = np.inf
#     Ws = 1 / Ws
#     S_1_svec = np.einsum('Atab,aj,bm->Atjm',S_1_ao,svec,svec)
#     nondegen_sval, degen_sval_list = group_indexes_by_diff(sval, threshold=1e-15)
#     svec_nondegen = svec[:,nondegen_sval]
#     Ws_nondegen = Ws[nondegen_sval]
#     sval_nondegen = sval[nondegen_sval]
#     S_1_svec_nondegen = S_1_svec[:,:,:,nondegen_sval]
#     caolo_1 = (
#         np.einsum('mj,pj,rm,Atjm->Atpr',Ws_nondegen,svec,svec_nondegen*(sval_nondegen**(-1/2)),S_1_svec_nondegen)
#     )
#     caolo_1 += caolo_1.swapaxes(-1,-2)
#     caolo_1 -= 0.5*np.einsum('pm,rm,Atmm->Atpr',
#                              svec_nondegen,svec_nondegen*(sval_nondegen**(-3/2)),S_1_svec_nondegen[:,:,nondegen_sval])
#     cloao_1 = (
#         np.einsum('mj,pj,rm,Atjm->Atpr',Ws_nondegen,svec,svec_nondegen*(sval_nondegen**(1/2)),S_1_svec_nondegen)
#     )
#     cloao_1 += cloao_1.swapaxes(-1,-2)
#     cloao_1 += 0.5*np.einsum('pm,rm,Atmm->Atpr',
#                              svec_nondegen,svec_nondegen*(sval_nondegen**(-1/2)),S_1_svec_nondegen[:,:,nondegen_sval])
#     for degen_sval in degen_sval_list:
#         degen_sval_comp = [x for x in range(nao) if x not in degen_sval]
#         svec_degen = svec[:,degen_sval]
#         svec_degen_comp = svec[:,degen_sval_comp]
#         Ws_degen = Ws[degen_sval,:][:,degen_sval_comp]
#         sval_degen = sval[degen_sval]
#         S_1_svec_degen_comp = np.einsum('Atab,aj,bm->Atjm',S_1_ao,svec_degen_comp,svec_degen)
#         S_1_svec_degen = np.einsum('Atab,aj,bm->Atjm',S_1_ao,svec_degen,svec_degen)
#         caolo_1_degen = np.einsum('mj,pj,rm,Atjm->Atpr',
#                                   Ws_degen,svec_degen_comp,svec_degen*(sval_degen**(-1/2)),S_1_svec_degen_comp)
#         caolo_1_degen += caolo_1_degen.swapaxes(-1,-2)
#         caolo_1_degen -= 0.5*np.einsum('pm,rm,Atmm->Atpr',
#                                        svec_degen,svec_degen*(sval_degen**(-3/2)),S_1_svec_degen)
#         cloao_1_degen = np.einsum('mj,pj,rm,Atjm->Atpr',
#                                   Ws_degen,svec_degen_comp,svec_degen*(sval_degen**(1/2)),S_1_svec_degen_comp)
#         cloao_1_degen += cloao_1_degen.swapaxes(-1,-2)
#         cloao_1_degen += 0.5*np.einsum('pm,rm,Atmm->Atpr',
#                                        svec_degen,svec_degen*(sval_degen**(-1/2)),S_1_svec_degen)
#         caolo_1 += caolo_1_degen
#         cloao_1 += cloao_1_degen
    
#     t0 = log.timer('lowdin_1', *t0)
#     return caolo_1, cloao_1

# def get_lowdin_1(mf, S_1_ao, log):
#     t0 = (logger.process_clock(), logger.perf_counter())
    
#     nao = mf.mol.nao
#     S = mf.get_ovlp()
#     sval,svec = np.linalg.eigh(S)
#     Ws = sval[:,None] - sval[None,:]
#     Ws[range(nao), range(nao)] = np.inf
#     if (abs(Ws) < 1e-12).sum() > 0:
#         if (abs(Ws) < 1e-15).sum() > 0:
#             log.warn(f'''
#     {(abs(Ws) < 1e-15).sum()} degenerate eigenvalues of the overlap matrix are found,
#     the gradients are unreliable! Please fine-tune your initial structure!
#     ''')
#         else:
#             log.warn(f'''
#     {(abs(Ws) < 1e-12).sum()} nearly degenerate eigenvalues of the overlap matrix are found,
#     the gradients may suffer from numerical instability.
#     ''')
#     Ws = 1 / Ws
#     S_1_svec = einsum('Atab,aj,bm->Atjm', S_1_ao, svec, svec)
#     caolo_1_imds = einsum('mj,pj,rm,Atjm->Atpr',Ws,svec,svec*(sval**(-1/2)),S_1_svec)
#     cloao_1_imds = einsum('mj,pj,rm,Atjm->Atpr',Ws,svec,svec*(sval**(1/2)),S_1_svec)
#     caolo_1 = (
#         + caolo_1_imds
#         + caolo_1_imds.swapaxes(-1,-2)
#         - 0.5*einsum('pm,rm,Atmm->Atpr',svec,svec*(sval**(-3/2)),S_1_svec)
#     )
#     cloao_1 = (
#         + cloao_1_imds
#         + cloao_1_imds.swapaxes(-1,-2)
#         + 0.5*einsum('pm,rm,Atmm->Atpr',svec,svec*(sval**(-1/2)),S_1_svec)
#     )
    
#     t0 = log.timer('lowdin_1', *t0)
#     return caolo_1, cloao_1

def get_lowdin_1(mf, S_1_ao, log):
    from pyscf.lo.orth import lowdin
    t0 = (logger.process_clock(), logger.perf_counter())
    
    S = mf.get_ovlp()
    caolo = lowdin(S)
    cloao = lib.dot(caolo, S)
    Z, Y = np.linalg.eigh(cloao)
    YBY = einsum('iu,Atij,jv,uv->Atuv', Y, S_1_ao, Y, 1/lib.direct_sum('u+v->uv', Z, Z))
    cloao_1 = einsum('iu,Atuv,jv->Atij', Y, YBY, Y)
    caolo_1 = -1 * einsum('ui,Atij,vj->Atuv', caolo, cloao_1, caolo)
    
    t0 = log.timer('lowdin_1', *t0)
    return caolo_1, cloao_1

def get_grad_dm1(H_1_ao, dm_core, es_relaxed_rdm1_global, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    grad_dm1 = einsum('Atuv,uv->At', H_1_ao, dm_core+es_relaxed_rdm1_global)
    
    t0 = log.timer('grad_dm1', *t0)
    return grad_dm1

def get_grad_veff(mf, atmlst, dm_core, es_relaxed_rdm1_global, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mol = mf.mol
    aoslices = mol.aoslice_by_atom()
    grad_veff = np.zeros((len(atmlst),3))
    dm_eff = dm_core*0.5 + es_relaxed_rdm1_global
    vj,vk = mf.Gradients().get_jk(mol, [dm_core, dm_eff])
    vhf = vj - 0.5*vk
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        grad_veff[k] += np.einsum('xij,ij->x', vhf[0][:,p0:p1], dm_eff[p0:p1])*2
        grad_veff[k] += np.einsum('xij,ij->x', vhf[1][:,p0:p1], dm_core[p0:p1])*2
    if getattr(mf, 'with_df', None) is not None:
        grad_veff_aux = vj.aux - vk.aux * .5
        grad_veff_aux = grad_veff_aux[0,1] + grad_veff_aux[1,0]
        for k, ia in enumerate(atmlst):
            grad_veff[k] += grad_veff_aux[ia]
    
    t0 = log.timer('grad_veff', *t0)
    return grad_veff

def get_grad_dm2(mf, atmlst, ao2eo, es_relaxed_rdm2, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mol = mf.mol
    nao = mol.nao
    neo = ao2eo.shape[-1]
    aoslices = mol.aoslice_by_atom()
    grad_dm2 = np.zeros((len(atmlst),3))
    
    if getattr(mf, 'with_df', None) is None:
        from pyscf.grad.mp2 import _shell_prange
        diag_idx = np.arange(nao)
        diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
        casdm2 = es_relaxed_rdm2.copy()
        mo_cas = ao2eo
        ncas = neo
        nao_pair = nao * (nao+1) // 2
        casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
        dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                    (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
        dm2buf = lib.pack_tril(dm2buf)
        dm2buf[:,diag_idx] *= .5
        dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
        casdm2 = casdm2_cc = None
        
        max_memory = mf.max_memory - lib.current_memory()[0]
        blksize = int(max_memory*.9e6/8 / (4*(aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
        blksize = min(nao, max(2, blksize))
        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            q1 = 0
            for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
                q0, q1 = q1, q1 + nf
                dm2_ao = einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
                shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
                eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                 shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
                grad_dm2[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
                eri1 = None
    else:
        from pyscf import mcscf
        from pyscf.df.grad.casdm2_util import (solve_df_rdm2, grad_elec_dferi,
                                               grad_elec_auxresponse_dferi)
        casdm2 = es_relaxed_rdm2.copy()
        mo_cas = ao2eo
        mc_grad = mcscf.CASSCF(mf, mol.nao, mol.nelec)
        dfcasdm2 = solve_df_rdm2(mc_grad, mo_cas=mo_cas, ci=0, casdm2=casdm2)
        if atmlst is None:
            atmlst = range(mol.natm)
        grad_dm2 = grad_elec_dferi(mc_grad, mo_cas=mo_cas, ci=0, dfcasdm2=dfcasdm2, atmlst=atmlst,
                                   max_memory=mc_grad.max_memory)[0]
        grad_dm2_aux = grad_elec_auxresponse_dferi(mc_grad, mo_cas=mo_cas, ci=0, dfcasdm2=dfcasdm2,
                                                   atmlst=atmlst, max_memory=mc_grad.max_memory)[0]
        for k, ia in enumerate(atmlst):
            grad_dm2[k] += grad_dm2_aux[ia]
        dfcasdm2 = casdm2 = None
    
    t0 = log.timer('grad_dm2', *t0)
    return grad_dm2

def get_I(mf, dm_core, ao2eo, ao2core, es_relaxed_rdm1_global, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mol = mf.mol
    nao = mol.nao
    neo = ao2eo.shape[-1]
    hcore = mf.get_hcore()
    S = mf.get_ovlp()
    vj, vk = mf.get_jk(dm=dm_core)
    heff = hcore + vj - 0.5*vk
    veff_relaxed = 0.5*mf.get_veff(dm=es_relaxed_rdm1_global).sum(axis=0)
    I1 = 4*lib.dot(heff+veff_relaxed, ao2core)
    
    diag_idx = np.arange(neo)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    Gamma_buf = lib.pack_tril(es_relaxed_rdm2.reshape(neo**2,neo,neo)).reshape(neo,neo,-1)*2
    Gamma_buf[:,:,diag_idx] *= .5
    if getattr(mf, 'with_df', None) is None:
        eri_buf = ao2mo.kernel(mol,(np.eye(nao),ao2eo,ao2eo,ao2eo)).reshape(nao,neo,neo*(neo+1)//2)
    else:
        eri_buf = mf.with_df.ao2mo((np.eye(nao),ao2eo,ao2eo,ao2eo)).reshape(nao,neo,neo*(neo+1)//2)
    I2 = 2*(
        + einsum('uv,vq,pq->up', heff, ao2eo, es_relaxed_rdm1)
        + einsum('iqr,pqr->ip', eri_buf, Gamma_buf)
        - einsum('uv,vq,pq->up', S, ao2eo, es_relaxed_X)
    )
    
    t0 = log.timer('I', *t0)
    return I1, I2

def get_grad_X(S_1_ao, ao2eo, es_relaxed_X, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    grad_X = -einsum('Atuv,up,vq,pq->At',S_1_ao,ao2eo,ao2eo,es_relaxed_X)
    
    t0 = log.timer('grad_X', *t0)
    return grad_X

def get_grad_I(mydmet_grad, I1, I2, caolo_1, cloao_1, S_1_mo, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mydmet = mydmet_grad.base
    mf = mydmet.mf_or_cas
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mol = mf.mol
    nao = mol.nao
    nocc = mol.nelec[0]
    ndo = mol.nelec[1]
    nho = mol.spin
    nmo = nao
    nvir = nmo - nocc
    nvo = nvir
    n_indep_pair = nho*ndo + nvo*ndo + nvo*nho
    sd, sh, so, sv, sa = slice(0, ndo), slice(ndo, nocc), slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
    mo_coeff_occ = mo_coeff[:, mo_occ>0]
    
    ldm, caolo, cloao = mydmet.lowdin_orth()
    imp_idx = mydmet.imp_idx
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]
    ldm_env = ldm[env_idx,:][:,env_idx]
    occ_env, orb_env = np.linalg.eigh(ldm_env)
    thres = mydmet.threshold
    nimp = len(imp_idx)
    nvir = (occ_env <  thres).sum()
    nbath = ((occ_env >= thres) & (occ_env <= 2-thres)).sum()
    ncore = (occ_env > 2-thres).sum()
    lo2eo = scipy.linalg.block_diag(np.eye(nimp),orb_env[:,nvir:nvir+nbath])
    lo2core = np.vstack((np.zeros((nimp,ncore)),orb_env[:,nvir+nbath:]))
    
    mf = mydmet.mf_or_cas
    dma, dmb = mf.make_rdm1()
    dm = dma + dmb
    cloao_env_dm = einsum('nl,sl->ns', cloao[env_idx], dm)
    core_vec_comp = orb_env[:,:nvir+nbath]
    core_vec = orb_env[:,nvir+nbath:]
    bath_vec = orb_env[:,nvir:nvir+nbath]
    W_bath = occ_env[nvir:nvir+nbath][:,None] - occ_env[None,:]
    W_bath[range(nbath),range(nvir,nvir+nbath)] = 1
    nbath_degenerate = 0
    bath_degenerate_groups = group_indexes_by_diff(occ_env[nvir:nvir+nbath], threshold=1e-8)[1]
    for group in bath_degenerate_groups:
        nbath_degenerate += len(group)
    if mydmet_grad._perturb_anyway:
        W_bath = W_bath/(W_bath**2 + mydmet_grad._perturbation)
        W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
        W_core = (2-occ_env[:nvir+nbath])
        W_core = W_core/(W_core**2 + mydmet_grad._perturbation)
    else:
        if nbath_degenerate > 0:
            log.debug1(f'''
    {nbath_degenerate} near degenerate bath orbitals are found, the gradients may be unreliable.
    {mydmet_grad._perturbation} perturbation is added to the eigenvalues.
    ''')
            W_bath = W_bath/(W_bath**2 + mydmet_grad._perturbation)
            W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
            W_core = (2-occ_env[:nvir+nbath])
            W_core = W_core/(W_core**2 + mydmet_grad._perturbation)
        else:
            W_bath = 1/W_bath
            W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
            W_core = 1/(2-occ_env[:nvir+nbath])
    grad_I1 = (
        + einsum('uj,uv,k,mk,nj,vk,ns,Atms->At',
                 I1,caolo[:,nimp:],W_core,core_vec_comp,core_vec,core_vec_comp,cloao_env_dm,cloao_1[:,:,env_idx])
        + einsum('uj,uv,k,nk,mj,vk,ns,Atms->At',
                 I1,caolo[:,nimp:],W_core,core_vec_comp,core_vec,core_vec_comp,cloao_env_dm,cloao_1[:,:,env_idx])
        + einsum('uj,Atuv,vj->At',I1,caolo_1,lo2core)
    )
    grad_I2 = (
        + einsum('uj,uv,jk,mk,nj,vk,ns,Atms->At',
                 I2[:,nimp:],caolo[:,nimp:],W_bath,orb_env,bath_vec,orb_env,cloao_env_dm,cloao_1[:,:,env_idx])
        + einsum('uj,uv,jk,nk,mj,vk,ns,Atms->At',
                 I2[:,nimp:],caolo[:,nimp:],W_bath,orb_env,bath_vec,orb_env,cloao_env_dm,cloao_1[:,:,env_idx])
        + einsum('up,Atuv,vp->At',I2,caolo_1,lo2eo)
    )
    grad_I = grad_I1 + grad_I2
    
    caolo_env = caolo[:,nimp:]
    cloao_env = cloao[env_idx]
    Y = (
        + einsum('uj,k,mk,nj,vk,uv,ms,nl,si,lq->qi',
                  I1,W_core,core_vec_comp,core_vec,core_vec_comp,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,k,nk,mj,vk,uv,ms,nl,si,lq->qi',
                  I1,W_core,core_vec_comp,core_vec,core_vec_comp,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,jk,mk,nj,vk,uv,ms,nl,si,lq->qi',
                  I2[:,nimp:],W_bath,orb_env,bath_vec,orb_env,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,jk,nk,mj,vk,uv,ms,nl,si,lq->qi',
                  I2[:,nimp:],W_bath,orb_env,bath_vec,orb_env,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
    )
    Y_tilde_hd = (2*Y[sh,sd] - Y[sd,sh].T).flatten()
    Y_tilde_vd = 2*Y[sv,sd].flatten()
    Y_tilde_vh = Y[sv,sh].flatten()
    W = np.concatenate((Y_tilde_hd,Y_tilde_vd,Y_tilde_vh))
    grad_I += (
        - einsum('ij,Atij->At',Y[sd,sd],S_1_mo[:,:,sd,sd])
        - 0.5*einsum('uv,Atuv->At',Y[sh,sh],S_1_mo[:,:,sh,sh])
        - einsum('iu,Atui->At',Y[sd,sh],S_1_mo[:,:,sh,sd])
    )
    
    t0 = log.timer('grad_I', *t0)
    return grad_I, W

def gen_contract_Z_A(mf):
    mo_coeff = mf.mo_coeff
    mol = mf.mol
    nao = mol.nao
    nocc = mol.nelec[0]
    ndo = mol.nelec[1]
    nho = mol.spin
    nmo = nao
    nvir = nmo - nocc
    nvo = nvir
    n_indep_pair = nho*ndo + nvo*ndo + nvo*nho
    sd, sh, so, sv, sa = slice(0, ndo), slice(ndo, nocc), slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
    
    f = np.zeros(nao)
    f[sd] = 1; f[sh] = 0.5
    g = np.zeros(nao)
    g[sh] = 0.5
    kappa = np.zeros((nao,nao))
    kappa[sd,sh] = -1
    kappa[sh,sd] = -1
    kappa[sh,sv] = 1
    kappa[sv,sh] = 1
    delta = np.eye(nao)
    
    k_temp1 = einsum('ul,lm->um',mf.get_k(dm=einsum('vs,os->vo',mo_coeff[:,sh],mo_coeff[:,sh])),mo_coeff)
    def contract_Z_A(Z):
        Z_hd = Z[:nho*ndo].reshape(nho,ndo)
        Z_vd = Z[nho*ndo:nho*ndo+nvo*ndo].reshape(nvo,ndo)
        Z_vh = Z[nho*ndo+nvo*ndo:].reshape(nvo,nho)
        
        ZA_temp = (
            + lib.einsum('hd,ud,hd,hr->ur',Z_hd,mo_coeff[:,sd],kappa[sh,sd],delta[sh])
            + lib.einsum('hd,ud,hd,hr->ur',Z_vd,mo_coeff[:,sd],kappa[sv,sd],delta[sv])
            + lib.einsum('hd,ud,hd,hr->ur',Z_vh,mo_coeff[:,sh],kappa[sv,sh],delta[sv])
            + lib.einsum('hd,uh,hd,dr->ur',Z_hd,mo_coeff[:,sh],kappa[sh,sd],delta[sd])
            + lib.einsum('hd,uh,hd,dr->ur',Z_vd,mo_coeff[:,sv],kappa[sv,sd],delta[sd])
            + lib.einsum('hd,uh,hd,dr->ur',Z_vh,mo_coeff[:,sv],kappa[sv,sh],delta[sh])
        )
        Z_hd_ao = lib.einsum('hd,uh,vd->uv',Z_hd,mo_coeff[:,sh],mo_coeff[:,sd])
        Z_vd_ao = lib.einsum('hd,uh,vd->uv',Z_vd,mo_coeff[:,sv],mo_coeff[:,sd])
        Z_vh_ao = lib.einsum('hd,uh,vd->uv',Z_vh,mo_coeff[:,sv],mo_coeff[:,sh])
        j_temp,k_temp = mf.get_jk(dm=[Z_hd_ao,Z_vd_ao,Z_vh_ao],hermi=0)
        ZA = (
            + 0.5*lib.einsum('hd,ud,dm,hr,um->mr',Z_hd,mo_coeff[:,sd],kappa[sd],delta[sh],k_temp1)
            + 0.5*lib.einsum('hd,ud,dm,hr,um->mr',Z_vd,mo_coeff[:,sd],kappa[sd],delta[sv],k_temp1)
            + 0.5*lib.einsum('hd,ud,dm,hr,um->mr',Z_vh,mo_coeff[:,sh],kappa[sh],delta[sv],k_temp1)
            + 0.5*lib.einsum('hd,uh,hm,dr,um->mr',Z_hd,mo_coeff[:,sh],kappa[sh],delta[sd],k_temp1)
            + 0.5*lib.einsum('hd,uh,hm,dr,um->mr',Z_vd,mo_coeff[:,sv],kappa[sv],delta[sd],k_temp1)
            + 0.5*lib.einsum('hd,uh,hm,dr,um->mr',Z_vh,mo_coeff[:,sv],kappa[sv],delta[sh],k_temp1)
            - 0.5*lib.einsum('ur,um->mr',ZA_temp,k_temp1)
            + 4*lib.einsum('ol,or,lm,r->mr',j_temp.sum(axis=0),mo_coeff,mo_coeff,f)
            - lib.einsum('vo,vm,or,r->mr',k_temp.sum(axis=0),mo_coeff,mo_coeff,f)
            - lib.einsum('vo,vr,om,r->mr',k_temp.sum(axis=0),mo_coeff,mo_coeff,f)
            - lib.einsum('vo,vm,or,r->mr',-k_temp[0]+k_temp[2],mo_coeff,mo_coeff,g)
            - lib.einsum('vo,vr,om,r->mr',-k_temp[0]+k_temp[2],mo_coeff,mo_coeff,g)
        )
        return ZA
    return contract_Z_A

def get_grad_Z_vector(mf, atmlst, Z, H_1_mo, S_1_mo, contract_Z_A, get_indep_pair_slice, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mo_coeff = mf.mo_coeff
    mol = mf.mol
    nao = mol.nao
    nocc = mol.nelec[0]
    ndo = mol.nelec[1]
    nho = mol.spin
    nmo = nao
    nvir = nmo - nocc
    nvo = nvir
    n_indep_pair = nho*ndo + nvo*ndo + nvo*nho
    sd, sh, so, sv, sa = slice(0, ndo), slice(ndo, nocc), slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
    
    f = np.zeros(nao)
    f[sd] = 1; f[sh] = 0.5
    g = np.zeros(nao)
    g[sh] = 0.5
    kappa = np.zeros((nao,nao))
    kappa[sd,sh] = -1
    kappa[sh,sd] = -1
    kappa[sh,sv] = 1
    kappa[sv,sh] = 1
    delta = np.eye(nao)
    
    dma, dmb = mf.make_rdm1()
    dm = dma + dmb
    
    grad_Z = np.zeros((len(atmlst),3))
    aoslices = mol.aoslice_by_atom()
    
    grad_Z += einsum('p,Atp->At', Z, get_indep_pair_slice(H_1_mo))
    ZA = contract_Z_A(Z)
    grad_Z -= (
        + einsum('rm,Atmr->At',ZA[sd,sh],S_1_mo[:,:,sh,sd])
        + einsum('rm,Atmr->At',ZA[sd,sv],S_1_mo[:,:,sv,sd])
        + einsum('rm,Atmr->At',ZA[sh,sv],S_1_mo[:,:,sv,sh])
        + 0.5*einsum('rm,Atmr->At',ZA[sd,sd],S_1_mo[:,:,sd,sd])
        + 0.5*einsum('rm,Atmr->At',ZA[sh,sh],S_1_mo[:,:,sh,sh])
        + 0.5*einsum('rm,Atmr->At',ZA[sv,sv],S_1_mo[:,:,sv,sv])
    )
    
    Z_hd = Z[:nho*ndo].reshape(nho,ndo)
    Z_vd = Z[nho*ndo:nho*ndo+nvo*ndo].reshape(nvo,ndo)
    Z_vh = Z[nho*ndo+nvo*ndo:].reshape(nvo,nho)
    Z_ao = (
        + einsum('ij,ui,vj->uv',Z_hd,mo_coeff[:,sh],mo_coeff[:,sd])
        + einsum('ij,ui,vj->uv',Z_vd,mo_coeff[:,sv],mo_coeff[:,sd])
        + einsum('ij,ui,vj->uv',Z_vh,mo_coeff[:,sv],mo_coeff[:,sh])
    )
    Z_ao = (Z_ao+Z_ao.T)/2
    vj,vk = mf.Gradients().get_jk(dm=[Z_ao,dm])
    vhf_temp = vj - 0.5*vk
    
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        grad_Z[k] += einsum('xij,ij->x', vhf_temp[0][:,p0:p1], dm[p0:p1])*2
        grad_Z[k] += einsum('xij,ij->x', vhf_temp[1][:,p0:p1], Z_ao[p0:p1])*2
    if getattr(mf, 'with_df', None) is not None:
        grad_Z_aux = vj.aux - vk.aux * .5
        grad_Z_aux = grad_Z_aux[0,1] + grad_Z_aux[1,0]
        for k, ia in enumerate(atmlst):
            grad_Z[k] += grad_Z_aux[ia]
    
    Z_ao = (
        + einsum('ij,ui,vj,ij->uv',Z_hd,mo_coeff[:,sh],mo_coeff[:,sd],kappa[sh,sd])
        + einsum('ij,ui,vj,ij->uv',Z_vd,mo_coeff[:,sv],mo_coeff[:,sd],kappa[sv,sd])
        + einsum('ij,ui,vj,ij->uv',Z_vh,mo_coeff[:,sv],mo_coeff[:,sh],kappa[sv,sh])
    )
    Z_ao = (Z_ao+Z_ao.T)/2
    if mf.mol.spin != 0:
        vj,vk = mf.Gradients().get_jk(dm=[Z_ao,dma-dmb])
        vhf_temp = 0.5*vk
        
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            grad_Z[k] -= einsum('xij,ij->x', vhf_temp[0][:,p0:p1], (dma-dmb)[p0:p1])*2
            grad_Z[k] -= einsum('xij,ij->x', vhf_temp[1][:,p0:p1], Z_ao[p0:p1])*2
        if getattr(mf, 'with_df', None) is not None:
            grad_Z_aux = vk.aux * .5
            grad_Z_aux = grad_Z_aux[0,1] + grad_Z_aux[1,0]
            for k, ia in enumerate(atmlst):
                grad_Z[k] -= grad_Z_aux[ia]
    
    grad_Z -= einsum('p,Atp->At',Z,get_indep_pair_slice(einsum('q,Atpq->Atpq',mf.mo_energy,S_1_mo)))
    
    t0 = log.timer('grad Z-vector', *t0)
    return grad_Z

from pyscf.lib import logger

def grad_elec(mydmet_grad, atmlst=None):
    mydmet = mydmet_grad.base
    solver_option = mydmet_grad.solver_option
    mf = mydmet.mf_or_cas
    mo_coeff = mf.mo_coeff
    mol = mydmet.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    de = np.zeros((len(atmlst),3))
    log = logger.Logger(mydmet_grad.stdout, mydmet_grad.verbose)
    
    imp_idx = mydmet.imp_idx
    if not np.array_equal(np.array(imp_idx), np.arange(len(imp_idx))):
        raise NotImplementedError('DMET analytic gradient only support the case in which the impurity index is from zero and continuous.')
    
    es_mf = mydmet.es_mf
    fo_ene = mydmet.fo_ene
    ao2eo = mydmet.es_orb
    ao2core = mydmet.fo_orb
    dm_core = lib.dot(ao2core, ao2core.T)*2
    neo = ao2eo.shape[-1]
    nao = mol.nao
    get_indep_pair_slice = gen_get_indep_pair_slice(mf)
    
    es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = get_es_imds(es_mf, solver_option, log)
    S_1_ao, S_1_mo = get_S_1(mf, atmlst, log)
    H_1_ao, H_1_mo = get_H_1(mf, atmlst, log)
    caolo_1, cloao_1 = get_lowdin_1(mf, S_1_ao, log)
    epsilon_pq = mf.mo_energy[:,None] - mf.mo_energy[None,:]
    epsilon_indep_pair = get_indep_pair_slice(epsilon_pq)
    es_relaxed_rdm1_global = reduce(lib.dot, (ao2eo, es_relaxed_rdm1, ao2eo.T))
    I1, I2 = get_I(mf, dm_core, ao2eo, ao2core, es_relaxed_rdm1_global, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X, log)
    grad_I, W = get_grad_I(mydmet_grad, I1, I2, caolo_1, cloao_1, S_1_mo, log)
    
    t1 = (logger.process_clock(), logger.perf_counter())
    contract_Z_A = gen_contract_Z_A(mf)
    def vind_vo(Z):
        Z = Z.flatten()
        ZA = contract_Z_A(Z)
        ZA -= ZA.swapaxes(-1,-2)
        v = get_indep_pair_slice(ZA, axis=0)/epsilon_indep_pair
        return v
    Z = lib.krylov(vind_vo, -W/epsilon_indep_pair,tol=mydmet_grad._krylov_conv_tol,max_cycle=200,verbose=mydmet_grad.verbose)
    t1 = log.timer('Z-vector equation', *t1)
    
    de += get_grad_dm1(H_1_ao, dm_core, es_relaxed_rdm1_global, log)
    de += get_grad_veff(mf, atmlst, dm_core, es_relaxed_rdm1_global, log)
    de += get_grad_dm2(mf, atmlst, ao2eo, es_relaxed_rdm2, log)
    de += get_grad_X(S_1_ao, ao2eo, es_relaxed_X, log)
    de += grad_I
    de += get_grad_Z_vector(mf, atmlst, Z, H_1_mo, S_1_mo, contract_Z_A, get_indep_pair_slice, log)
    e = es_ene + fo_ene

    del es_relaxed_rdm2
    return de, e, es_solver

from pyscf.grad.rhf import GradientsBase
class SSDMET_Gradients(GradientsBase):
    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.mol = method.mol
        self.base = method
        self.max_memory = self.base.max_mem
        self.unit = 'au'
        self.es_solver = None
        self.solver_option = None
        self.es_ene = None
        self._krylov_conv_tol = 1e-7
        self._perturb_anyway = False
        self._perturbation = 1e-10

        self.atmlst = None
        self.de = None
        self.e = None
        
    grad_elec = grad_elec
    
    def kernel(self, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de, e, es_solver = self.grad_elec(atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        self.e = e
        self.es_solver = es_solver
        logger.timer(self, 'DMET gradients', *cput0)
        self._finalize()
        return self.de, self.e

SSDMET_Grad = SSDMET_Gradients

ssdmet.SSDMET.Gradients = lib.class_as_method(SSDMET_Gradients)
