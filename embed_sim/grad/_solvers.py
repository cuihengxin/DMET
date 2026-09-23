from pyscf import scf, mcscf, fci, ao2mo, lib
from pyscf.lib import logger
from functools import reduce, partial
import numpy as np
from embed_sim import sacasscf_mixer

from numpy import einsum
einsum = partial(einsum, optimize=True)

def ROHF(es_mf, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    es_solver = es_mf
    dma,dmb = es_solver.make_rdm1()
    dm = dma + dmb
    es_relaxed_rdm2 = (
        + einsum('ij,kl->ijkl',dm,dm)
        - einsum('il,jk->ijkl',dma,dma)
        - einsum('il,jk->ijkl',dmb,dmb)
    )
    dmea,dmeb = es_solver.Gradients().make_rdm1e(es_solver.mo_energy,es_solver.mo_coeff,es_solver.mo_occ)
    es_relaxed_rdm1 = dm
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = dmea + dmeb
    es_ene = es_solver.e_tot

    t0 = log.timer(f'embedded space ROHF intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def SA_CASSCF(es_mf, solver_option, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    ncas = solver_option['ncas']
    nelec = solver_option['nelec']
    state = solver_option.get('state', None)
    statelis = solver_option.get('statelis', None)
    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 50)
    es_mo = solver_option.get('init_guess', None)
    
    es_solver = sacasscf_mixer.sacasscf_mixer(es_mf, ncas, nelec, statelis)
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    es_solver.kernel(es_mo)
    t0 = log.timer(f'embedded space SA-CASSCF', *t0)

    if state is None:
        es_relaxed_rdm1_act, es_relaxed_rdm2_act = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        es_relaxed_rdm1_mo, es_relaxed_rdm2_mo = mcscf.addons._make_rdm12_on_mo(es_relaxed_rdm1_act,es_relaxed_rdm2_act,es_solver.ncore,es_solver.ncas,neo)
        mo = es_solver.mo_coeff
        eri = ao2mo.kernel(es_mf._eri, mo, compact=False).reshape([neo]*4)
        hcore = einsum('ij,ip,jq->pq', es_mf.get_hcore(), mo, mo)
        es_relaxed_X_mo = (einsum('pr,qr->pq',es_relaxed_rdm1_mo, hcore)
                           + einsum('ptrs,qtrs->pq',es_relaxed_rdm2_mo, eri))
        es_relaxed_rdm1 = einsum('pq,ip,jq->ij', es_relaxed_rdm1_mo, mo, mo)
        es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', es_relaxed_rdm2_mo, mo, mo, mo, mo)
        del es_relaxed_rdm2_mo
        es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
        es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
        es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
        es_relaxed_rdm2 *= 0.125
        es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)
        es_ene = es_solver.e_average
    else:
        converged, Lvec, bvec, Aop, Adiag = es_solver.Gradients(state=state).solve_lagrange()
        Lorb, Lci = es_solver.Gradients(state=state).unpack_uniq_var(Lvec)
        kappa = -Lorb
        t0 = log.timer(f'embedded space CP-SA-CASSCF equations', *t0)
        gamma_SA, Gamma_SA = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_SA, Gamma_SA = mcscf.addons._make_rdm12_on_mo(gamma_SA,Gamma_SA,es_solver.ncore,es_solver.ncas,neo)
        gamma_theta, Gamma_theta = es_solver.fcisolver.states_make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_theta, Gamma_theta = gamma_theta[state], Gamma_theta[state]
        gamma_theta, Gamma_theta = mcscf.addons._make_rdm12_on_mo(gamma_theta,Gamma_theta,es_solver.ncore,es_solver.ncas,neo)
        gamma_tilde = (
            + einsum('oq,op->pq',gamma_SA,kappa)
            - einsum('po,qo->pq',gamma_SA,kappa)
        )
        Gamma_tilde = (
            + einsum('oqrs,op->pqrs',Gamma_SA,kappa)
            + einsum('pors,oq->pqrs',Gamma_SA,kappa)
            + einsum('pqos,or->pqrs',Gamma_SA,kappa)
            + einsum('pqro,os->pqrs',Gamma_SA,kappa)
        )
        gamma_bar, Gamma_bar = es_solver.fcisolver.trans_rdm12(Lci,es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_bar += gamma_bar.T
        Gamma_bar += einsum('ijkl->lkji', Gamma_bar)
        def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
            nocc = ncas + ncore
            dm1 = np.zeros((nmo,nmo))
            idx = np.arange(ncore)
            dm1[ncore:nocc,ncore:nocc] = casdm1
        
            dm2 = np.zeros((nmo,nmo,nmo,nmo))
            dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
            for i in range(ncore):
                dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
                dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
            return dm1, dm2
        gamma_bar, Gamma_bar = _make_rdm12_on_mo(gamma_bar,Gamma_bar,es_solver.ncore,es_solver.ncas,neo)
        gamma_mo = gamma_theta + gamma_tilde + gamma_bar
        Gamma_mo = Gamma_theta + Gamma_tilde + Gamma_bar
        mo = es_solver.mo_coeff
        hcore = einsum('ij,ip,jq->pq',es_mf.get_hcore(), mo, mo)
        eri = ao2mo.kernel(es_mf._eri,mo,compact=False).reshape([neo]*4)
        es_relaxed_X_mo = (einsum('pr,qr->pq', gamma_mo, hcore)
                           + einsum('ptrs,qtrs->pq', Gamma_mo, eri))
        es_relaxed_rdm1 = einsum('pq,ip,jq->ij',gamma_mo, mo, mo)
        es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', Gamma_mo, mo, mo, mo, mo)
        del Gamma_mo, Gamma_SA, Gamma_theta, Gamma_tilde, Gamma_bar
        es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
        es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
        es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
        es_relaxed_rdm2 *= 0.125
        es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)
        es_ene = es_solver.e_states[state]
    
    t0 = log.timer(f'embedded space SA-CASSCF intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def SA_HCISCF(es_mf, solver_option, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    ncas = solver_option['ncas']
    nelec = solver_option['nelec']
    state = solver_option.get('state', None)
    statelis = solver_option.get('statelis', None)
    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 50)
    es_mo = solver_option.get('init_guess', None)
    
    from embed_sim.hci import HCI
    epsilon1 = solver_option.get('epsilon1', 1e-5)
    es_solver = mcscf.CASSCF(es_mf, ncas, nelec)
    hcisolver = HCI(es_mf.mol, epsilon1=epsilon1)
    hcisolver.spin = es_mf.mol.spin
    hcisolver.nroots = np.sum(statelis)
    hcisolver.prefix = 'solver'
    nroots = hcisolver.nroots
    es_solver = mcscf.addons.state_average_mix_(es_solver, [hcisolver], weights=np.ones(nroots)/nroots)
    es_solver.internal_rotation = True
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    es_solver.kernel(es_mo)
    t0 = log.timer(f'embedded space SA-HCISCF', *t0)

    if state is None:
        es_relaxed_rdm1_act, es_relaxed_rdm2_act = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        es_relaxed_rdm1_mo, es_relaxed_rdm2_mo = mcscf.addons._make_rdm12_on_mo(es_relaxed_rdm1_act,es_relaxed_rdm2_act,es_solver.ncore,es_solver.ncas,neo)
        mo = es_solver.mo_coeff
        eri = ao2mo.kernel(es_mf._eri, mo, compact=False).reshape([neo]*4)
        hcore = einsum('ij,ip,jq->pq', es_mf.get_hcore(), mo, mo)
        es_relaxed_X_mo = (einsum('pr,qr->pq',es_relaxed_rdm1_mo, hcore)
                           + einsum('ptrs,qtrs->pq',es_relaxed_rdm2_mo, eri))
        es_relaxed_rdm1 = einsum('pq,ip,jq->ij', es_relaxed_rdm1_mo, mo, mo)
        es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', es_relaxed_rdm2_mo, mo, mo, mo, mo)
        del es_relaxed_rdm2_mo
        es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
        es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
        es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
        es_relaxed_rdm2 *= 0.125
        es_relaxed_X = 0.5*(es_relaxed_X+es_relaxed_X.T)
        es_ene = es_solver.e_average
    else:
        converged, Lvec, bvec, Aop, Adiag = es_solver.Gradients(state=state).solve_lagrange()
        Lorb, Lci = es_solver.Gradients(state=state).unpack_uniq_var(Lvec)
        kappa = -Lorb
        t0 = log.timer(f'embedded space CP-SA-HCISCF equations', *t0)
        gamma_SA, Gamma_SA = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_SA, Gamma_SA = mcscf.addons._make_rdm12_on_mo(gamma_SA,Gamma_SA,es_solver.ncore,es_solver.ncas,neo)
        gamma_theta, Gamma_theta = es_solver.fcisolver.states_make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_theta, Gamma_theta = gamma_theta[state], Gamma_theta[state]
        gamma_theta, Gamma_theta = mcscf.addons._make_rdm12_on_mo(gamma_theta,Gamma_theta,es_solver.ncore,es_solver.ncas,neo)
        gamma_tilde = (
            + einsum('oq,op->pq',gamma_SA,kappa)
            - einsum('po,qo->pq',gamma_SA,kappa)
        )
        Gamma_tilde = (
            + einsum('oqrs,op->pqrs',Gamma_SA,kappa)
            + einsum('pors,oq->pqrs',Gamma_SA,kappa)
            + einsum('pqos,or->pqrs',Gamma_SA,kappa)
            + einsum('pqro,os->pqrs',Gamma_SA,kappa)
        )
        gamma_bar, Gamma_bar = es_solver.fcisolver.trans_rdm12(Lci,es_solver.ci,es_solver.ncas,es_solver.nelecas)
        gamma_bar += gamma_bar.T
        Gamma_bar += einsum('ijkl->lkji', Gamma_bar)
        def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
            nocc = ncas + ncore
            dm1 = np.zeros((nmo,nmo))
            idx = np.arange(ncore)
            dm1[ncore:nocc,ncore:nocc] = casdm1
        
            dm2 = np.zeros((nmo,nmo,nmo,nmo))
            dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
            for i in range(ncore):
                dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
                dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
            return dm1, dm2
        gamma_bar, Gamma_bar = _make_rdm12_on_mo(gamma_bar,Gamma_bar,es_solver.ncore,es_solver.ncas,neo)
        gamma_mo = gamma_theta + gamma_tilde + gamma_bar
        Gamma_mo = Gamma_theta + Gamma_tilde + Gamma_bar
        mo = es_solver.mo_coeff
        hcore = einsum('ij,ip,jq->pq',es_mf.get_hcore(), mo, mo)
        eri = ao2mo.kernel(es_mf._eri,mo,compact=False).reshape([neo]*4)
        es_relaxed_X_mo = (einsum('pr,qr->pq', gamma_mo, hcore)
                           + einsum('ptrs,qtrs->pq', Gamma_mo, eri))
        es_relaxed_rdm1 = einsum('pq,ip,jq->ij', gamma_mo, mo, mo)
        es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', Gamma_mo, mo, mo, mo, mo)
        del Gamma_mo, Gamma_SA, Gamma_theta, Gamma_tilde, Gamma_bar
        es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
        es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
        es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
        es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
        es_relaxed_rdm2 *= 0.125
        es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)
        es_ene = es_solver.e_states[state]
    
    t0 = log.timer(f'embedded space SA-HCISCF intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def alpha_CASSCF(es_mf, solver_option, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    ncas = solver_option['ncas']
    nelec = solver_option['nelec']
    alpha = solver_option['alpha']
    state = solver_option.get('state', None)
    statelis = solver_option.get('statelis', None)
    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 50)
    es_mo = solver_option.get('init_guess', None)
    
    es_solver = sacasscf_mixer.sacasscf_mixer(es_mf, ncas, nelec, statelis)
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    es_solver.kernel(es_mo)
    t0 = log.timer(f'embedded space SA-CASSCF', *t0)
    converged, Lvec, bvec, Aop, Adiag = es_solver.Gradients(state=state).solve_lagrange()
    Lorb, Lci = es_solver.Gradients(state=state).unpack_uniq_var(Lvec)
    kappa = -Lorb
    t0 = log.timer(f'embedded space CP-SA-CASSCF equations', *t0)
    gamma_SA, Gamma_SA = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
    gamma_SA, Gamma_SA = mcscf.addons._make_rdm12_on_mo(gamma_SA,Gamma_SA,es_solver.ncore,es_solver.ncas,neo)
    gamma_theta, Gamma_theta = es_solver.fcisolver.states_make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
    gamma_theta, Gamma_theta = gamma_theta[state], Gamma_theta[state]
    gamma_theta, Gamma_theta = mcscf.addons._make_rdm12_on_mo(gamma_theta,Gamma_theta,es_solver.ncore,es_solver.ncas,neo)
    gamma_tilde = (
        + einsum('oq,op->pq',gamma_SA,kappa)
        - einsum('po,qo->pq',gamma_SA,kappa)
    )
    Gamma_tilde = (
        + einsum('oqrs,op->pqrs',Gamma_SA,kappa)
        + einsum('pors,oq->pqrs',Gamma_SA,kappa)
        + einsum('pqos,or->pqrs',Gamma_SA,kappa)
        + einsum('pqro,os->pqrs',Gamma_SA,kappa)
    )
    gamma_bar, Gamma_bar = es_solver.fcisolver.trans_rdm12(Lci,es_solver.ci,es_solver.ncas,es_solver.nelecas)
    gamma_bar += gamma_bar.T
    Gamma_bar += einsum('ijkl->lkji', Gamma_bar)
    def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
        nocc = ncas + ncore
        dm1 = np.zeros((nmo,nmo))
        idx = np.arange(ncore)
        dm1[ncore:nocc,ncore:nocc] = casdm1
    
        dm2 = np.zeros((nmo,nmo,nmo,nmo))
        dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
        for i in range(ncore):
            dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
            dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
        return dm1, dm2
    gamma_bar, Gamma_bar = _make_rdm12_on_mo(gamma_bar,Gamma_bar,es_solver.ncore,es_solver.ncas,neo)
    gamma_mo = gamma_theta + gamma_tilde + gamma_bar
    Gamma_mo = Gamma_theta + Gamma_tilde + Gamma_bar
    mo = es_solver.mo_coeff
    hcore = einsum('ij,ip,jq->pq',es_mf.get_hcore(), mo, mo)
    eri = ao2mo.kernel(es_mf._eri,mo,compact=False).reshape([neo]*4)
    es_relaxed_X_mo = (einsum('pr,qr->pq', gamma_mo, hcore)
                       + einsum('ptrs,qtrs->pq', Gamma_mo, eri))
    es_relaxed_rdm1 = einsum('pq,ip,jq->ij', gamma_mo, mo, mo)
    es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', Gamma_mo, mo, mo, mo, mo)
    del Gamma_mo, Gamma_SA, Gamma_theta, Gamma_tilde, Gamma_bar
    es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
    es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)
    
    es_relaxed_rdm1_act, es_relaxed_rdm2_act = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
    es_relaxed_rdm1_SA_mo, es_relaxed_rdm2_SA_mo = mcscf.addons._make_rdm12_on_mo(es_relaxed_rdm1_act,es_relaxed_rdm2_act,es_solver.ncore,es_solver.ncas,neo)
    es_relaxed_X_SA_mo = (einsum('pr,qr->pq',es_relaxed_rdm1_SA_mo, hcore)
                          + einsum('ptrs,qtrs->pq',es_relaxed_rdm2_SA_mo, eri))
    es_relaxed_rdm1_SA = einsum('pq,ip,jq->ij', es_relaxed_rdm1_SA_mo, mo, mo)
    es_relaxed_rdm2_SA = einsum('pqrs,ip,jq,kr,ls->ijkl', es_relaxed_rdm2_SA_mo, mo, mo, mo, mo)
    es_relaxed_X_SA = einsum('pq,ip,jq->ij', es_relaxed_X_SA_mo, mo, mo)
    es_relaxed_rdm1_SA = 0.5*(es_relaxed_rdm1_SA + es_relaxed_rdm1_SA.T)
    es_relaxed_rdm2_SA += einsum('ijkl->ijlk', es_relaxed_rdm2_SA)
    es_relaxed_rdm2_SA += einsum('ijkl->jikl', es_relaxed_rdm2_SA)
    es_relaxed_rdm2_SA += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2_SA *= 0.125
    es_relaxed_X_SA = 0.5*(es_relaxed_X_SA + es_relaxed_X_SA.T)
    
    es_relaxed_rdm1 = alpha*es_relaxed_rdm1 + (1-alpha)*es_relaxed_rdm1_SA
    es_relaxed_rdm2 = alpha*es_relaxed_rdm2 + (1-alpha)*es_relaxed_rdm2_SA
    es_relaxed_X = alpha*es_relaxed_X + (1-alpha)*es_relaxed_X_SA
    es_ene = alpha*es_solver.e_states[state] + (1-alpha)*es_solver.e_average
    
    t0 = log.timer(f'embedded space alpha-CASSCF intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def CASSCF(es_mf, solver_option, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    ncas = solver_option['ncas']
    nelec = solver_option['nelec']
    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 50)
    es_mo = solver_option.get('init_guess', None)
    fix_spin = solver_option.get('fix_spin', True)
    
    es_solver = mcscf.CASSCF(es_mf, ncas, nelec)
    if fix_spin:
        ss = es_mf.mol.spin
        es_solver.fcisolver.spin = ss
        es_solver.fcisolver = fci.addons.fix_spin(es_solver.fcisolver,ss=ss,shift=0.5)
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    if 'mc2step' in solver_option.keys() and solver_option['mc2step']:
        es_solver.mc2step(es_mo)
    else:
        es_solver.kernel(es_mo)
    t0 = log.timer(f'embedded space CASSCF', *t0)
    
    es_relaxed_rdm1_act, es_relaxed_rdm2_act = es_solver.fcisolver.make_rdm12(es_solver.ci,es_solver.ncas,es_solver.nelecas)
    es_relaxed_rdm1_mo, es_relaxed_rdm2_mo = mcscf.addons._make_rdm12_on_mo(es_relaxed_rdm1_act,es_relaxed_rdm2_act,es_solver.ncore,es_solver.ncas,neo)
    mo = es_solver.mo_coeff
    eri = ao2mo.kernel(es_mf._eri, mo, compact=False).reshape([neo]*4)
    hcore = einsum('ij,ip,jq->pq', es_mf.get_hcore(), mo, mo)
    es_relaxed_X_mo = (einsum('pr,qr->pq',es_relaxed_rdm1_mo, hcore)
                       + einsum('ptrs,qtrs->pq',es_relaxed_rdm2_mo, eri))
    es_relaxed_rdm1 = einsum('pq,ip,jq->ij', es_relaxed_rdm1_mo, mo, mo)
    es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', es_relaxed_rdm2_mo, mo, mo, mo, mo)
    del es_relaxed_rdm2_mo
    es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
    es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)
    es_ene = es_solver.e_tot
    
    t0 = log.timer(f'embedded space CASSCF intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def MP2(es_mf, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    from pyscf import mp
    from pyscf.ao2mo import _ao2mo
    from pyscf.grad.mp2 import _index_frozen_active, has_frozen_orbitals, _response_dm1
    from pyscf.cc import ccsd_rdm
    from pyscf.grad import rhf as rhf_grad
    from pyscf.scf.addons import convert_to_rhf

    if es_mf.mol.spin != 0:
        raise NotImplementedError
    es_mf = convert_to_rhf(es_mf, scf.RHF(es_mf.mol))
    
    es_solver = mp.MP2(es_mf)
    es_solver.kernel()
    mo_coeff = es_solver.mo_coeff
    mo_energy = es_solver._scf.mo_energy
    mo_occ = es_solver.mo_occ
    t2 = es_solver.t2
    d1 = mp.mp2._gamma1_intermediates(es_solver, t2)
    doo, dvv = d1
    with_frozen = has_frozen_orbitals(es_solver)
    OA, VA, OF, VF = _index_frozen_active(es_solver.get_frozen_mask(), mo_occ)
    orbo = mo_coeff[:,OA]
    orbv = mo_coeff[:,VA]
    nao, nocc = orbo.shape
    nmo = nao
    nvir = orbv.shape[1]

    part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc**2,nvir**2),
                            np.asarray(orbv.T, order='F'), (0,nao,0,nao),
                            's1', 's1').reshape(nocc,nocc,nao,nao)
    # part_dm2 = (part_dm2.transpose(0,2,3,1) * 4 -
    #             part_dm2.transpose(0,3,2,1) * 2)
    part_dm2 = 4 * einsum('ijkl->iklj', part_dm2) - 2 * np.einsum('ijkl->ilkj', part_dm2)
    hf_dm1 = es_solver._scf.make_rdm1(mo_coeff, mo_occ)
    dm2 = es_solver.make_rdm2()
    mp_dm1 = es_solver.make_rdm1()
    mp_dm1[np.diag_indices(nocc)] -= 2
    for i in range(nocc):
        dm2[i,i,:,:] -= mp_dm1.T * 2
        dm2[:,:,i,i] -= mp_dm1.T * 2
        dm2[:,i,i,:] += mp_dm1.T
        dm2[i,:,:,i] += mp_dm1
    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] -= 4
            dm2[i,j,j,i] += 2
    dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mo_coeff)

    Imat0 = einsum('ipkl,iqkl->pq', ao2mo.restore(1, es_solver._scf._eri, nmo), dm2)
    Imat = reduce(np.dot, (mo_coeff.T, Imat0, es_solver._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = es_solver._scf.get_veff(dm=dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(es_solver, Xvo)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(np.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = np.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(np.dot, (p1, es_solver._scf.get_veff(dm=dm1+dm1.T), p1))

    dm1p = hf_dm1*0.5 + dm1
    dm1 += hf_dm1
    zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    es_relaxed_rdm1 = dm1
    es_relaxed_rdm2 = (
        + einsum('kl,ij->ijkl', hf_dm1, dm1p)
        + einsum('kl,ij->ijkl', dm1p, hf_dm1)
        - 0.5*einsum('jk,il->ijkl', hf_dm1, dm1p)
        - 0.5*einsum('jk,il->ijkl', dm1p, hf_dm1)
        + dm2
    )
    es_relaxed_X = -(im1 + im1.T)/2 + (zeta+zeta.T)/2 + vhf_s1occ
    es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)

    es_ene = es_solver.e_tot
    
    t0 = log.timer(f'embedded space MP2 intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def CCSD(es_mf, solver_option, log=None):
    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    from pyscf import cc
    from pyscf.grad.mp2 import _index_frozen_active, has_frozen_orbitals
    from pyscf.grad.ccsd import _rdm2_mo2ao, _response_dm1
    from pyscf.grad import rhf as rhf_grad
    from pyscf.cc import ccsd_rdm
    from pyscf.scf.addons import convert_to_rhf

    if es_mf.mol.spin != 0:
        raise NotImplementedError
    es_mf = convert_to_rhf(es_mf, scf.RHF(es_mf.mol))

    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 100)

    es_solver = cc.CCSD(es_mf)
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    es_solver.kernel()
    es_solver.solve_lambda()
    t1, t2, l1, l2 = es_solver.t1, es_solver.t2, es_solver.l1, es_solver.l2
    eris = None
    d1 = ccsd_rdm._gamma1_intermediates(es_solver, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    fdm2 = lib.H5TmpFile()
    d2 = ccsd_rdm._gamma2_outcore(es_solver, t1, t2, l1, l2, fdm2, True)

    mo_coeff = es_solver.mo_coeff
    mo_energy = es_solver._scf.mo_energy
    mo_occ = es_solver.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mo_occ > 0)
    with_frozen = has_frozen_orbitals(es_solver)
    OA, VA, OF, VF = _index_frozen_active(es_solver.get_frozen_mask(), mo_occ)

    mo_active = mo_coeff[:,np.hstack((OA,VA))]
    _rdm2_mo2ao(es_solver, d2, mo_active, fdm2)
    hf_dm1 = es_solver._scf.make_rdm1(mo_coeff, mo_occ)
    Imat0 = einsum('ipkl,iqkl->pq', ao2mo.restore(1, es_solver._scf._eri, nmo), ao2mo.restore(1, fdm2['dm2'], nmo)) * 0.5
    Imat = reduce(np.dot, (mo_coeff.T, Imat0, es_solver._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = es_solver._scf.get_veff(dm=dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(es_solver, Xvo, eris)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(np.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = np.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(np.dot, (p1, es_solver._scf.get_veff(dm=dm1+dm1.T), p1))

    dm1p = hf_dm1*0.5 + dm1
    dm1 += hf_dm1
    zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    es_relaxed_rdm1 = dm1
    es_relaxed_rdm2 = (
        + np.einsum('kl,ij->ijkl', hf_dm1, dm1p)
        + np.einsum('kl,ij->ijkl', dm1p, hf_dm1)
        - 0.5*np.einsum('jk,il->ijkl', hf_dm1, dm1p)
        - 0.5*np.einsum('jk,il->ijkl', dm1p, hf_dm1)
        + ao2mo.restore(1, fdm2['dm2'], nmo) * 0.5
    )
    es_relaxed_X = -(im1+im1.T)/2 + (zeta+zeta.T)/2 + vhf_s1occ
    es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)

    es_ene = es_solver.e_tot
    
    t0 = log.timer(f'embedded space CCSD intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X

def DSRG_MRPT2(es_mf, solver_option, log=None):
    import re, os
    import forte
    import psi4
    from forte.modules import (
        OptionsFactory,
        ObjectsFromPySCF,
        ActiveSpaceInts,
        ActiveSpaceSolver,
    )
    from forte.proc.dsrg import ProcedureDSRG

    if log is None:
        log = logger.new_logger(es_mf, es_mf.verbose)
    neo = es_mf.mo_coeff.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())

    ncas = solver_option['ncas']
    nelec = solver_option['nelec']
    conv_tol = solver_option.get('conv_tol', 1e-9)
    max_cycle = solver_option.get('max_cycle', 50)
    es_mo = solver_option.get('init_guess', None)
    fix_spin = solver_option.get('fix_spin', True)
    flow_param = solver_option.get('flow_param', 1.0)
    
    es_mf.mol.nao_nr = lambda *args: neo
    es_solver = mcscf.CASSCF(es_mf, ncas, nelec)
    if fix_spin:
        ss = es_mf.mol.spin
        try:
            from pyscf.csf_fci import csf_solver
        except ImportError:
            es_solver.fcisolver.spin = ss
            es_solver.fcisolver = fci.addons.fix_spin(es_solver.fcisolver, ss=ss, shift=0.5)
        else:
            es_solver.fcisolver = csf_solver(es_mf.mol, smult=ss+1)
    es_solver.conv_tol = conv_tol
    es_solver.max_cycle = max_cycle
    if 'mc2step' in solver_option.keys() and solver_option['mc2step']:
        es_solver.mc2step(es_mo)
    else:
        es_solver.kernel(es_mo)
    
    core = slice(0, es_solver.ncore)
    active = slice(es_solver.ncore, es_solver.ncore + es_solver.ncas)
    virt = slice(es_solver.ncore + es_solver.ncas, neo)
    _fock_canon = einsum('pi,pq,qj->ij', es_solver.mo_coeff, es_solver.get_fock(), es_solver.mo_coeff)
    semicanonicalize_core = np.linalg.eigh(_fock_canon[core, core])[1]
    semicanonicalize_active = np.linalg.eigh(_fock_canon[active, active])[1]
    semicanonicalize_virt = np.linalg.eigh(_fock_canon[virt, virt])[1]
    C_canon = np.hstack([
        es_solver.mo_coeff[:,core] @ semicanonicalize_core,
        es_solver.mo_coeff[:,active] @ semicanonicalize_active,
        es_solver.mo_coeff[:,virt] @ semicanonicalize_virt
    ])
    es_solver.max_cycle = 0
    _ = es_solver.kernel(C_canon)

    t0 = log.timer(f'embedded space CASSCF', *t0)

    options = {'JOB_TYPE':'NEWDRIVER',
               'PRINT':0,
               'INT_TYPE':'PYSCF',
               'REF_TYPE':'CASSCF',
               'ACTIVE_SPACE_SOLVER':'DETCI',
               'CORRELATION_SOLVER':'DSRG-MRPT2',
               'FORCE_DIAG_METHOD':True,
               'RESTRICTED_DOCC':[es_solver.ncore],
               'RESTRICTED_UOCC':[neo-es_solver.ncore-es_solver.ncas],
               'ACTIVE':[es_solver.ncas],
               'DSRG_S':flow_param,
               'MCSCF_NO_ORBOPT':True,
               'MCSCF_REFERENCE':True,
               'SEMI_CANONICAL':False,
               'MULTIPLICITY':es_mf.mol.spin+1,
               'MCSCF_MULTIPLICITY':es_mf.mol.spin+1,
               }
    forte.banner()
    max_memory = int(es_solver.max_memory - lib.current_memory()[0]) * 1e6
    psi4.set_memory(max_memory)
    psi4.set_num_threads(lib.num_threads())
    data = OptionsFactory(options=options)
    data = data.run()
    data = ObjectsFromPySCF(es_solver, options={'forte_options':options, 'pyscf_obj':es_solver}).run(data)

    state_map = forte.to_state_nroots_map(data.state_weights_map)
    data = ActiveSpaceInts(active="ACTIVE", core=["RESTRICTED_DOCC"]).run(data)
    
    active_space_solver_type = data.options.get_str("ACTIVE_SPACE_SOLVER")
    data = ActiveSpaceSolver(solver_type=active_space_solver_type).run(data)
    
    dsrg_proc = ProcedureDSRG(data.active_space_solver, data.state_weights_map, data.mo_space_info, data.ints, data.options, data.scf_info)
    es_ene = dsrg_proc.compute_energy()
    
    state = list(state_map.keys())[0]
    ci_vectors = data.active_space_solver.eigenvectors(state)
    dsrg_proc.compute_gradient(ci_vectors)

    # def parse_forte_1(dat, size):
    #     with open(dat, 'r+') as f:
    #         lines = f.read()
    #     data = re.findall(r'[-+]?\d*\.\d+', lines)
    #     mat = np.zeros((size,size), dtype=np.float64)
    #     for p0,p1 in lib.prange(0, size, 5):
    #         mat[:,p0:p1] = np.array(data[p0*size:p1*size], dtype=np.float64).reshape(size,p1-p0)
    #     return mat
    def parse_forte_2(dataa, datbb, datab, size):
        mat = np.zeros([size]*4, dtype=np.float64)
        
        with open(dataa, 'r+') as f:
            lines = f.readlines()
        lines = [line for line in lines]
        for line in lines:
            line = line.split()
            i, j, k, l = [int(x) for x in line[:4]]
            val = np.float64(line[-1]) * 4
            mat[i,k,j,l] += val
            mat[i,l,k,j] -= val
        
        with open(datbb, 'r+') as f:
            lines = f.readlines()
        lines = [line for line in lines]
        for line in lines:
            line = line.split()
            i, j, k, l = [int(x) for x in line[:4]]
            val = np.float64(line[-1]) * 4
            mat[i,k,j,l] += val
            mat[i,l,k,j] -= val
        
        with open(datab, 'r+') as f:
            lines = f.readlines()
        lines = [line for line in lines]
        for line in lines:
            line = line.split()
            i, j, k, l = [int(x) for x in line[:4]]
            val = np.float64(line[-1]) * 2
            mat[i,j,k,l] += val
        return mat
    # es_relaxed_rdm1_mo = parse_forte_1('D1.dat', neo)*2
    # es_relaxed_X_mo = parse_forte_1('L.dat', neo)
    es_relaxed_rdm1_mo = np.array(dsrg_proc.dsrg_solver.relaxed_rdm1_)*2
    es_relaxed_X_mo = np.array(dsrg_proc.dsrg_solver.lagrangian_)
    es_relaxed_rdm2_mo = parse_forte_2('d2aa.dat', 'd2bb.dat', 'd2ab.dat', neo)
    
    # os.remove('D1.dat')
    # os.remove('L.dat')
    os.remove('d2aa.dat')
    os.remove('d2bb.dat')
    os.remove('d2ab.dat')
    
    mo = es_solver.mo_coeff
    es_relaxed_rdm1 = einsum('pq,ip,jq->ij', es_relaxed_rdm1_mo, mo, mo)
    es_relaxed_X = einsum('pq,ip,jq->ij', es_relaxed_X_mo, mo, mo)
    es_relaxed_rdm2 = einsum('pqrs,ip,jq,kr,ls->ijkl', es_relaxed_rdm2_mo, mo, mo, mo, mo)
    del es_relaxed_rdm2_mo
    es_relaxed_rdm1 = 0.5*(es_relaxed_rdm1 + es_relaxed_rdm1.T)
    es_relaxed_rdm2 += einsum('ijkl->ijlk', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->jikl', es_relaxed_rdm2)
    es_relaxed_rdm2 += einsum('ijkl->klij', es_relaxed_rdm2)
    es_relaxed_rdm2 *= 0.125
    es_relaxed_X = 0.5*(es_relaxed_X + es_relaxed_X.T)

    t0 = log.timer(f'embedded space DSRG-MRPT2(c) intermediates', *t0)
    return es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X
