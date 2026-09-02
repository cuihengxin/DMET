import numpy as np
from functools import reduce
from scipy.linalg import block_diag
import h5py

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, ao2mo

from embed_sim.BNO_bath import get_RMP2_bath, get_UMP2_bath, get_ROMP2_bath, get_RMP2_bath_sos, get_UMP2_bath_sos, get_ROMP2_bath_sos
from embed_sim.bath_selection import count_imp_env_bonds, partition_env_by_bath_count
from embed_sim import iao_helper
from embed_sim import ic_helper

import os

def compare_imp_idx(imp_idx1, imp_idx2):
    imp_idx1 = np.array(imp_idx1)
    imp_idx2 = np.array(imp_idx2)
    try:
        return np.all(imp_idx1 == imp_idx2)
    except ValueError:
        return False

def mf_or_cas_make_rdm1s(mf_or_cas):
    from pyscf.scf.hf import RHF
    from pyscf.scf.rohf import ROHF
    from embed_sim.cahf import CAHF
    from pyscf.mcscf.mc1step import CASSCF
    # I don't know whether there is a general way to calculate rdm1s
    # If there is, better to use that function
    if isinstance(mf_or_cas, CASSCF): 
        print('DMET from CASSCF')
        dma, dmb = mf_or_cas.make_rdm1s()
        dm = np.stack((dma, dmb), axis=0)
    elif isinstance(mf_or_cas, CAHF):
        dma = dmb = np.dot(mf_or_cas.mo_coeff*mf_or_cas.mo_occ, mf_or_cas.mo_coeff.conj().T) / 2
        dm = np.stack((dma, dmb), axis=0)
    elif isinstance(mf_or_cas, ROHF):
        print('DMET from ROHF')
        dma, dmb = mf_or_cas.make_rdm1()
        dm = np.stack((dma, dmb), axis=0)
    elif isinstance(mf_or_cas, RHF):
        print('DMET from RHF')
        dm = mf_or_cas.make_rdm1()
    else:
        raise TypeError('starting point not supported',  mf_or_cas.__class__)
    return dm

def lowdin_orth(mol, ovlp=None):
    # lowdin orthonormalize
    if ovlp is None:
        s = mol.intor_symmetric('int1e_ovlp')
    else:
        s = ovlp
    caolo, cloao = lowdin(s), lowdin(s) @ s # caolo=lowdin(s)=s^-1/2, cloao=lowdin(s)@s=s^1/2
    return caolo, cloao
    

def build_embeded_subspace(ldm, imp_idx, lo_meth='lowdin', thres=1e-12, es_natorb=True,
                           bath_norb=None, bath_core_cutoff=0.5):
    """
    Returns C(AO->AS), entropy loss, and orbital composition
    """
    # from orthonormalized obital 

    # s = mf_or_cas.mol.intor_symmetric('int1e_ovlp')
    # caolo, cloao = lowdin(s), lowdin(s) @ s # caolo=lowdin(s)=s^-1/2, cloao=lowdin(s)@s=s^1/2
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]

    # dma, dmb = mf_or_cas_make_rdm1s(mf_or_cas) # in atomic orbital

    # ldma = reduce(np.dot,(cloao,dma,cloao.conj().T)) # in lowdin orbital
    # ldmb = reduce(np.dot,(cloao,dmb,cloao.conj().T))

    # ldm = ldma+ldmb

    # ldm = reduce(np.dot,(cloao,dm,cloao.conj().T))

    # ldma_env = ldma[env_idx,:][:,env_idx]
    # ldmb_env = ldmb[env_idx,:][:,env_idx]

    # nat_occa, nat_coeffa = np.linalg.eigh(ldma_env)
    # nat_occb, nat_coeffb = np.linalg.eigh(ldmb_env)

    ldm_imp = ldm[imp_idx,:][:,imp_idx]
    ldm_env = ldm[env_idx,:][:,env_idx]
    ldm_imp_env = ldm[imp_idx,:][:,env_idx]
    ldm_env_imp = ldm[env_idx,:][:,imp_idx]

    occ_env, orb_env = np.linalg.eigh(ldm_env) # occupation and orbitals on environment

    nimp = len(imp_idx)
    if bath_norb is None:
        nfv = np.sum(occ_env <  thres) # frozen virtual
        nbath = np.sum((occ_env >= thres) & (occ_env <= 2-thres)) # bath orbital
        nfo = np.sum(occ_env > 2-thres) # frozen occupied

        # defined w.r.t enviroment orbital index
        fv_idx = np.nonzero(occ_env <  thres)[0]
        bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 2-thres))[0]
        fo_idx = np.nonzero(occ_env > 2-thres)[0]
    else:
        # Fixed-size bath: keep the `bath_norb` environment natural orbitals
        # with occupation closest to 1 (one bath orbital per bond scheme,
        # Sun & Chan, JCTC 10, 3784 (2014)).
        bath_idx, fo_idx, fv_idx = partition_env_by_bath_count(
            occ_env, bath_norb, thres=thres, core_cutoff=bath_core_cutoff)
        nbath = len(bath_idx)
        nfo = len(fo_idx)
        nfv = len(fv_idx)

    orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx], orb_env[:, fv_idx]))
    
    if es_natorb:
        ldm_es = np.block([[ldm_imp, ldm_imp_env @ orb_env[:,0:nbath]],
                           [orb_env[:,0:nbath].T.conj() @ ldm_env_imp, orb_env[:,0:nbath].T.conj() @ ldm_env @ orb_env[:,0:nbath]]])
        es_occ, es_nat_orb = np.linalg.eigh(ldm_es)
        es_occ = es_occ[::-1]
        es_nat_orb = es_nat_orb[:,::-1]

        cloes = block_diag(np.eye(nimp), orb_env) @ block_diag(es_nat_orb, np.eye(nfo+nfv))
    else:
        es_occ = None
        cloes = block_diag(np.eye(nimp), orb_env)
    
    rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    cloes = cloes[rearange_idx, :]

    return cloes, nimp, nbath, nfo, nfv, es_occ

def get_rdiis_property(ldm1s, imp_idx, rdiis_property='dS', thres=1e-12):
    # for RDIIS
    ldm = ldm1s[0]+ldm1s[1]
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]

    ldm_env = ldm[env_idx,:][:,env_idx]

    occ_env, orb_env = np.linalg.eigh(ldm_env)

    ldma_env = ldm1s[0][env_idx,:][:,env_idx]
    ldmb_env = ldm1s[1][env_idx,:][:,env_idx]

    if rdiis_property == 'P':
        pol = np.trace(ldma_env-ldmb_env)
        return pol
    
    if rdiis_property == 'dS':
        occ_enva, nat_coeffa = np.linalg.eigh(ldma_env)
        occ_envb, nat_coeffb = np.linalg.eigh(ldmb_env)

        occ_enva = occ_enva[occ_enva > thres]
        occ_envb = occ_envb[occ_envb > thres]
        occ_env = occ_env[occ_env > thres]
        occ_enva = occ_enva[occ_enva < 1-thres]
        occ_envb = occ_envb[occ_envb < 1-thres]
        occ_env = occ_env[occ_env < 2-thres]
        
        ent = - np.sum(occ_enva*np.log(occ_enva)) - np.sum(occ_envb*np.log(occ_envb))
        ent2 = - np.sum((1-occ_enva)*np.log(1-occ_enva)) - np.sum((1-occ_envb)*np.log(1-occ_envb))
        entr = -2*np.sum(occ_env/2*np.log(occ_env/2))
        entr2 = -2*np.sum((1-occ_env/2)*np.log(1-occ_env/2))
        return entr - ent

def round_off_occ(mo_occ, threshold = 1e-8): 
    # round off occpuation close to 2 or 0 to be integral 
    mo_occ = np.where(np.abs(mo_occ-2)>threshold, mo_occ, int(2))
    mo_occ = np.where(np.abs(mo_occ-1)>threshold, mo_occ, int(1))
    mo_occ = np.where(np.abs(mo_occ)>threshold, mo_occ, int(0))
    return mo_occ

def split_occ(mo_occ):
    if mo_occ.ndim == 2:
        return round_off_occ(mo_occ)
    else:
        mo_occ = round_off_occ(mo_occ)
        split = np.zeros((2, np.shape(mo_occ)[0]))
        split[0] = np.where(mo_occ-1>-1e-8, 1, 0)
        split[1] = np.where(mo_occ-2>-1e-8, 1, 0)
        return split

def make_es_int1e(mf_or_cas, fo_orb, es_orb):
    hcore = mf_or_cas.get_hcore() # DO NOT use get_hcore(mol), since x2c 1e term is not included

    # HF J/K from env frozen occupied orbital
    fo_dm = fo_orb @ fo_orb.T.conj()*2
    vj, vk = mf_or_cas.get_jk(mol=mf_or_cas.mol, dm=fo_dm)

    fock = hcore + vj - 0.5 * vk

    es_int1e = reduce(np.dot, (es_orb.T.conj(), fock, es_orb)) # AO to embedded space
    return es_int1e

def make_es_int2e(mf, es_orb):
    if getattr(mf, 'with_df', False):
        es_int2e = mf.with_df.ao2mo(es_orb)
    else:
        es_int2e = ao2mo.full(mf.mol, es_orb)
    return ao2mo.restore(8, es_int2e, es_orb.shape[-1])

from pyscf import lib
from pyscf.lib import logger

class SSDMET(lib.StreamObject):
    """
    single-shot DMET with impurity-environment partition
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, es_natorb=True,
                 bath_option=None, bath_norb=None, readmp2=False, bath_core_cutoff=0.5, verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory # TODO
        self.readmp2 = readmp2
        self.verbose = verbose # TODO
        self.log = lib.logger.new_logger(self.mol, self.verbose)

        # inputs
        self.dm = None
        self.dm_pair = None
        self._imp_idx = []
        if imp_idx is not None:
            self.imp_idx = imp_idx
        else:
            self.log.info('impurity index not assigned, use the first atom as impurity')
            self.imp_idx = self.mol.atom_symbol(0)
        self.threshold = threshold
        self.es_natorb = es_natorb
        self.bath_option = bath_option
        # Fixed-size bath selection (one bath orbital per bond scheme):
        #   None       -> default threshold-based selection
        #   int        -> exactly `bath_norb` most-entangled bath orbitals
        #   'per_bond' -> number of impurity-environment bonds (resolved in build)
        self.bath_norb = bath_norb
        self.bath_core_cutoff = bath_core_cutoff

        # NOT inputs
        self.fo_orb = None
        self.fv_orb = None
        self.es_orb = None
        self.es_occ = None

        self.nfo = None
        self.nfv = None
        self.nes = None

        self.es_int1e = None
        self.es_int2e = None

        self.es_mf = None

        # Added for extensions
        self.caolo = None
        self.cloao = None
        self.lo_cloes = None
        self.open_shell = None

    def dump_flags(self):
        log = logger.new_logger(self, 4)
        log.info('')
        log.info('******** %s ********', self.__class__)

    @property
    def imp_idx(self):
        return self._imp_idx
    
    @imp_idx.setter
    def imp_idx(self, imp_idx):
        self._imp_idx = gto.mole._aolabels2baslst(self.mol, imp_idx, base=0)

    def make_es_int1e(self):
        return make_es_int1e(self.mf_or_cas, self.fo_orb, self.es_orb)

    def make_es_int2e(self):
        return make_es_int2e(self.mf_or_cas, self.es_orb)
    
    def load_chk(self, chk_fname):
        try:
            if not '_dmet_chk.h5' in chk_fname:
                chk_fname = chk_fname + '_dmet_chk.h5'
            if not os.path.isfile(chk_fname):
                return False
        except:
            return False

        self.log.info(f'load chk file {chk_fname}')
        with h5py.File(chk_fname, 'r') as fh5:
            dm_check = np.allclose(self.dm, fh5['dm'][:], atol=1e-5)
            imp_idx_check = compare_imp_idx(self.imp_idx, fh5['imp_idx'][:])
            threshold_check = self.threshold == fh5['threshold'][()]
            if 'bath_norb' in fh5:
                bath_norb_check = str(self.bath_norb) == str(fh5['bath_norb'][()])
            else:
                bath_norb_check = self.bath_norb is None
            if dm_check & imp_idx_check & threshold_check & bath_norb_check:
                self.fo_orb = fh5['fo_orb'][:]
                self.fv_orb = fh5['fv_orb'][:]
                self.es_orb = fh5['es_orb'][:]
                self.es_occ = fh5['es_occ'][:]
                self.es_int1e = fh5['es_int1e'][:]
                self.es_int2e = fh5['es_int2e'][:]
                self.es_dm = fh5['es_dm'][:]

                self.nfo = np.shape(self.fo_orb)[1]
                self.nfv = np.shape(self.fv_orb)[1]
                self.nes = np.shape(self.es_orb)[1]
                return True
            else:
                self.log.info(f'density matrix check {dm_check}')
                self.log.info(f'impurity index check {imp_idx_check}')
                self.log.info(f'threshold check {threshold_check}')
                self.log.info(f'bath_norb check {bath_norb_check}')
                self.log.info(f'build dmet subspace with imp idx {self.imp_idx} threshold {self.threshold}')
                return False
    
    def save_chk(self, chk_fname):
        with h5py.File(chk_fname, 'w') as fh5:
            fh5['dm'] = self.dm
            fh5['imp_idx'] = self.imp_idx
            fh5['threshold'] = self.threshold
            fh5['bath_norb'] = str(self.bath_norb)

            fh5['fo_orb'] = self.fo_orb
            fh5['fv_orb'] = self.fv_orb
            fh5['es_orb'] = self.es_orb
            fh5['es_occ'] = self.es_occ
            fh5['es_int1e'] = self.es_int1e
            fh5['es_int2e'] = self.es_int2e
            fh5['es_dm'] = self.es_dm
        return 
    
    def lowdin_orth(self, restore_imp = False, iaopao = False):
        # lowdin orthonormalize
        caolo, cloao = lowdin_orth(self.mol)
        lo2ao = cloao
        if restore_imp:
            imp_idx = self.imp_idx
            mask_env = np.ones(len(caolo), dtype=bool)
            mask_env[imp_idx] = False

            Q1 = cloao[:, imp_idx]
            Q1, _ = np.linalg.qr(Q1) # orthonormalize
            P = np.eye(*cloao.shape) - Q1 @ Q1.T.conj()
            B = P @ cloao[:, mask_env]
            from scipy.linalg import svd
            U, S, Vh = svd(B, full_matrices=False)

            Q = np.zeros(cloao.shape)
            Q[:, imp_idx] = Q1
            Q[:, mask_env] = U[:, 0: cloao.shape[0] - len(imp_idx)]
            cloao = Q.T.conj() @ cloao
            caolo = caolo @ Q
        if iaopao:
            caolo = iao_helper.localize_iao(self.mol, self.mf_or_cas, lo2ao)
            cloao = np.linalg.inv(caolo)
        ldm = reduce(lib.dot, (cloao, self.dm, cloao.conj().T))
        return ldm, caolo, cloao
        
    def build(self, restore_imp = False, iaopao = False, chk_fname_load='', save_chk=True, xc = None, mp2method='full'):
        self.dump_flags()
        dm = mf_or_cas_make_rdm1s(self.mf_or_cas)
        if dm.ndim == 3: # ROHF density matrix have dimension (2, nao, nao)
            self.dm_pair = dm            
            self.dm = dm[0] + dm[1]
            open_shell = True
        else:
            self.dm = dm
            open_shell = False

        loaded = self.load_chk(chk_fname_load)
        
        if not loaded:
            ldm, caolo, cloao = self.lowdin_orth(restore_imp, iaopao)

            bath_norb = self.bath_norb
            if isinstance(bath_norb, str):
                if bath_norb.lower() in ('per_bond', 'perbond', 'one_per_bond'):
                    bath_norb = count_imp_env_bonds(self.mol, self.imp_idx)
                    self.log.info(f'one bath orbital per bond: {bath_norb} impurity-environment bond(s) detected')
                else:
                    raise ValueError(f'unknown bath_norb string: {bath_norb!r}; use an int or "per_bond"')

            cloes, nimp, nbath, nfo, nfv, self.es_occ = build_embeded_subspace(
                ldm, self.imp_idx, thres=self.threshold, es_natorb=self.es_natorb,
                bath_norb=bath_norb, bath_core_cutoff=self.bath_core_cutoff)
            
            self.caolo = caolo
            self.cloao = cloao
            self.lo_cloes = cloes
            self.open_shell = open_shell

            caoes = lib.dot(caolo, cloes)

            self.fo_orb = caoes[:, nimp+nbath: nimp+nbath+nfo]
            self.fv_orb = caoes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
            self.es_orb = caoes[:, :nimp+nbath]
        
            self.nfo = nfo
            self.nfv = nfv
            self.nes = nimp + nbath
            self.log.info(f"****Restore imp: {restore_imp}")
            self.log.info(f"****IAOPAO: {iaopao}")
            self.log.info(f'number of impurity orbitals = {nimp}')
            self.log.info(f'number of bath orbitals = {nbath}')
            self.log.info(f'number of embedded cluster orbitals = {nimp+nbath}')
            self.log.info(f'number of frozen occupied orbitals = {nfo}')
            self.log.info(f'number of frozen virtual orbitals = {nfv}')
            self.log.info(f'number of frozen orbitals = {nfo+nfv}')
            self.log.info(f'percentage of embedded cluster orbitals = {((nimp+nbath)/self.mol.nao)*100:.2f}%%')
            self.log.info(f'percentage of frozen orbitals = {((nfo+nfv)/self.mol.nao)*100:.2f}%%')

            self.es_int1e = self.make_es_int1e()
            self.es_int2e = self.make_es_int2e()

            self.es_dm = self.make_es_dm(open_shell, cloes[:, :nimp+nbath], cloao, dm)

            if self.bath_option is not None:
                self.log.info('')
                if self.es_natorb:
                    raise RuntimeError('es_natorb must be turned off when using extra bath_option')
                lo2core = cloes[:, nimp+nbath: nimp+nbath+nfo]
                lo2vir = cloes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
                if isinstance(self.bath_option, dict):
                    if len(self.bath_option.keys()) == 1:
                        if 'MP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.info('ROMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['MP2'])
                            else:
                                self.log.info('RMP2 bath expansion in used by default')
                                if mp2method == 'sos':
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath_sos(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                            lo2core, lo2vir, eta=self.bath_option['MP2'])
                                else:
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                        lo2core, lo2vir, eta=self.bath_option['MP2'])
                        elif 'RMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.info('ROMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['RMP2'])
                            else:
                                if mp2method == 'sos':
                                    # Direct-only (opposite-spin, SOS) MP2 bath (see comment in the 'MP2' branch)
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath_sos(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                             lo2core, lo2vir, eta=self.bath_option['RMP2'])
                                else:
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                         lo2core, lo2vir, eta=self.bath_option['RMP2'])
                        elif 'ROMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                if mp2method == 'sos':
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath_sos(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                             lo2core, lo2vir, readmp2 = self.readmp2, eta=self.bath_option['ROMP2'])
                                else:
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                          lo2core, lo2vir, readmp2 = self.readmp2, eta=self.bath_option['ROMP2'])
                            else:
                                self.log.info('ROMP2 bath expansion is degraded to RMP2 for closed-shell systems')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['ROMP2'])
                        elif 'UMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.warn('UMP2 bath expansion is less preferred than ROMP2 for ROHF, the results must be checked carefully!')
                                if mp2method == 'sos':
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_UMP2_bath_sos(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                             lo2core, lo2vir, eta=self.bath_option['UMP2'])
                                else:
                                    lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_UMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['UMP2'])
                            else:
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_UMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['UMP2'])
                        else:
                            raise NotImplementedError('Currently only MP2, RMP2, ROMP2 and UMP2 are supported')
                    else:
                        raise NotImplementedError('Only one key should be in bath_option')
                else:
                    raise NotImplementedError('The bath_option should be a dictionary')
                
                lo2eo = np.hstack([cloes[:, :nimp+nbath], lo2MP2_bath])
                self.es_orb = lib.dot(caolo, lo2eo)
                self.fo_orb = lib.dot(caolo, lo2MP2_core)
                self.fv_orb = lib.dot(caolo, lo2MP2_vir)

                nbath += lo2MP2_bath.shape[-1]
                nfo = self.fo_orb.shape[-1]
                nfv = self.fv_orb.shape[-1]
                self.nfo = nfo
                self.nfv = nfv
                self.nes = nimp + nbath
                self.log.info(f'number of impurity orbitals = {nimp}')
                self.log.info(f'number of bath orbitals = {nbath}')
                self.log.info(f'number of embedded cluster orbitals = {nimp+nbath}')
                self.log.info(f'number of frozen occupied orbitals = {nfo}')
                self.log.info(f'number of frozen virtual orbitals = {nfv}')
                self.log.info(f'number of frozen orbitals = {nfo+nfv}')
                self.log.info(f'percentage of embedded cluster orbitals = {((nimp+nbath)/self.mol.nao)*100:.2f}%%')
                self.log.info(f'percentage of frozen orbitals = {((nfo+nfv)/self.mol.nao)*100:.2f}%%')

                self.es_int1e = self.make_es_int1e()
                self.es_int2e = self.make_es_int2e()
                self.es_dm = self.make_es_dm(open_shell, lo2eo, cloao, dm)
            else:
                pass

        self.es_mf = self.ROHF()
        #self.fo_ene()
        if xc is not None:
            self.calc_fo_ene_dft(xc)
        else:
            self.calc_fo_ene()
        self.log.info('')
        self.log.info(f'energy from frozen occupied orbitals = {self.fo_ene}')
        self.log.info(f'deviation from DMET exact condition = {self.es_mf.e_tot+self.fo_ene-self.mf_or_cas.e_tot}')
        self.log.info(f'energy from embedded cluster with low level calculation = {self.es_mf.e_tot}')
        self.log.info(f'total energy with low level calculation = {self.mf_or_cas.e_tot}')

        if save_chk:
            chk_fname_save = self.title + '_dmet_chk.h5'
            self.save_chk(chk_fname_save)
        return self.es_mf
    
    def make_es_dm(self, open_shell, lo2es, cloao, dm):
        if open_shell:
            if self.es_natorb:
                es_dm = np.zeros((2, self.nes, self.nes))
                es_dm[0] = np.diag(np.int32(self.es_occ>1-1e-3))
                es_dm[1] = np.diag(np.int32(self.es_occ>2-1e-3))
            else:
                es_dm = np.zeros((2, self.nes, self.nes))
                dma, dmb = dm
                ldma = reduce(lib.dot, (cloao, dma, cloao.conj().T))
                ldmb = reduce(lib.dot, (cloao, dmb, cloao.conj().T))
                es_dm[0] = reduce(lib.dot, (lo2es.conj().T, ldma, lo2es))
                es_dm[1] = reduce(lib.dot, (lo2es.conj().T, ldmb, lo2es))
        else:
            if self.es_natorb:
                es_dm = np.zeros((self.nes, self.nes))
                es_dm = np.diag(np.int32(self.es_occ>1-1e-3))
            else:
                es_dm = np.zeros((self.nes, self.nes))
                ldm = reduce(lib.dot, (cloao, dm, cloao.conj().T))
                es_dm = reduce(lib.dot, (lo2es.conj().T, ldm, lo2es))
        return es_dm
    
    def ROHF(self):
        mol = gto.M()
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelectron = self.mf_or_cas.mol.nelectron - 2*self.nfo
        mol.spin = self.mol.spin

        if mol.spin != 0:
            es_mf = scf.ROHF(mol).x2c()
        else:
            es_mf = scf.RHF(mol).x2c()
        es_mf.max_memory = self.max_mem
        es_mf.mo_energy = np.zeros((self.nes))
        es_mf.conv_tol = self.mf_or_cas.conv_tol
        es_mf.conv_tol_grad = getattr(self.mf_or_cas, 'conv_tol_grad', None)

        es_ovlp = reduce(lib.dot, (self.es_orb.conj().T, self.mol.intor_symmetric('int1e_ovlp'), self.es_orb))
        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: es_ovlp
        es_mf._eri = self.es_int2e
        es_mf.mo_coeff = np.eye(self.nes)

        # assume we only perfrom ROHF-in-ROHF embedding
        es_mf.max_cycle = self.mf_or_cas.max_cycle
        # assert lib.einsum('ijj->', es_dm) == mol.nelectron
        es_mf.level_shift = self.mf_or_cas.level_shift
        es_mf.conv_check = False
        es_mf.kernel(self.es_dm)
        self.es_occ = es_mf.mo_occ
        return es_mf
    
    def avas(self, aolabels, *args, **kwargs):
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ) # make 2/0 occupation to be int
        ncas, nelec, mo = myavas.avas(total_mf, aolabels, ncore=self.nfo, nunocc = self.nfv, canonicalize=False, *args, **kwargs) # canonicalize should be set to False, since it require orbital energy

        es_mo = reduce(lib.dot, (self.es_orb.T.conj(), self.mol.intor_symmetric('int1e_ovlp'), mo[:, self.nfo: self.nfo+self.nes]))
        return ncas, nelec, es_mo 
    def avas_export(self, aolabels, *args, **kwargs):
        """AVAS with orbital analysis + molden export for visualization.

        Same as avas(), but additionally:
        - Prints mapping: AVAS active index -> original ROHF MO index
        - Exports _avas_full.molden (all MOs in AVAS order, correct occ)
        - Exports _avas_active.molden (only active MOs, correct occ)

        Open _avas_full.molden to see active orbitals at their real MO indices.
        """
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ)

        threshold = kwargs.pop('threshold', 0.5)
        avas_obj = myavas.AVAS(total_mf, aolabels, ncore=self.nfo, nunocc=self.nfv,
                               threshold=threshold, canonicalize=False, *args, **kwargs)
        ncas, nelec, mo = avas_obj.kernel()

        # AVAS mo ordering: [frozen_core | mocore | mocas | movir | frozen_virt]
        ncore = self.nfo
        nocc_total = int(np.sum(total_mf.mo_occ > 0))
        ncas_occ = int(np.sum(avas_obj.occ_weights >= threshold))
        n_inactive_occ = nocc_total - ncore - ncas_occ
        active_start = ncore + n_inactive_occ
        active_end = active_start + ncas
        avas_active = mo[:, active_start:active_end]

        self.log.info("AVAS: ncore=%d, nocc_total=%d, ncas_occ=%d",
                      ncore, nocc_total, ncas_occ)
        self.log.info("AVAS: n_inactive_occ(mocore)=%d, ncas=%d, nelec=%d",
                      n_inactive_occ, ncas, nelec)

        # --- overlap with original ROHF MOs ---
        ovlp_ao = self.mol.intor_symmetric('int1e_ovlp')
        rohf_mo = self.mf_or_cas.mo_coeff
        rohf_mo_occ = self.mf_or_cas.mo_occ
        if isinstance(rohf_mo, (list, tuple)):
            rohf_mo = rohf_mo[0]
            rohf_mo_occ = rohf_mo_occ[0]
        overlap_rohf = reduce(lib.dot, (avas_active.T.conj(), ovlp_ao, rohf_mo))

        # AVAS eigenvalues per active orbital
        occ_w = avas_obj.occ_weights[avas_obj.occ_weights >= threshold]
        vir_w = avas_obj.vir_weights[avas_obj.vir_weights >= threshold]
        active_eigs = np.hstack([occ_w, vir_w])

        rohf_alpha_occ = np.minimum(rohf_mo_occ, 1.0)
        rohf_beta_occ = rohf_mo_occ - rohf_alpha_occ

        # --- mapping table ---
        self.log.info("AVAS: --- Active -> ROHF canonical MO mapping ---")
        self.log.info("AVAS: %4s  %8s  %8s  %10s  %10s  %6s  %6s",
                      "idx", "ROHF_MO", "overlap", "eig", "pop(a/b)", "occ_a", "occ_b")
        occ_active = np.zeros(ncas)
        for i in range(ncas):
            overlaps = np.abs(overlap_rohf[i])
            sorted_idx = np.argsort(overlaps)[::-1]
            best = sorted_idx[0]
            alpha_pop = np.dot(overlaps**2, rohf_alpha_occ)
            beta_pop = np.dot(overlaps**2, rohf_beta_occ)
            occ_active[i] = np.dot(overlaps**2, rohf_mo_occ)
            self.log.info("AVAS: %4d  %8d  %8.4f  %10.4f  %6.2f/%-6.2f %6s  %6s",
                          i, best, overlaps[best], active_eigs[i],
                          alpha_pop, beta_pop,
                          "occ" if rohf_mo_occ[best] >= 2 else ("1" if rohf_mo_occ[best] >= 1 else "0"),
                          "occ" if rohf_mo_occ[best] >= 2 else ("1" if rohf_mo_occ[best] >= 1 else "0"))
            others = [f"MO#{j}({overlaps[j]:.3f})" for j in sorted_idx[1:4] if overlaps[j] > 0.1]
            if others:
                self.log.info("AVAS:         also: %s", ", ".join(others))
        occ_active = np.round(occ_active)

        # --- occupations for FULL AVAS-reordered MO matrix ---
        s_full = reduce(lib.dot, (mo.T.conj(), ovlp_ao, total_mf.mo_coeff))
        nmo = mo.shape[1]
        mo_occ_full = np.zeros(nmo)
        for i in range(nmo):
            overlaps_i = np.abs(s_full[i])
            mo_occ_full[i] = np.dot(overlaps_i**2, total_mf.mo_occ)
        mo_occ_full = np.round(mo_occ_full)

        # --- export molden ---
        from pyscf import tools
        tools.molden.from_mo(self.mol, self.title+'_avas_full.molden', mo, occ=mo_occ_full)
        tools.molden.from_mo(self.mol, self.title+'_avas_active.molden', avas_active, occ=occ_active)
        self.log.info("AVAS: exported %s_avas_full.molden  (all %d MOs, active=MO#%d-%d 1-indexed, occ=%s)",
                      self.title, nmo, active_start+1, active_end, occ_active.tolist())
        self.log.info("AVAS: exported %s_avas_active.molden (%d active MOs only)",
                      self.title, ncas)

        es_mo = reduce(lib.dot, (self.es_orb.T.conj(), ovlp_ao,
                                  mo[:, self.nfo: self.nfo+self.nes]))
        return ncas, nelec, es_mo

    def avas_export_orj(self, aolabels, *args, **kwargs):
        """AVAS with orbital analysis + molden export for visualization.

        Same as avas(), but additionally:
        - Prints mapping: AVAS active index -> original ROHF MO index
        - Exports _avas_full.molden (all MOs in AVAS order, correct occ)
        - Exports _avas_active.molden (only active MOs, correct occ)

        Open _avas_full.molden to see active orbitals at their real MO indices.
        """
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ)

        threshold = kwargs.pop('threshold', 0.5)
        avas_obj = myavas.AVAS(total_mf, aolabels, ncore=self.nfo, nunocc=self.nfv,
                               threshold=threshold, canonicalize=False, *args, **kwargs)
        ncas, nelec, mo = avas_obj.kernel()

        # AVAS mo ordering: [frozen_core | mocore | mocas | movir | frozen_virt]
        ncore = self.nfo
        nocc_total = int(np.sum(total_mf.mo_occ > 0))
        ncas_occ = int(np.sum(avas_obj.occ_weights >= threshold))
        n_inactive_occ = nocc_total - ncore - ncas_occ
        active_start = ncore + n_inactive_occ
        active_end = active_start + ncas
        avas_active = mo[:, active_start:active_end]

        self.log.info("AVAS: ncore=%d, nocc_total=%d, ncas_occ=%d",
                      ncore, nocc_total, ncas_occ)
        self.log.info("AVAS: n_inactive_occ(mocore)=%d, ncas=%d, nelec=%d",
                      n_inactive_occ, ncas, nelec)

        # --- overlap with original ROHF MOs ---
        ovlp_ao = self.mol.intor_symmetric('int1e_ovlp')
        rohf_mo = self.mf_or_cas.mo_coeff
        rohf_mo_occ = self.mf_or_cas.mo_occ
        if isinstance(rohf_mo, (list, tuple)):
            rohf_mo = rohf_mo[0]
            rohf_mo_occ = rohf_mo_occ[0]
        overlap_rohf = reduce(lib.dot, (avas_active.T.conj(), ovlp_ao, rohf_mo))

        # AVAS eigenvalues per active orbital
        occ_w = avas_obj.occ_weights[avas_obj.occ_weights >= threshold]
        vir_w = avas_obj.vir_weights[avas_obj.vir_weights >= threshold]
        active_eigs = np.hstack([occ_w, vir_w])

        rohf_alpha_occ = np.minimum(rohf_mo_occ, 1.0)
        rohf_beta_occ = rohf_mo_occ - rohf_alpha_occ

        # --- mapping table ---
        self.log.info("AVAS: --- Active -> ROHF canonical MO mapping ---")
        self.log.info("AVAS: %4s  %8s  %8s  %10s  %10s  %6s  %6s",
                      "idx", "ROHF_MO", "overlap", "eig", "pop(a/b)", "occ_a", "occ_b")
        occ_active = np.zeros(ncas)
        for i in range(ncas):
            overlaps = np.abs(overlap_rohf[i])
            sorted_idx = np.argsort(overlaps)[::-1]
            best = sorted_idx[0]
            alpha_pop = np.dot(overlaps**2, rohf_alpha_occ)
            beta_pop = np.dot(overlaps**2, rohf_beta_occ)
            occ_active[i] = np.dot(overlaps**2, rohf_mo_occ)
            self.log.info("AVAS: %4d  %8d  %8.4f  %10.4f  %6.2f/%-6.2f %6s  %6s",
                          i, best, overlaps[best], active_eigs[i],
                          alpha_pop, beta_pop,
                          "occ" if rohf_mo_occ[best] >= 2 else ("1" if rohf_mo_occ[best] >= 1 else "0"),
                          "occ" if rohf_mo_occ[best] >= 2 else ("1" if rohf_mo_occ[best] >= 1 else "0"))
            others = [f"MO#{j}({overlaps[j]:.3f})" for j in sorted_idx[1:4] if overlaps[j] > 0.1]
            if others:
                self.log.info("AVAS:         also: %s", ", ".join(others))
        occ_active = np.round(occ_active)

        # --- occupations for FULL AVAS-reordered MO matrix ---
        s_full = reduce(lib.dot, (mo.T.conj(), ovlp_ao, total_mf.mo_coeff))
        nmo = mo.shape[1]
        mo_occ_full = np.zeros(nmo)
        for i in range(nmo):
            overlaps_i = np.abs(s_full[i])
            mo_occ_full[i] = np.dot(overlaps_i**2, total_mf.mo_occ)
        mo_occ_full = np.round(mo_occ_full)

        # --- export molden ---
        from pyscf import tools
        tools.molden.from_mo(self.mol, self.title+'_avas_full.molden', mo, occ=mo_occ_full)
        tools.molden.from_mo(self.mol, self.title+'_avas_active.molden', avas_active, occ=occ_active)
        self.log.info("AVAS: exported %s_avas_full.molden  (all %d MOs, active=MO#%d-%d 1-indexed, occ=%s)",
                      self.title, nmo, active_start+1, active_end, occ_active.tolist())
        self.log.info("AVAS: exported %s_avas_active.molden (%d active MOs only)",
                      self.title, ncas)

        es_mo = reduce(lib.dot, (self.es_orb.T.conj(), ovlp_ao,
                                  mo[:, self.nfo: self.nfo+self.nes]))
        return ncas, nelec, es_mo, mo

    def total_mf(self):
        total_mf = scf.rohf.ROHF(self.mol).x2c()
        total_mf.mo_coeff = np.hstack((self.fo_orb, lib.dot(self.es_orb, self.es_mf.mo_coeff), self.fv_orb))
        total_mf.mo_occ = np.hstack((2*np.ones(self.nfo), self.es_occ, np.zeros(self.nfv)))
        return total_mf
    
    def total_cas(self, es_cas):
        from embed_sim import sacasscf_mixer
        total_cas = sacasscf_mixer.sacasscf_mixer(self.mf_or_cas, es_cas.ncas, es_cas.nelecas, statelis=sacasscf_mixer.read_statelis(es_cas), weights=es_cas.weights)
        total_cas.fcisolver = es_cas.fcisolver
        total_cas.ci = es_cas.ci
        total_cas.mo_coeff = np.hstack((self.fo_orb, self.es_orb @ es_cas.mo_coeff, self.fv_orb))
        return total_cas
    
    def calc_fo_ene(self, e_nuc = True):
        # energy of frozen occupied orbitals and nuclear-nuclear repulsion
        dm_fo = self.fo_orb @ self.fo_orb.T.conj()*2

        h1e = self.mf_or_cas.get_hcore()
        if isinstance(dm_fo, np.ndarray) and dm_fo.ndim == 2:
            dm_fo = np.array((dm_fo*.5, dm_fo*.5))
        # get_veff in casci and rohf differ by a factor 2: rohf.get_veff = casci.get_veff * 2
        # we manually build vhf
        vj, vk = self.mf_or_cas.get_jk(self.mol, dm_fo)
        vhf = vj[0] + vj[1] - vk
        
        if h1e[0].ndim < dm_fo[0].ndim:  # get [0] because h1e and dm may not be ndarrays
            h1e = (h1e, h1e)
        e1 = lib.einsum('ij,ji->', h1e[0], dm_fo[0])
        e1+= lib.einsum('ij,ji->', h1e[1], dm_fo[1])
        e_coul =(lib.einsum('ij,ji->', vhf[0], dm_fo[0]) +
                lib.einsum('ij,ji->', vhf[1], dm_fo[1])) * .5
        e_elec = (e1 + e_coul).real
        fo_ene = e_elec
        if e_nuc:
            e_nuc = self.mf_or_cas.energy_nuc()
            fo_ene += e_nuc
        self.fo_ene = fo_ene
        return fo_ene
    

    def calc_fo_ene_dft(self, xc, e_nuc = True):
        # energy of frozen occupied orbitals and nuclear-nuclear repulsion 
        # with the socalled DFT one shot solution for core electrons
        if xc is None:
            raise ValueError('xc functional must be provided for DFT one shot calculation')
        else:
            self.log.info(f'Calculating frozen occupied orbital energy with DFT functional {xc} ...')
        dm_fo = self.fo_orb @ self.fo_orb.T.conj()*2
        num_fo = self.fo_orb.shape[1]
        h1e = self.mf_or_cas.get_hcore()
        if isinstance(dm_fo, np.ndarray) and dm_fo.ndim == 2:
            dm_fo = np.array((dm_fo*.5, dm_fo*.5))
        # get_veff in casci and rohf differ by a factor 2: rohf.get_veff = casci.get_veff * 2
        # we manually build vhf
        vj, vk = self.mf_or_cas.get_jk(self.mol, dm_fo)
        vhf = vj[0] + vj[1] - vk
        
        if h1e[0].ndim < dm_fo[0].ndim:  # get [0] because h1e and dm may not be ndarrays
            h1e = (h1e, h1e)
        e1 = lib.einsum('ij,ji->', h1e[0], dm_fo[0])
        e1+= lib.einsum('ij,ji->', h1e[1], dm_fo[1])
        e_coul =(lib.einsum('ij,ji->', vhf[0], dm_fo[0]) +
                lib.einsum('ij,ji->', vhf[1], dm_fo[1])) * .5
        e_elec = (e1 + e_coul).real
        fo_ene = e_elec
        if e_nuc:
            fo_ene += self.mf_or_cas.energy_nuc()
        self.fo_ene = fo_ene


        # --- DFT core energy (non-self-consistent on HF density) ---
        # Use exactly the same Coulomb and exchange formulas as the HF energy above,
        # but replace the exchange part with DFT XC.
        #
        # HF core energy (above):
        #   e_coul = 0.5 * [Tr((vj[0]+vj[1]-vk)*dm_fo[0]) + Tr((vj[0]+vj[1]-vk)*dm_fo[1])]
        #          = 0.5 * Tr((J - K)*D)   [since dm_fo[0]=dm_fo[1]=D/2]
        #
        # DFT core energy:
        #   e_coul_dft = 0.5 * Tr(J*D)      (Coulomb only, no K)
        #   e_xc       = from grid integration
        #   + hybrid correction if needed
        
        dm_fo_tot = dm_fo[0] + dm_fo[1]
        
        # One-electron part: same as HF
        if getattr(h1e[0], 'ndim', 0) < getattr(dm_fo[0], 'ndim', 2): 
            h1e = (h1e, h1e)
        dft_e1 = np.einsum('ij,ji->', h1e[0], dm_fo[0]) + np.einsum('ij,ji->', h1e[1], dm_fo[1])
        
        # Coulomb-only part: 0.5 * Tr(J * D) 
        # J from total density: vj[0] + vj[1] = J(D/2) + J(D/2) = J(D)
        # dm_fo[0] + dm_fo[1] = D
        # So: 0.5 * Tr(J(D) * D)
        vj_tot = vj[0] + vj[1]
        dft_e_coul = 0.5 * np.einsum('ij,ji->', vj_tot, dm_fo_tot).real
        
        # XC from grid (for pure functionals this is the full XC; 
        # for hybrids this is the LDA/GGA part only, HF exchange added below)
        from pyscf import dft
        ni = dft.numint.NumInt()
        grids = dft.gen_grid.Grids(self.mf_or_cas.mol)
        grids.level = 3
        grids.build()
        nelec, exc, vxc = ni.nr_rks(self.mf_or_cas.mol, grids, xc, dm_fo_tot, hermi = 1)
        
        # Add exact exchange for hybrid functionals (including pure 'hf')
        # ni.libxc.is_hybrid_xc('hf') returns False, but hybrid_coeff('hf') returns 1.0.
        # Use hybrid_coeff to decide: if it returns > 0, there is exact exchange.
        try:
            hybrid_coeff = ni.libxc.hybrid_coeff(xc, spin=self.mol.spin)
        except Exception:
            hybrid_coeff = 0.0
        
        if hybrid_coeff > 0:
            # E_x_HF = -0.25 * omega * Tr(K(D) * D)
            # Recompute K from total density matrix for correct scaling
            _, vk_tot = self.mf_or_cas.get_jk(self.mol, dm_fo_tot, with_j=False, with_k=True)
            hx_added = hybrid_coeff * (-0.25 * np.dot(vk_tot.ravel(), dm_fo_tot.ravel()))
            exc += hx_added
            print(f"  hybrid coeff = {hybrid_coeff:.4f}, HF exchange added = {hx_added:.6f}")
        
        edft = (exc + dft_e_coul + dft_e1).real
        if e_nuc:
            edft += self.mf_or_cas.energy_nuc()
        self.log.info(f"DFT energy of frozen occupied orbitals = {edft}")
        self.log.info(f"HF energy of frozen occupied orbitals = {fo_ene}")
        self.fo_ene = edft

        return edft
    # 20260129 new CCSD(T)
    def ccsdt_solver(self, with_t=True):
        " run ccsd(t) as the solver for dmet subspace "
        from pyscf import cc
        
        if self.es_mf is None:
            raise RuntimeError('embedded subspace is not built yet, please run build() first')
        
        self.log.info('Running CCSD solver for embedded cluster ...')
        self.log.info('=' * 60)
        
        mycc = cc.CCSD(self.es_mf)
        mycc.verbose = self.verbose
        mycc.max_memory = self.max_mem
        
        
        mycc.kernel()
        
        
        global_mf_energy = self.mf_or_cas.e_tot
        frag_corr_energy = mycc.e_corr
        
        e_tot_direct = mycc.e_tot + self.fo_ene
        e_tot_correction_based = global_mf_energy + frag_corr_energy
        
        self.log.info(f'Global MF Energy      = %.12f', global_mf_energy)
        self.log.info(f'Frozen Env Energy     = %.12f', self.fo_ene)
        self.log.info(f'Emb CCSD Corr Energy  = %.12f', mycc.e_corr)
        self.log.info(f'Total Energy (Direct) = %.12f (Sensitive to bath size)', e_tot_direct)
        self.log.info(f'Total Energy (Corr)   = %.12f (Recommended for PES)', e_tot_correction_based)

        # Check if electrons are leaking from the fragment
        try:
            rdm1 = mycc.make_rdm1()
            nimp = len(self.imp_idx)
            nel_frag = np.trace(rdm1[:nimp, :nimp])
            self.log.info(f'Fragment Electron Number (Impurity Trace) = %.4f', nel_frag)
            
        except Exception as e:
            self.log.warn(f"Could not calculate properties: {e}")

        
        final_e_tot = e_tot_direct 

        if with_t:
            et = mycc.ccsd_t()
            e_tot_ccsd_t = final_e_tot + et
            e_tot_corr_t = e_tot_correction_based + et
            
            self.log.info('CCSD(T) correlation energy = %.12f', mycc.e_corr + et)
            self.log.info('Total CCSD(T) Energy (Direct) = %.12f', e_tot_ccsd_t)
            self.log.info('Total CCSD(T) Energy (Corr)   = %.12f', e_tot_corr_t)
            return mycc, e_tot_ccsd_t, et + mycc.e_corr
            
        return final_e_tot
    def mp2_solver(self):
        from pyscf import mp
        if self.es_mf is None:
            raise RuntimeError('embedded subspace is not built yet, please run build() first')
        # check spin 
        

        self.log.info('Running MP2 for EO space')
        self.log.info('='*60)
        if self.es_mf.mol.spin != 0:
            self.log.info('ROMP2 is used for ROHF reference')
            mymp2 = mp.UMP2(self.es_mf)
        else:
            self.log.info('RMP2 is used ')
            mymp2 = mp.MP2(self.es_mf)
        mymp2.verbose = self.verbose
        mymp2.max_memory = self.max_mem
        mymp2.kernel()
        e_tot_mp2 = mymp2.e_tot + self.fo_ene
        self.log.info('MP2 correlation energy = %.12f', mymp2.e_corr)
        self.log.info('Total MP2 energy = %.12f', e_tot_mp2)
        return e_tot_mp2, mymp2.e_corr

    def nuc_grad_method(self, **kwargs):
        """Analytic nuclear gradient of the one-shot DMET energy.

        Stage 1 supports a closed-shell RHF reference with the embedded mean
        field as cluster solver (HF-in-HF); see embed_sim/grad/ssdmet.py.
        """
        from embed_sim.grad.ssdmet import SSDMETGradients
        return SSDMETGradients(self, **kwargs)

    def density_fit(self, with_df=None):
        from embed_sim.df import DFSSDMET
        if with_df is None:
            if not getattr(self.mf_or_cas, 'with_df', False):
                raise NotImplementedError
            else:
                with_df = self.mf_or_cas.with_df
        return DFSSDMET(self.mf_or_cas, self.title, imp_idx=self.imp_idx, threshold=self.threshold,
                        with_df=with_df, es_natorb=self.es_natorb, bath_option=self.bath_option,
                        bath_norb=self.bath_norb, bath_core_cutoff=self.bath_core_cutoff, verbose=self.verbose)
