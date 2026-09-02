import numpy as np
from functools import reduce
from scipy.linalg import block_diag
import h5py

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, ao2mo

from embed_sim.BNO_bath import get_RMP2_bath, get_UMP2_bath, get_ROMP2_bath

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
    
'''
def build_embeded_subspace(ldm, imp_idx, lo_meth='lowdin', thres=1e-12, es_natorb=True):
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
    # try to change to the SVD for the coupling block of imp and env.
    #U, sigma, V= np.linalg.svd(ldm_env_imp, full_matrices = False)

    nimp = len(imp_idx)
    nfv = np.sum(occ_env <  thres) # frozen virtual 
    nbath = np.sum((occ_env >= thres) & (occ_env <= 2-thres)) # bath orbital
    nfo = np.sum(occ_env > 2-thres) # frozen occupied

    # defined w.r.t enviroment orbital index
    fv_idx = np.nonzero(occ_env <  thres)[0]
    bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 2-thres))[0]
    fo_idx = np.nonzero(occ_env > 2-thres)[0]

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
'''
def build_embeded_subspace(ldm, imp_idx, thres=1e-12, es_natorb=True, use_svd=False):
    """
    Returns C(AO->AS), entropy loss, and orbital composition
    """
    env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]
    nenv = len(env_idx)
    nimp = len(imp_idx)

    ldm_imp = ldm[imp_idx,:][:,imp_idx]
    ldm_env = ldm[env_idx,:][:,env_idx]
    ldm_imp_env = ldm[imp_idx,:][:,env_idx]
    ldm_env_imp = ldm[env_idx,:][:,imp_idx]

    if use_svd:
        # U (nenv, nimp)，when full_matrices=False
        U, S, Vh = np.linalg.svd(ldm_env_imp, full_matrices=False)
        
        # find bath orbitals
        bath_mask = S > thres
        orb_bath = U[:, bath_mask] 
        nbath = orb_bath.shape[1]
        
        # projector operator P = I - |Bath><Bath|
        P = np.eye(nenv) - orb_bath @ orb_bath.T.conj()
        
        # 4. 找到补空间的基 (P 的非零特征向量)
        # P 是幂等矩阵，特征值只有 0 和 1。特征值~1 的对应的特征向量就是 Frozen 空间的基
        w_proj, v_proj = np.linalg.eigh(P)
        frozen_basis_mask = w_proj > 0.5
        frozen_basis = v_proj[:, frozen_basis_mask]
        
        # 5. 在补空间内对角化密度矩阵，区分 Frozen Occ 和 Frozen Vir
        # 将 ldm_env 投影到 Frozen Basis 中: D_sub = C_frz^T * D_env * C_frz
        ldm_frozen_sub = reduce(np.dot, (frozen_basis.T.conj(), ldm_env, frozen_basis))
        w_sub, v_sub = np.linalg.eigh(ldm_frozen_sub)
        
        # 恢复到 LO 基下的表达
        orb_frozen_LO = np.dot(frozen_basis, v_sub)
        
        # 6. 分类 Frozen Occ 和 Frozen Vir
        fo_mask = w_sub > 2 - thres
        fv_mask = w_sub < thres
        
        orb_fo = orb_frozen_LO[:, fo_mask]
        orb_fv = orb_frozen_LO[:, fv_mask]
        
        # 7. 拼装最终的 orb_env (顺序: Bath, FO, FV)
        orb_env = np.hstack([orb_bath, orb_fo, orb_fv])
        
        nfo = orb_fo.shape[1]
        nfv = orb_fv.shape[1]
        
        occ_env = np.zeros(nenv) 
        
    else:
       # 对环境做对角化
        occ_env, orb_env = np.linalg.eigh(ldm_env)
        nfv = np.sum(occ_env <  thres) # frozen virtual 
        nbath = np.sum((occ_env >= thres) & (occ_env <= 2-thres)) # bath orbital
        nfo = np.sum(occ_env > 2-thres) # frozen occupied

        # defined w.r.t enviroment orbital index
        fv_idx = np.nonzero(occ_env <  thres)[0]
        bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 2-thres))[0]
        fo_idx = np.nonzero(occ_env > 2-thres)[0]

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
        
        # es_dm_temp = cloes[:, :nimp+nbath].T @ ldm @ cloes[:, :nimp+nbath]
        # es_occ = np.diag(es_dm_temp) 
    
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
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, es_natorb=True, bath_option=None, verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory # TODO
        self.verbose = verbose # TODO
        self.log = lib.logger.new_logger(self.mol, self.verbose)

        # inputs
        self.dm = None
        self._imp_idx = []
        if imp_idx is not None:
            self.imp_idx = imp_idx
        else:
            self.log.info('impurity index not assigned, use the first atom as impurity')
            self.imp_idx = self.mol.atom_symbol(0)
        self.threshold = threshold
        self.es_natorb = es_natorb
        self.bath_option = bath_option

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
        # 记录 AO->LO 与 LO 基下的子空间变换，便于后续手动扩展
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
            if dm_check & imp_idx_check & threshold_check:
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
                self.log.info(f'build dmet subspace with imp idx {self.imp_idx} threshold {self.threshold}')
                return False
    
    def save_chk(self, chk_fname):
        with h5py.File(chk_fname, 'w') as fh5:
            fh5['dm'] = self.dm
            fh5['imp_idx'] = self.imp_idx
            fh5['threshold'] = self.threshold

            fh5['fo_orb'] = self.fo_orb
            fh5['fv_orb'] = self.fv_orb
            fh5['es_orb'] = self.es_orb
            fh5['es_occ'] = self.es_occ
            fh5['es_int1e'] = self.es_int1e
            fh5['es_int2e'] = self.es_int2e
            fh5['es_dm'] = self.es_dm
        return 
    
    def lowdin_orth(self, restore_imp = False):
        # lowdin orthonormalize
        caolo, cloao = lowdin_orth(self.mol)
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

        ldm = reduce(lib.dot, (cloao, self.dm, cloao.conj().T))
        return ldm, caolo, cloao
        
    def build(self, restore_imp = False, chk_fname_load='', save_chk=True, use_svd=False):
        self.dump_flags()
        dm = mf_or_cas_make_rdm1s(self.mf_or_cas)
        if dm.ndim == 3: # ROHF density matrix have dimension (2, nao, nao)
            self.dm = dm[0] + dm[1]
            open_shell = True
        else:
            self.dm = dm
            open_shell = False
        self.open_shell = open_shell
        loaded = self.load_chk(chk_fname_load)
        
        if not loaded:
            ldm, caolo, cloao = self.lowdin_orth(restore_imp)
            self.caolo = caolo
            self.cloao = cloao

            cloes, nimp, nbath, nfo, nfv, self.es_occ = build_embeded_subspace(ldm, self.imp_idx, thres=self.threshold, es_natorb=self.es_natorb, use_svd=use_svd)
            self.lo_cloes = cloes
            caoes = lib.dot(caolo, cloes)

            self.fo_orb = caoes[:, nimp+nbath: nimp+nbath+nfo]
            self.fv_orb = caoes[:, nimp+nbath+nfo: nimp+nbath+nfo+nfv]
            self.es_orb = caoes[:, :nimp+nbath]
        
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
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['MP2'])
                        elif 'RMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.info('ROMP2 bath expansion in used by default')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['RMP2'])
                            else:
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['RMP2'])
                        elif 'ROMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_ROMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                      lo2core, lo2vir, eta=self.bath_option['ROMP2'])
                            else:
                                self.log.info('ROMP2 bath expansion is degraded to RMP2 for closed-shell systems')
                                lo2MP2_bath, lo2MP2_core, lo2MP2_vir = get_RMP2_bath(self.mf_or_cas, self.es_mf, self.es_orb, self.fo_orb, self.fv_orb,
                                                                                     lo2core, lo2vir, eta=self.bath_option['ROMP2'])
                        elif 'UMP2' in self.bath_option.keys():
                            self.es_mf = self.ROHF()
                            if open_shell:
                                self.log.warn('UMP2 bath expansion is less preferred than ROMP2 for ROHF, the results must be checked carefully!')
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
        self.fo_ene()
        self.log.info('')
        self.log.info(f'energy from frozen occupied orbitals = {self.fo_ene}')
        self.log.info(f'deviation from DMET exact condition = {self.es_mf.e_tot+self.fo_ene-self.mf_or_cas.e_tot}')
        self.log.info(f'energy from embedded cluster with low level calculation = {self.es_mf.e_tot}')
        self.log.info(f'total energy with low level calculation = {self.mf_or_cas.e_tot}')

        if save_chk:
            chk_fname_save = self.title + '_dmet_chk.h5'
            self.save_chk(chk_fname_save)
        return self.es_mf
    def append_bath_by_env_idx(self, env_idx):
        """
        手动将当前环境块中的指定列追加为新的 bath 轨道。
        env_idx: List[int]，相对于当前 LO 基下环境块的列索引。
                 环境块的顺序为 [bath | fo | fv]，长度为 nbath+nfo+nfv。
        """
        if self.lo_cloes is None or self.caolo is None or self.cloao is None:
            raise RuntimeError("Embedded subspace not built or transforms not cached. Run build() first.")

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        nao = self.mol.nao

        # 现有嵌入子空间（LO 基）：Q_emb = [imp | bath]
        Q_emb = self.lo_cloes[:, :nimp+nbath]
        I = np.eye(nao)

        # 环境块（LO 基）：[bath | fo | fv]
        env_block = self.lo_cloes[:, nimp: nimp+nbath+self.nfo+self.nfv]

        # 候选列，先投影到嵌入子空间的正交补，再正交化
        P = I - Q_emb @ Q_emb.T.conj()
        Bcand = env_block[:, env_idx]
        from numpy.linalg import svd
        U, S, Vh = svd(P @ Bcand, full_matrices=False)
        lo2New_bath = U[:, :Bcand.shape[1]]

        # 重新生成与新嵌入子空间正交的环境基，并切分为 core/vir（保持数量不变）
        P2 = I - np.hstack([Q_emb, lo2New_bath]) @ np.hstack([Q_emb, lo2New_bath]).T.conj()
        C = P2 @ env_block
        Qenv, _ = np.linalg.qr(C)
        lo2New_core = Qenv[:, :self.nfo]
        lo2New_vir  = Qenv[:, self.nfo: self.nfo+self.nfv]

        # 更新 AO 基下的轨道
        lo2eo = np.hstack([Q_emb, lo2New_bath])
        self.es_orb = lib.dot(self.caolo, lo2eo)
        self.fo_orb = lib.dot(self.caolo, lo2New_core)
        self.fv_orb = lib.dot(self.caolo, lo2New_vir)

        # 更新计数与积分/密度
        nbath_new = nbath + lo2New_bath.shape[1]
        self.nes = nimp + nbath_new
        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        self.es_dm = self.make_es_dm(self.open_shell, lo2eo, self.cloao, self.dm)
        self.es_mf = self.ROHF()
    '''
    def analyze_bath_composition(self, threshold=0.1):
        """
        分析 Bath 轨道的原子成分。
        原理：计算 Bath 轨道在各个原子上的 Mulliken 布局。
        """
        if self.es_orb is None:
            self.log.warn("Embedded subspace not built.")
            return

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        # 获取 Bath 轨道的 AO 系数 (AO x Nbath)
        # self.es_orb 的结构是 [Imp | Bath]
        bath_orb_coeff = self.es_orb[:, nimp:nimp+nbath]
        
        # 重叠矩阵 S
        S = self.mol.intor_symmetric('int1e_ovlp')
        
        self.log.info(f"{'='*20} Bath Orbital Composition Analysis {'='*20}")
        
        total_atoms = [self.mol.atom_symbol(i) for i in range(self.mol.natm)]
        ao_labels = self.mol.ao_labels(fmt=None) # List of (atom_id, atom_symbol, orbital_label, component)

        for ib in range(nbath):
            C = bath_orb_coeff[:, ib]
            # Mulliken population: P_mu = C_mu * (S @ C)_mu
            # 这里我们归约到原子上
            SC = np.dot(S, C)
            pop = C * SC 
            
            atom_pops = np.zeros(self.mol.natm)
            for iao, label in enumerate(ao_labels):
                atom_id = label[0]
                atom_pops[atom_id] += pop[iao]
            
            # 找出贡献最大的几个原子
            sorted_indices = np.argsort(np.abs(atom_pops))[::-1]
            
            comp_str = []
            for idx in sorted_indices:
                val = atom_pops[idx]
                if abs(val) > threshold:
                    comp_str.append(f"{total_atoms[idx]}{idx}({val:.2f})")
            
            self.log.info(f"Bath {ib+1}: {', '.join(comp_str)}")
        self.log.info("="*65)
    '''
# ...existing code...
    def analyze_bath_composition(self, threshold=0.1):
        """
        分析 Bath 轨道的原子成分。
        原理：计算 Bath 轨道在各个原子上的 Mulliken 布局。
        """
        if self.es_orb is None:
            self.log.warn("Embedded subspace not built.")
            return

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        # 获取 Bath 轨道的 AO 系数 (AO x Nbath)
        # self.es_orb 的结构是 [Imp | Bath]
        bath_orb_coeff = self.es_orb[:, nimp:nimp+nbath]
        
        # 重叠矩阵 S
        S = self.mol.intor_symmetric('int1e_ovlp')
        
        self.log.info(f"{'='*20} Bath Orbital Composition Analysis {'='*20}")
        
        total_atoms = [self.mol.atom_symbol(i) for i in range(self.mol.natm)]
        
        # 获取两种格式的标签
        # ao_labels_str: ['0 C 1s', '0 C 2px', ...] 用于打印详细信息
        ao_labels_str = self.mol.ao_labels() 
        # ao_labels_fmt: [(0, 'C', '1s', ''), ...] 用于原子归约
        ao_labels_fmt = self.mol.ao_labels(fmt=None) 

        for ib in range(nbath):
            C = bath_orb_coeff[:, ib]
            # Mulliken population: P_mu = C_mu * (S @ C)_mu
            SC = np.dot(S, C)
            pop = C * SC 
            
            # --- 1. 原子层面的归约 (Atom Sum) ---
            atom_pops = np.zeros(self.mol.natm)
            for iao, label in enumerate(ao_labels_fmt):
                atom_id = label[0]
                atom_pops[atom_id] += pop[iao]
            
            sorted_indices = np.argsort(np.abs(atom_pops))[::-1]
            
            comp_str = []
            for idx in sorted_indices:
                val = atom_pops[idx]
                if abs(val) > threshold:
                    comp_str.append(f"{total_atoms[idx]}{idx}({val:.2f})")
            if not comp_str and len(sorted_indices) > 0:
                for idx in sorted_indices[:3]:
                    val = atom_pops[idx]
                    comp_str.append(f"{total_atoms[idx]}{idx}({val:.2f})")            
            self.log.info(f"Bath {ib+1} [Atom]: {', '.join(comp_str)}")

            # --- 2. 具体 AO 轨道层面的细节 (Orbital Detail) ---
            # 对所有 AO 的贡献排序
            sorted_ao_idx = np.argsort(np.abs(pop))[::-1]
            orb_details = []
            
            # 使用稍微宽松一点的阈值来显示细节，或者沿用 threshold
            detail_threshold = threshold 
            
            for idx in sorted_ao_idx:
                val = pop[idx]
                if abs(val) > detail_threshold:
                    # 格式化字符串，去掉多余空格，例如 "0 C 2px"
                    lbl = ao_labels_str[idx].strip()
                    orb_details.append(f"{lbl}({val:.2f})")
            
            if orb_details:
                # 缩进打印细节，方便阅读
                self.log.info(f"        [Detail]: {', '.join(orb_details)}")

        self.log.info("="*65)
    
    def find_bath_indices_from_reference(self, ref_bath_coeff, metric='overlap'):
        """
        根据参考 Bath 轨道 (ref_bath_coeff)，在当前环境空间中寻找最相似的轨道。
        这用于解决 PES 上的不连续性问题。

        Args:
            ref_bath_coeff (ndarray): 参考构型的 Bath 轨道系数 (AO 基)。
            metric (str): 目前仅支持 'overlap'。

        Returns:
            list: 推荐追加到 Bath 中的环境轨道索引 (可以直接传给 append_bath_by_env_idx)。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        # 1. 获取当前所有环境轨道（包含现有的 Bath + Frozen Occ + Frozen Vir）
        #    注意：append_bath_by_env_idx 的索引是相对于这个全环境块的。
        #    环境块在 lo_cloes 中的位置是 [nimp :]
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        
        # 获取环境轨道在 AO 基下的表达
        # lo_cloes 是 LO 基，需要左乘 caolo 变回 AO 基
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices] 
        
        # 2. 计算当前环境轨道与参考 Bath 的重叠矩阵
        #    O_ij = <Psi_env_i | S | Psi_ref_j>
        S = self.mol.intor_symmetric('int1e_ovlp')
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S, ref_bath_coeff))
        
        # 3. 匹配逻辑
        #    对于每一个参考 Bath 轨道，找到当前环境空间中重叠最大的那个轨道
        n_ref = ref_bath_coeff.shape[1]
        recommended_indices = []
        
        matched_magnitudes = []

        self.log.info(f"{'='*20} Consistent Bath Search {'='*20}")
        used_indices = set()

        for i_ref in range(n_ref):
            # 取出第 i_ref 个参考轨道与所有当前环境轨道的重叠向量
            overlaps = np.abs(ovlp_mat[:, i_ref])
            
            # 排序（从大到小）
            sorted_args = np.argsort(overlaps)[::-1]
            
            # 找到第一个还没被选过的索引
            best_idx = -1
            for idx in sorted_args:
                if idx not in used_indices:
                    best_idx = idx
                    break
            
            if best_idx != -1:
                overlap_val = overlaps[best_idx]
                used_indices.add(best_idx)
                
                # 判断这个索引当前属于哪个区？(Bath, FO, 还是 FV?)
                # env block 结构: [Current Bath (0~nbath-1) | FO | FV]
                status = "Unknown"
                if best_idx < nbath_current:
                    status = "Already in Bath"
                elif best_idx < nbath_current + self.nfo:
                    status = "In Frozen Occupied (Will be recovered)"
                    recommended_indices.append(best_idx)
                else:
                    status = "In Frozen Virtual (Will be recovered)"
                    recommended_indices.append(best_idx)
                
                self.log.info(f"Ref Bath {i_ref}: Matches Env idx {best_idx} (Overlap={overlap_val:.4f}) -> {status}")
            else:
                self.log.warn(f"Ref Bath {i_ref}: No unique match found!")

        self.log.info(f"Indices to append: {recommended_indices}")
        self.log.info("="*65)

        return recommended_indices
    '''
    def find_bath_indices_from_reference_svd(self, ref_bath_coeff):
        """
        使用 SVD 进行全局最优匹配（适合大几何变化）。
        Args:
            ref_bath_coeff (ndarray): 参考构型的 Bath 轨道系数（AO 基）。
            threshold (float): 奇异值筛选阈值，决定匹配强度下限。
        Returns:
            list: 推荐追加到 Bath 的环境轨道索引（可直接传给 append_bath_by_env_idx）。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
    # 1. 计算重叠矩阵
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]
        S = self.mol.intor_symmetric('int1e_ovlp')
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S, ref_bath_coeff))
    
    # 2. SVD分解
        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
    
    # 3. 选择奇异值大于阈值的配对
        threshold = 0.5
        valid_pairs = sigma > threshold
    
    # 4. 返回环境轨道索引
        recommended_indices = []
        for i in range(np.sum(valid_pairs)):
        # U的列对应当前环境轨道的线性组合
        # 需要找到U[:, i]中权重最大的原始轨道
            weights = np.abs(U[:, i])
            best_env_idx = np.argmax(weights)
        
            if best_env_idx not in recommended_indices:
                recommended_indices.append(best_env_idx)
                self.log.info(
                    f"SVD pair {i}: σ={sigma[i]:.4f}, "
                    f"Env idx {best_env_idx} (weight={weights[best_env_idx]:.3f})"
                )
    
        return recommended_indices  '''  
    '''
    def find_bath_indices_from_reference_svd(self, ref_bath_coeff, threshold=0.5):
        """
        使用 SVD 进行全局最优匹配（适合大几何变化）。
        Args:
            ref_bath_coeff (ndarray): 参考构型的 Bath 轨道系数（AO 基）。
            threshold (float): 奇异值筛选阈值，决定匹配强度下限。
        Returns:
            list: 推荐追加到 Bathw 的环境轨道索引（可直接传给 append_bath_by_env_idx）。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp

        # 1) 构造当前环境块的 AO 轨道
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        # 2) 重叠矩阵 O = C_env^† S C_ref
        S = self.mol.intor_symmetric('int1e_ovlp')
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S, ref_bath_coeff))

        # 3) SVD：ovlp_mat = U Σ Vh
        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)

        # 4) 依据奇异值从大到小匹配环境轨道（跳过已在 Bath 的）
        recommended_indices = []
        used_env = set()

        self.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
        for i, s in enumerate(sigma):
            if s < threshold:
                self.log.info(f"Stop at pair {i}: σ={s:.4f} < threshold={threshold}")
                break

            weights = np.abs(U[:, i])
            sorted_env = np.argsort(weights)[::-1]

            best_env_idx = -1
            for idx in sorted_env:
                if idx in used_env:
                    continue
                best_env_idx = idx
                break

            if best_env_idx < 0:
                self.log.warn(f"SVD pair {i}: No unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            if best_env_idx < nbath_current:
                status = "Already in Bath (skip)"
            elif best_env_idx < nbath_current + self.nfo:
                status = "In Frozen Occupied (Will be recovered)"
                recommended_indices.append(best_env_idx)
            else:
                status = "In Frozen Virtual (Will be recovered)"
                recommended_indices.append(best_env_idx)

            self.log.info(
                f"SVD pair {i}: σ={s:.4f}, Env idx {best_env_idx} "
                f"(weight={weights[best_env_idx]:.3f}) -> {status}"
            )

        self.log.info(f"Indices to append: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices
    '''
    def find_bath_indices_from_reference_svd(self, ref_bath_coeff, threshold=None):
        """
        使用 SVD 进行全局最优匹配（适合大几何变化）。
        修改策略：不再主要依赖阈值，而是强制补齐到参考体系的 Bath 数目。
        
        Args:
            ref_bath_coeff (ndarray): 参考构型的 Bath 轨道系数（AO 基）。
            threshold (float): (保留参数以兼容旧代码，但在本逻辑中主要由数目决定)。
        Returns:
            list: 推荐追加到 Bath 的环境轨道索引（可直接传给 append_bath_by_env_idx）。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        # 1. 确定目标数目
        # 参考体系的 Bath 数目
        n_ref_bath = ref_bath_coeff.shape[1]
        # 需要找回的数目 = 参考总数 - 当前已有的
        n_needed = n_ref_bath - nbath_current
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
        self.log.info(f"Reference Bath Size: {n_ref_bath} | Current Bath Size: {nbath_current}")
        
        if n_needed <= 0:
            self.log.info("Current bath size is already >= Reference bath size. No extension needed.")
            self.log.info("="*65)
            return []
            
        self.log.info(f"Target: Recover {n_needed} orbitals from Frozen space to match reference size.")

        # 2. 构造当前环境块的 AO 轨道
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        # 3. 计算重叠矩阵 O = C_env^T S C_ref
        S = self.mol.intor_symmetric('int1e_ovlp')
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S, ref_bath_coeff))

        # 4. SVD 分解：ovlp_mat = U Σ Vh
        # sigma 已经从大到小排列，代表与参考 Bath 的重叠程度
        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)

        recommended_indices = []
        used_env = set()

        # 5. 遍历 SVD 模式，直到找到足够的 Frozen 轨道
        for i, s in enumerate(sigma):
            if len(recommended_indices) >= n_needed:
                break
                
            # U[:, i] 是当前环境基下的系数向量，找出权重最大的那个物理轨道
            weights = np.abs(U[:, i])
            sorted_env = np.argsort(weights)[::-1]
            
            best_env_idx = -1
            # 在权重最大的轨道中，找一个没被用过的
            for idx in sorted_env:
                if idx not in used_env:
                    best_env_idx = idx
                    break
            
            if best_env_idx == -1:
                self.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            
            # 判断这个最佳匹配的轨道在哪里
            status = ""
            if best_env_idx < nbath_current:
                # 如果最佳匹配是当前的 Bath，说明不需要回复，它已经在里面了
                status = "Match Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                # 掉进了 Frozen Occupied
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                # 掉进了 Frozen Virtual
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): {status}")

        if len(recommended_indices) < n_needed:
            self.log.warn(f"Only found {len(recommended_indices)} orbitals to recover, expected {n_needed}.")

        self.log.info(f"Indices to append: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices

    def replace_bath_by_reference_indices(self, ref_bath_coeff, threshold=0.5):
        """用参考 Bath（SVD 匹配）直接替换当前 Bath，而不是追加。"""
        idxs = self.find_bath_indices_from_reference_svd(ref_bath_coeff, threshold)
        if len(idxs) == 0:
            self.log.info("No indices to replace.")
            return

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        env_loc = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)

        # 取出环境块（LO 基）
        env_lo = self.lo_cloes[:, env_loc]

        # 将选中的环境轨道放到最前面作为新 Bath，剩余的放回 Frozen
        idxs = list(idxs)
        rest = [i for i in range(env_lo.shape[1]) if i not in idxs]
        new_env_order = idxs + rest
        env_lo = env_lo[:, new_env_order]

        # 更新嵌入变换矩阵与 Bath 数目
        self.lo_cloes[:, env_loc] = env_lo
        self.nes = nimp + len(idxs)  # 只保留选中的作为 Bath

        # 刷新嵌入 ROHF 与后续性质
        self.refresh_embedded_mf()
    '''# ...existing code...
    def use_reference_bath(self, ref_bath_coeff_ao):
        """
        强制使用参考构型的 Bath 轨道（投影并正交化），而非从当前 Schmidt 分解中选择。
        这能保证 Bath 空间随几何构型平滑变化。
        
        Args:
            ref_bath_coeff_ao (ndarray): 参考 Bath 在其原构型下的 AO 系数。
                                         注意：通常这应该是投影到当前 AO 基下的形式，
                                         或者如果是同一分子不同构型，需确保 AO 维度一致。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        self.log.info(f"{'='*20} Projecting Reference Bath {'='*20}")
        
        nimp = len(self.imp_idx)
        nao = self.mol.nao
        
        # 1. 准备基组转换矩阵
        # S: 当前构型的重叠矩阵
        # caolo: AO -> LO 变换矩阵 (S^-1/2)
        # cloao: LO -> AO 变换矩阵 (S^1/2)
        S = self.mol.intor_symmetric('int1e_ovlp')
        
        # 2. 将参考 Bath (AO) 转换到当前 LO 基
        #    C_lo = C_lo_ao @ S @ C_ao
        ref_bath_lo = self.cloao @ S @ ref_bath_coeff_ao
        
        # 3. 对 Impurity 空间进行正交化 (Impurity 在 LO 基下是单位向量的前 nimp 列)
        #    P_imp = sum |i><i| for i in imp
        #    Bath' = (1 - P_imp) Bath_ref
        #    在 LO 基下，Imp 只是占据了前 nimp 个索引
        ref_bath_lo_orth = ref_bath_lo.copy()
        ref_bath_lo_orth[:nimp, :] = 0.0  # 移除 Impurity 成分
        
        # 4. 对投影后的 Bath 进行正交归一化 (SVD)
        U, s, Vh = np.linalg.svd(ref_bath_lo_orth, full_matrices=False)
        # 选取非零模式作为新的 Bath
        # 通常保留与输入 ref_bath 数量一致的轨道，除非线性相关严重
        n_ref_bath = ref_bath_coeff_ao.shape[1]
        new_bath_lo = U[:, :n_ref_bath]
        
        self.log.info(f"Reference Bath projected. Singular values min/max: {s[:n_ref_bath].min():.4f} / {s[:n_ref_bath].max():.4f}")
        
        # 5. 构建剩余环境空间 (Frozen Part)
        #    全空间 I，扣除 Imp 和 New Bath
        #    P_rem = I - P_imp - P_bath
        P_occupied = np.zeros((nao, nao))
        P_occupied[:nimp, :nimp] = np.eye(nimp) # Imp projector
        P_occupied += new_bath_lo @ new_bath_lo.T # Bath projector
        
        P_rem = np.eye(nao) - P_occupied
        
        # 6. 在剩余空间中对角化密度矩阵，以分离 Frozen Occ 和 Frozen Vir
        #    D_lo = current density in LO basis
        #    目标: 找到 P_rem 空间中 D_lo 的本征态
        ldm = reduce(lib.dot, (self.cloao, self.dm, self.cloao.conj().T))
        
        # 为了数值稳定性，先找到 P_rem 的列空间基
        w_rem, v_rem = np.linalg.eigh(P_rem)
        rem_mask = w_rem > 0.5
        basis_rem = v_rem[:, rem_mask] # 剩余空间的基向量
        
        # 在该子空间投影并对角化密度
        dm_rem_sub = basis_rem.T @ ldm @ basis_rem
        w_sub, v_sub = np.linalg.eigh(dm_rem_sub)
        
        # 变换回 LO 全空间
        orbs_rem_lo = basis_rem @ v_sub
        
        # 7. 根据占据数划分 FO / FV
        #    注意 eigh 返回是升序，所以 Occ(2.0) 在后面
        tol = 1e-2
        is_fo = w_sub > (2.0 - tol)
        is_fv = w_sub <= (2.0 - tol) # 其余作为虚轨道，包括部分占据的（如果有）
        
        new_fo_lo = orbs_rem_lo[:, is_fo]
        new_fv_lo = orbs_rem_lo[:, is_fv]
        
        # 翻转顺序让占据数大的在 FO 里通常是好的，但 dm定义顺序无所谓，只要分组对
        
        # 8. 组装新的 lo_cloes 矩阵 [Imp | Bath | FO | FV]
        lo_imp = np.eye(nao)[:, :nimp]
        
        new_lo_cloes = np.hstack([lo_imp, new_bath_lo, new_fo_lo, new_fv_lo])
        
        # 9. 更新对象属性
        self.lo_cloes = new_lo_cloes
        self.nes = nimp + new_bath_lo.shape[1]
        self.nfo = new_fo_lo.shape[1]
        self.nfv = new_fv_lo.shape[1]
        
        self.log.info(f"New Check: N_imp={nimp}, N_bath={self.nes-nimp}, N_fo={self.nfo}, N_fv={self.nfv}")
        
        # 10. 重建 AO 基下的轨道和积分
        self.es_orb = self.caolo @ self.lo_cloes[:, :self.nes]
        self._build_embedded_integrals_and_mf()
        
        self.log.info("Embedding space rebuilt using projected reference bath.")
        self.log.info("="*65)
    '''
    def _build_embedded_integrals_and_mf(self):
        """内部辅助函数：重建积分和 Mean-Field"""
        # 更新辅助轨道
        self.fo_orb = self.caolo @ self.lo_cloes[:, self.nes:self.nes+self.nfo]
        self.fv_orb = self.caolo @ self.lo_cloes[:, self.nes+self.nfo:]
        
        # 1e 积分
        self.es_int1e = make_es_int1e(self.mf_or_cas, self.fo_orb, self.es_orb)
        # 2e 积分
        self.es_int2e = make_es_int2e(self.mf_or_cas, self.es_orb)
        
        # 投影密度
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, self.dm)
        
        # 重置 MF
        self.es_mf = self.ROHF()

    def build_consistent_with_reference(self, ref_bath_coeff_ao, metric='overlap', strict_size=True):
        """
        构建嵌入空间，但 Bath 轨道的选择优先考虑与参考 Bath 的相似度，
        而不是仅仅基于占据数阈值。这确保了势能面扫描时的轨道平滑性。
        
        Args:
            ref_bath_coeff_ao (ndarray): 参考 Bath 的 AO 系数 (AO x N_bath_ref)
            strict_size (bool): True 表示强制 Bath 数目与参考一致；
                                如果 False，则仅作为排序依据（暂未完全实现，建议 True）。
        """
        self.dump_flags()

        # 1. 准备密度矩阵
        dm = mf_or_cas_make_rdm1s(self.mf_or_cas)
        if dm.ndim == 3: 
            self.dm = dm[0] + dm[1]
            self.open_shell = True
        else:
            self.dm = dm
            self.open_shell = False

        # 2. 获取正交化基 (LO) 和变换矩阵
        # 使用 restore_imp=True 保持与 build 一致的 impurity 定义
        # ldm: LO 基下的密度矩阵
        ldm, caolo, cloao = self.lowdin_orth(restore_imp=True) 
        self.caolo = caolo
        self.cloao = cloao
        
        nao = self.mol.nao
        nimp = len(self.imp_idx)
        env_idx = [x for x in range(nao) if x not in self.imp_idx]
        
        # 3. 对环境密度进行 Schmidt 分解 (获得所有的环境自然轨道)
        # ldm_env 是 LO 基下环境块的密度
        ldm_env = ldm[np.ix_(env_idx, env_idx)]
        # evals_env: 占据数 (升序); evecs_env: 对应轨道系数
        evals_env, evecs_env = np.linalg.eigh(ldm_env)
        
        # 将环境轨道转换到 AO 基 (用于重叠计算)
        # 构造全 LO 空间下的 ENO 系数 (Imp 部分为 0)
        # lo_env_coeff: (Naao, Nenv)
        lo_env_coeff = np.zeros((nao, len(env_idx)))
        lo_env_coeff[env_idx, :] = evecs_env
        eno_ao = self.caolo @ lo_env_coeff
        
        # 4. 计算与参考 Bath 的匹配度
        S = self.mol.intor_symmetric('int1e_ovlp')
        # ovlp_mat[i, j] = <ENO_i | S | Ref_Bath_j>
        ovlp_mat = reduce(lib.dot, (eno_ao.T.conj(), S, ref_bath_coeff_ao))
        
        # 对每个环境自然轨道，计算它与参考 Bath 空间的相似度分数
        sim_scores = np.sum(np.abs(ovlp_mat)**2, axis=1)
        
        # 5. 挑选 Bath 轨道
        target_nbath = ref_bath_coeff_ao.shape[1]
        
        if strict_size:
            # 选 sim_scores 最大的作为 Bath
            # argsort 默认升序 -> [-N:] 取最后（最大）的 N 个
            top_indices = np.argsort(sim_scores)[-target_nbath:]
            bath_indices_in_env = sorted(top_indices) 
        else:
            self.log.warn("strict_size=False fallback: forcing strict size match.")
            top_indices = np.argsort(sim_scores)[-target_nbath:]
            bath_indices_in_env = sorted(top_indices)

        bath_list = list(bath_indices_in_env)
        
        # 6. 其余分为 Frozen Occ / Frozen Virt
        all_indices = set(range(len(env_idx)))
        bath_set = set(bath_list)
        frozen_indices = list(all_indices - bath_set)
        
        fo_list = []
        fv_list = []
        
        # 根据占据数划分 Frozen 部分
        for idx in frozen_indices:
            if evals_env[idx] > 1.0:
                fo_list.append(idx)
            else:
                fv_list.append(idx)
        
        # 7. 组装嵌入空间矩阵 lo_cloes [Imp | Bath | FO | FV]
        # Impurity 部分
        c_imp = np.eye(nao)[:, self.imp_idx]
        
        # 辅助函数：将环境轨道映射回全矩阵尺寸
        def embed_env_coeffs(coeffs_hum):
            res = np.zeros((nao, coeffs_hum.shape[1]))
            res[env_idx, :] = coeffs_hum
            return res

        c_bath_lo = embed_env_coeffs(evecs_env[:, bath_list])
        c_fo_lo   = embed_env_coeffs(evecs_env[:, fo_list])
        c_fv_lo   = embed_env_coeffs(evecs_env[:, fv_list])
        
        self.lo_cloes = np.hstack([c_imp, c_bath_lo, c_fo_lo, c_fv_lo])
        
        # 8. 更新属性
        self.nes = nimp + len(bath_list)
        self.nfo = len(fo_list)
        self.nfv = len(fv_list)
        
        self.log.info(f"Consistent Build: N_imp={nimp}, N_bath={self.nes-nimp}, N_fo={self.nfo}, N_fv={self.nfv}")
        self.log.info(f"Selection Similarity (Min/Max): {sim_scores[bath_list].min():.4f} / {sim_scores[bath_list].max():.4f}")

        # 9. 重建 AO 基轨道与积分
        self.es_orb = self.caolo @ self.lo_cloes[:, :self.nes]
        self._build_embedded_integrals_and_mf()
        
        # 计算 Frozen Energy (用于总能量修正)
        self.fo_ene()
        self.log.info(f'Frozen energy: {self.fo_ene:.6f}')
        self.log.info("="*65)
        self.log.info(f"Consistent Build selected indices: {sorted(bath_list)}")
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

        es_ovlp = reduce(lib.dot, (self.es_orb.conj().T, self.mol.intor_symmetric('int1e_ovlp'), self.es_orb))
        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: es_ovlp
        es_mf._eri = self.es_int2e
        es_mf.mo_coeff = np.eye(self.nes)

        # assume we only perfrom ROHF-in-ROHF embedding

        # assert lib.einsum('ijj->', es_dm) == mol.nelectron
        es_mf.level_shift = self.mf_or_cas.level_shift
        es_mf.conv_check = False
        es_mf.kernel(self.es_dm)
        self.es_occ = es_mf.mo_occ
        return es_mf
    '''
    def ROHF(self, relax_density=False):
        """
        构建嵌入空间的 Mean-Field 对象。
        
        Args:
            relax_density (bool): 
                False (默认) - 执行单次对角化 (One-shot)。使用投影密度构建 Fock 矩阵并对角化，
                              保持与原始 SSDMET 的一致性，不改变密度。
                True         - 执行 SCF 迭代。让密度在嵌入哈密顿量下松弛至自洽。
        """
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
        
        # 确保积分和维度匹配当前的 self.nes
        es_ovlp = reduce(lib.dot, (self.es_orb.conj().T, self.mol.intor_symmetric('int1e_ovlp'), self.es_orb))
        
        es_mf.get_hcore = lambda *args: self.es_int1e
        es_mf.get_ovlp = lambda *args: es_ovlp
        es_mf._eri = self.es_int2e
        
        # 准备密度矩阵（如果是开壳层 RDM，折叠成总密度用于 RHF 构建 Fock）
        dm_guess = self.es_dm
        if hasattr(dm_guess, 'ndim') and dm_guess.ndim == 3 and mol.spin == 0:
            dm_guess = dm_guess[0] + dm_guess[1]

        if relax_density:
            # 允许密度松弛（迭代 SCF）
            es_mf.mo_coeff = np.eye(self.nes)
            es_mf.mo_energy = np.zeros((self.nes))
            es_mf.level_shift = self.mf_or_cas.level_shift
            es_mf.conv_check = False
            es_mf.kernel(self.es_dm)
        else:
            # 禁止密度松弛 (单次对角化，保持一致性)
            # 1. 构建 Fock 矩阵: F = H_core + V_eff(P_proj)
            vhf = es_mf.get_veff(mol, dm_guess)
            fock = self.es_int1e + vhf
            
            # 2. 对角化 Fock 矩阵得到正则轨道
            # 使用现有重叠矩阵 es_ovlp (通常接近单位阵) 进行广义本征值求解
            e, c = es_mf.eig(fock, es_ovlp)
            
            # 3. 填充 MF 对象属性，供 CCSD 使用
            es_mf.mo_coeff = c
            es_mf.mo_energy = e
            
            # 4. 根据电子数确定占据情况
            nocc = mol.nelectron // 2
            mo_occ = np.zeros(self.nes)
            mo_occ[:nocc] = 2
            es_mf.mo_occ = mo_occ
            
        self.es_occ = es_mf.mo_occ
        return es_mf
    '''    
    def avas(self, aolabels, *args, **kwargs):
        from embed_sim import myavas
        total_mf = self.total_mf()
        total_mf.mo_occ = round_off_occ(total_mf.mo_occ) # make 2/0 occupation to be int
        ncas, nelec, mo = myavas.avas(total_mf, aolabels, ncore=self.nfo, nunocc = self.nfv, canonicalize=False, *args, **kwargs) # canonicalize should be set to False, since it require orbital energy

        es_mo = reduce(lib.dot, (self.es_orb.T.conj(), self.mol.intor_symmetric('int1e_ovlp'), mo[:, self.nfo: self.nfo+self.nes]))
        return ncas, nelec, es_mo 
    
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
    
    def fo_ene(self, e_nuc = True):
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
    def ccsdt_solver(self, with_t=True):
        " run ccsd(t) as the solver for dmet subspace "
        from pyscf import cc
        if self.mol.spin != 0:
            raise NotImplementedError('CCSD solver for open-shell system is not implemented yet')
        if self.es_mf is None:
            raise RuntimeError('embedded subspace is not built yet, please run build() first')
        self.log.info('Running CCSD solver for embedded cluster ...')
        self.log.info('=' * 60)
        mycc = cc.CCSD(self.es_mf)
        mycc.verbose = self.verbose
        mycc.max_memory = self.max_mem
        mycc.kernel()
        e_tot_ccsd = mycc.e_tot + self.fo_ene

        self.log.info('CCSD correlation energy = %.12f', mycc.e_corr)
        self.log.info('Total CCSD energy = %.12f', e_tot_ccsd)
        if with_t:
            et = mycc.ccsd_t()
            e_tot_ccsd_t = e_tot_ccsd + et
            self.log.info('CCSD(T) correlation energy = %.12f', mycc.e_corr + et)
            self.log.info('Total CCSD(T) energy = %.12f', e_tot_ccsd_t)
            return mycc, e_tot_ccsd_t
        return e_tot_ccsd
    def export_molden(self, filename, mode='es'):
        """
        导出轨道到 Molden 文件用于可视化 (如 VESTA, Molden, Jmol)。
        
        Args:
            filename (str): 输出文件名 (e.g. 'orbitals.molden').
            mode (str): 
                'es' (默认): 仅导出嵌入空间 (Embedded Space) 的轨道。
                             前 N_imp 个为杂质轨道，剩余为 Bath 轨道。
                'all': 导出全空间轨道 [冻结占据, 杂质, Bath, 冻结虚轨]。
        """
        from pyscf import tools
        if self.es_orb is None:
            self.log.warn("Embedded subspace not built. Cannot export orbitals. Run build() first.")
            return

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp

        self.log.info(f'Exporting orbitals to {filename} (mode={mode})')
        
        if mode == 'es':
            self.log.info(f'  [1 - {nimp}]: Impurity Orbitals (杂质轨道)')
            self.log.info(f'  [{nimp+1} - {nimp+nbath}]: Bath Orbitals (能浴轨道)')
            # es_orb 结构即为 [Impurity, Bath]
            tools.molden.from_mo(self.mol, filename, self.es_orb)
            
        elif mode == 'all':
            # 拼接所有轨道
            mo_coeff = np.hstack((self.fo_orb, self.es_orb, self.fv_orb))
            
            # 生成假的占据数数组，方便在可视化软件中区分区域
            # 冻结占有=2.0, 其他=0.0
            occ = np.zeros(mo_coeff.shape[1])
            occ[:self.nfo] = 2.0
            
            self.log.info(f'  [1 - {self.nfo}]: Frozen Occupied (冻结核心)')
            self.log.info(f'  [{self.nfo+1} - {self.nfo+nimp}]: Impurity Orbitals')
            self.log.info(f'  [{self.nfo+nimp+1} - {self.nfo+nimp+nbath}]: Bath Orbitals')
            self.log.info(f'  [Remaining]: Frozen Virtual')
            
            tools.molden.from_mo(self.mol, filename, mo_coeff, occ=occ)
        else:
            self.log.error(f"Unknown mode '{mode}' for export_molden")



    def density_fit(self, with_df=None):
        from embed_sim.df import DFSSDMET
        if with_df is None:
            if not getattr(self.mf_or_cas, 'with_df', False):
                raise NotImplementedError
            else:
                with_df = self.mf_or_cas.with_df
        return DFSSDMET(self.mf_or_cas, self.title, imp_idx=self.imp_idx, threshold=self.threshold,
                        with_df=with_df, es_natorb=self.es_natorb, bath_option=self.bath_option, verbose=self.verbose)
