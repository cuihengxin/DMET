import numpy as np
from functools import reduce
from scipy.linalg import block_diag
import h5py

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, ao2mo, lo

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
        #self.fo_ene()
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
# ...existing code... 0203 15:50
    def calculate_bath_overlap_scores(self, ref_bath_coeff_ao, ref_mol=None):
        """
        计算当前环境轨道与参考 Bath 轨道的相似度分数。
        [Fix] 使用 intor_cross 处理几何变化，移除 make_frag_dm 调用。
        """
        # --- 0. 确保密度矩阵已初始化 ---
        if self.dm is None:
            dm = mf_or_cas_make_rdm1s(self.mf_or_cas)
            if dm.ndim == 3: 
                # ROHF/UHF density: sum alpha + beta for spatial orbital definition
                self.dm = dm[0] + dm[1] 
                self.open_shell = True
            else:
                self.dm = dm
                self.open_shell = False

        # 1. 获取当前环境的 Schmidt 占据数和轨道
        try:
            ldm, caolo, cloao = self.lowdin_orth(restore_imp=True)
            self.caolo = caolo # 更新缓存
            self.cloao = cloao
        except Exception as e:
            self.log.error(f"Error in lowdin_orth: {e}")
            raise

        nao = self.mol.nao
        # 提取环境块索引
        env_idx = [x for x in range(nao) if x not in self.imp_idx]
        
        # 提取环境块密度矩阵 (在 LO 基下)
        dm_env_lo = ldm[np.ix_(env_idx, env_idx)]
        
        # 对环境密度矩阵进行对角化得到 Schmidt 轨道
        # evecs_env_lo_sub 是 LO 基下的系数 (N_env x N_env)
        evals_env, evecs_env_lo_sub = np.linalg.eigh(dm_env_lo)
        
        # 排序 (降序)
        idx_sort = np.argsort(evals_env)[::-1]
        evals_env = evals_env[idx_sort]
        evecs_env_lo_sub = evecs_env_lo_sub[:, idx_sort]
        
        # 转换回 AO 基底
        N_env = len(env_idx)
        C_env_in_LO = np.zeros((nao, N_env))
        C_env_in_LO[env_idx, :] = evecs_env_lo_sub
        evecs_env_ao = caolo @ C_env_in_LO  # Shape: (N_AO, N_env)
        
        # 2. 计算相似度 (Orbital Overlap)
        if ref_mol is None:
            s_cross = self.mol.intor_symmetric('int1e_ovlp')
        else:
            s_cross = gto.intor_cross('int1e_ovlp', self.mol, ref_mol)
            
        if ref_bath_coeff_ao is None:
             return np.zeros(N_env), evals_env, evecs_env_lo_sub

        # O_ij = < Psi_env_i(new) | Psi_bath_j(ref) >
        overlap_matrix = reduce(lib.dot, (evecs_env_ao.T.conj(), s_cross, ref_bath_coeff_ao))
        sim_scores = np.sum(np.abs(overlap_matrix)**2, axis=1)
        
        return sim_scores, evals_env, evecs_env_lo_sub

    def build_union_with_reference(self, ref_bath_coeff_ao, ref_mol=None):
        """
        构建嵌入空间，Bath 轨道集合取以下两者的并集：
        1. 集合1 (S_ref): 与参考 Bath 重叠最大的 M 个当前环境轨道 (M = 参考 Bath 数目)。
        2. 集合2 (S_std): 当前结构下，根据占据数阈值自动选出的 Bath 轨道。
        """
        self.dump_flags()
        
        # 1. 计算所有环境 Schmidt 轨道的重叠分数和占据数
        sim_scores, evals_env, evecs_env_sub = self.calculate_bath_overlap_scores(ref_bath_coeff_ao, ref_mol)
        
        nimp = len(self.imp_idx)
        nao = self.mol.nao
        env_idx = [x for x in range(nao) if x not in self.imp_idx]
        
        # 2. 选取 S_ref: Top-M Overlap
        M = ref_bath_coeff_ao.shape[1] 
        idx_by_score = np.argsort(sim_scores)[::-1]
        set1_indices = idx_by_score[:M]
        
        # 3. 选取 S_std: Threshold
        set2_indices = np.nonzero((evals_env >= self.threshold) & (evals_env <= 2 - self.threshold))[0]
        
        # 4. 取并集
        union_indices = sorted(list(set(set1_indices) | set(set2_indices)))
        
        # --- Detailed Logging ---
        self.log.info(f"{'='*20} Union Bath Strategy Detailed Report {'='*20}")
        self.log.info(f"Ref Size M={M} (Top-M Selection)")
        self.log.info(f"{'Env Idx':>7} | {'Overlap Score':>13} | {'Schmidt Occ':>11} | {'Source'}")
        self.log.info("-" * 55)
        
        for idx in union_indices:
            score = sim_scores[idx]
            occ = evals_env[idx]
            source = []
            if idx in set1_indices:
                source.append("Ref-TopM")
            if idx in set2_indices:
                source.append("Thres")
            
            source_str = "+".join(source)
            self.log.info(f"{idx:7d} | {score:13.6f} | {occ:11.6f} | {source_str}")
            
        self.log.info("-" * 55)
        self.log.info(f"Final Union Size: {len(union_indices)}")
        
        # 5. 构建 remaining indices (Frozen)
        all_env_indices = set(range(len(evals_env)))
        bath_set = set(union_indices)
        frozen_indices = list(all_env_indices - bath_set)
        
        fo_list = []
        fv_list = []
        for idx in frozen_indices:
            if evals_env[idx] > 1.0:
                fo_list.append(idx)
            else:
                fv_list.append(idx)
                
        # 6. 组装最终的变换矩阵 lo_cloes
        c_imp = np.eye(nao)[:, self.imp_idx]
        
        def expand_env_coeff(c_sub):
            c_full = np.zeros((nao, c_sub.shape[1]))
            c_full[env_idx, :] = c_sub
            return c_full
        
        c_bath_lo = expand_env_coeff(evecs_env_sub[:, union_indices])
        c_fo_lo   = expand_env_coeff(evecs_env_sub[:, fo_list])
        c_fv_lo   = expand_env_coeff(evecs_env_sub[:, fv_list])
        
        self.lo_cloes = np.hstack([c_imp, c_bath_lo, c_fo_lo, c_fv_lo])
        
        # 7. 更新计数
        self.nes = nimp + len(union_indices)
        self.nfo = len(fo_list)
        self.nfv = len(fv_list)
        
        # 8. 必须计算 self.es_occ
        ldm_full = reduce(lib.dot, (self.cloao, self.dm, self.cloao.conj().T))
        C_es_lo = self.lo_cloes[:, :self.nes]
        dm_es = reduce(lib.dot, (C_es_lo.T.conj(), ldm_full, C_es_lo))
        
        if self.es_natorb:
            es_occ, es_u = np.linalg.eigh(dm_es)
            idx_es = np.argsort(es_occ)[::-1]
            self.es_occ = es_occ[idx_es]
            es_u = es_u[:, idx_es]
            self.lo_cloes[:, :self.nes] = C_es_lo @ es_u
        else:
            self.es_occ = np.linalg.eigvalsh(dm_es)

        # 9. 重建积分
        self.es_orb = self.caolo @ self.lo_cloes[:, :self.nes]
        self.fo_orb = self.caolo @ self.lo_cloes[:, self.nes : self.nes+self.nfo]
        self.fv_orb = self.caolo @ self.lo_cloes[:, self.nes+self.nfo :]
        
        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, self.dm)
        
        self.es_mf = self.ROHF()
        self.calc_fo_ene()
        
        self.log.info(f"Union Build Done. Frozen Energy: {self.fo_ene:.6f}")
        self.log.info("="*65)
    def append_bath_by_env_idx(self, env_idx):
        """
        手动将当前环境块中的指定列追加为新的 bath 轨道。
        
        Args:
            env_idx: List[int]。相对于整个环境块 [Bath | FO | FV] 的索引。
                     (这与 find_bath_indices 函数的返回值是一致的)
        """
        if self.lo_cloes is None or self.caolo is None or self.cloao is None:
            raise RuntimeError("Embedded subspace not built or transforms not cached. Run build() first.")

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        # 1. 坐标转换：从 [Bath | FO | FV] 转换到 [FO | FV]
        #    因为我们要操作的是冻结部分。
        indices_to_move = [] # 存储相对于 [FO|FV] 块的局部索引
        
        for idx in env_idx:
            # 如果索引小于 nbath，说明它已经是在 Bath 里面了，不需要移动
            if idx < nbath:
                self.log.warn(f"Index {idx} is already in Bath (current nbath={nbath}), skipping.")
            else:
                # 减去 nbath 偏移量
                indices_to_move.append(idx - nbath)
        
        if not indices_to_move:
            self.log.warn("No valid Frozen orbitals selected to append.")
            return

        # 2. 准备数据结构
        # Q_emb: [Imp | Bath]
        Q_emb = self.lo_cloes[:, :nimp+nbath]
        # env_block: [FO | FV] (冻结部分)
        env_block = self.lo_cloes[:, nimp+nbath:] 
        
        # 3. 统计 FO/FV 移动情况并计算新电子数
        n_shifted_fo = 0
        n_shifted_fv = 0
        
        # 现在的 indices_to_move 是局部索引 (0 对应第一个 FO)
        for local_idx in indices_to_move:
            if local_idx < self.nfo:
                n_shifted_fo += 1
            else:
                n_shifted_fv += 1
        
        self.log.info(f"Appending Bath: Shifted {n_shifted_fo} from FO, {n_shifted_fv} from FV")

        # 4. 提取并移动轨道
        # 构建 mask 选择要移动的列 (作用于 env_block)
        mask_move = np.zeros(env_block.shape[1], dtype=bool)
        mask_move[indices_to_move] = True
        
        B_new_candidates = env_block[:, mask_move] 
        
        # 对新 Bath 做 QR 确保正交归一性 (虽然理论上本来就是正交的)
        lo2New_bath, _ = np.linalg.qr(B_new_candidates)
        
        # 5. 处理剩余的 Frozen 空间 (保持 FO/FV 的独立性，不混合)
        indices_all = np.arange(env_block.shape[1])
        indices_remain = indices_all[~mask_move]
        
        # 区分剩余的 FO 和 FV
        # env_block 的前 self.nfo 个是 FO
        idx_remain_fo = [i for i in indices_remain if i < self.nfo]
        idx_remain_fv = [i for i in indices_remain if i >= self.nfo]
        
        lo2New_core = env_block[:, idx_remain_fo]
        lo2New_vir  = env_block[:, idx_remain_fv]
        
        # 6. 更新 lo_cloes 矩阵
        self.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_core, lo2New_vir])
        
        # 7. 更新计数
        self.nes  = nimp + nbath + lo2New_bath.shape[1] 
        self.nfo -= n_shifted_fo
        self.nfv -= n_shifted_fv
        
        # 更新 AO 基下的轨道
        self.es_orb = lib.dot(self.caolo, self.lo_cloes[:, :self.nes])
        self.fo_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes : self.nes+self.nfo])
        self.fv_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes+self.nfo :])

        # 8. 更新积分和密度
        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, self.dm)
        
        # 9. 重新生成 MF 对象 (self.nfo 已更新，ROHF 会自动增加电子数)
        self.es_mf = self.ROHF()
        self.calc_fo_ene() 
        
        self.log.info(f"Bath appended. New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        self.log.info(f"Frozen Energy updated: {self.fo_ene:.6f}")        
        self.log.info("========================== End of Bath Append ====================")
        self.log.info(f"Bath appended. New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        self.log.info("========================== End of Bath Append ====================")
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
    
    def find_bath_indices_from_reference_svd(self, ref_coeff, threshold=None):
        """
        使用 SVD 进行全局最优匹配（适合大几何变化和 Imp 空间改变）。
        可以将新结构的全部环境轨道与旧结构的整个 EO (Imp+Bath) 空间进行对比。
        通过挑选与旧结构 EO 重叠度最高的 Frozen 轨道加入 Bath，
        自动补齐 Bath 使新结构总 EO 维数与旧结构总 EO 维数保持一致。
        
        Args:
            ref_coeff (ndarray): 参考构型的轨道系数（如整个旧 EO 空间，AO 基）。
            threshold (float): (保留参数以兼容旧代码)。
        Returns:
            list: 推荐追加到 Bath 的环境轨道索引（可直接传给 append_bath_by_env_idx）。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp_current = len(self.imp_idx)
        nbath_current = self.nes - nimp_current
        
        # 1. 确定目标数目
        # 参考体系传入的轨道总数目 (通常是旧体系的 n_imp + n_bath)
        n_ref = ref_coeff.shape[1]
        # 为了保证 EO 空间总数一致：new_nimp + new_nbath = n_ref
        target_nbath = n_ref - nimp_current
        n_needed = target_nbath - nbath_current
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search (Full EO Match) {'='*20}")
        self.log.info(f"Reference Target Size: {n_ref} | Current Imp Size: {nimp_current}")
        self.log.info(f"Current Bath Size:   {nbath_current} | Target Bath Size: {target_nbath}")
        
        if n_needed <= 0:
            self.log.info(f"Current Bath size ({nbath_current}) >= Target ({target_nbath}). No extension needed based on size.")
            self.log.info("="*65)
            return []
            
        self.log.info(f"Target: Recover {n_needed} orbitals from Frozen space to match reference total size.")

        # 2. 构造当前环境块的 AO 轨道
        env_loc_indices = slice(nimp_current, nimp_current + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        # 3. 计算重叠矩阵 O = C_env^T S C_ref
        S = self.mol.intor_symmetric('int1e_ovlp')
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S, ref_coeff))

        # 4. SVD 分解：ovlp_mat = U Σ Vh
        # sigma 已经从大到小排列，代表与参考空间的重叠程度
        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)

        recommended_indices = []
        used_env = set()
        num_modes_to_check = len(sigma)

        for i in range(num_modes_to_check):
            if len(recommended_indices) >= n_needed:
                break  # 已经找够了保证维度一致所需的轨道数

            s = sigma[i]
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
            
            # 判断这个最佳匹配的轨道在当前结构中被分配到了哪里
            status = ""
            if best_env_idx < nbath_current:
                # 如果最佳匹配是当前的 Bath，说明不需要捞回，它本就在 Bath 中
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                # 掉进了 Frozen Occupied
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                # 掉进了 Frozen Virtual
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices
    def find_bath_indices_from_reference_svd2(self, ref_coeff, ref_mol, threshold=None):
        """
        使用 SVD 进行全局最优匹配（适合大几何变化和 Imp 空间改变）。
        可以将新结构的全部环境轨道与旧结构的整个 EO (Imp+Bath) 空间进行对比。
        通过挑选与旧结构 EO 重叠度最高的 Frozen 轨道加入 Bath，
        自动补齐 Bath 使新结构总 EO 维数与旧结构总 EO 维数保持一致。
        
        Args:
            ref_coeff (ndarray): 参考构型的轨道系数（如整个旧 EO 空间，AO 基）。
            threshold (float): (保留参数以兼容旧代码)。
        Returns:
            list: 推荐追加到 Bath 的环境轨道索引（可直接传给 append_bath_by_env_idx）。
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp_current = len(self.imp_idx)
        nbath_current = self.nes - nimp_current
        
        # 1. 确定目标数目
        # 参考体系传入的轨道总数目 (通常是旧体系的 n_imp + n_bath)
        n_ref = ref_coeff.shape[1]
        # 为了保证 EO 空间总数一致：new_nimp + new_nbath = n_ref
        target_nbath = n_ref - nimp_current
        n_needed = target_nbath - nbath_current
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search (Full EO Match) {'='*20}")
        self.log.info(f"Reference Target Size: {n_ref} | Current Imp Size: {nimp_current}")
        self.log.info(f"Current Bath Size:   {nbath_current} | Target Bath Size: {target_nbath}")
        
        if n_needed <= 0:
            self.log.info(f"Current Bath size ({nbath_current}) >= Target ({target_nbath}). No extension needed based on size.")
            self.log.info("="*65)
            return []
            
        self.log.info(f"Target: Recover {n_needed} orbitals from Frozen space to match reference total size.")

        # 2. 构造当前环境块的 AO 轨道
        env_loc_indices = slice(nimp_current, nimp_current + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        # 3. 计算重叠矩阵 O = C_env^T S C_ref
        S_new = self.mol.intor_symmetric('int1e_ovlp')
        S_old = ref_mol.intor_symmetric('int1e_ovlp')
        _, S_new_half = lowdin_orth(self.mol)
        _, S_old_half = lowdin_orth(ref_mol)
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_coeff))

        # 4. SVD 分解：ovlp_mat = U Σ Vh
        # sigma 已经从大到小排列，代表与参考空间的重叠程度
        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)

        recommended_indices = []
        used_env = set()
        num_modes_to_check = len(sigma)

        for i in range(num_modes_to_check):
            if len(recommended_indices) >= n_needed:
                break  # 已经找够了保证维度一致所需的轨道数

            s = sigma[i]
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
            
            # 判断这个最佳匹配的轨道在当前结构中被分配到了哪里
            status = ""
            if best_env_idx < nbath_current:
                # 如果最佳匹配是当前的 Bath，说明不需要捞回，它本就在 Bath 中
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                # 掉进了 Frozen Occupied
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                # 掉进了 Frozen Virtual
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices


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
        self.calc_fo_ene()
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
    def mp2_solver(self):
        from pyscf import mp
        if self.es_mf is None:
            raise RuntimeError('embedded subspace is not built yet, please run build() first')
        self.log.info('Running MP2 for EO space')
        self.log.info('='*60)

        mymp2 = mp.MP2(self.es_mf)
        mymp2.verbose = self.verbose
        mymp2.max_memory = self.max_mem
        mymp2.kernel()
        e_tot_mp2 = mymp2.e_tot + self.fo_ene
        self.log.info('MP2 correlation energy = %.12f', mymp2.e_corr)
        self.log.info('Total MP2 energy = %.12f', e_tot_mp2)
        return e_tot_mp2, mymp2.e_corr
    # 20260129 new CCSD(T)
    def ccsdt_solver(self, with_t=True):
        " run ccsd(t) as the solver for dmet subspace "
        from pyscf import cc
        # [Modified] Removed the spin check restriction. PySCF cc.CCSD supports ROHF.
        # if self.mol.spin != 0:
        #    raise NotImplementedError('CCSD solver for open-shell system is not implemented yet')
        
        if self.es_mf is None:
            raise RuntimeError('embedded subspace is not built yet, please run build() first')
        
        self.log.info('Running CCSD solver for embedded cluster ...')
        self.log.info('=' * 60)
        
        # CCSD natively supports RHF and ROHF reference
        mycc = cc.CCSD(self.es_mf)
        mycc.verbose = self.verbose
        mycc.max_memory = self.max_mem
        
        # [Feature] Check for multireference issues in T1 amplitudes if needed
        # mycc.callback = ... 
        
        mycc.kernel()
        
        # --- Energy Formula Analysis ---
        # Formula 1 (Direct Sum): E_tot = E_frag_corr + E_frag_MF + E_frozen_MF
        # This is what existing code did (e_tot_ccsd = mycc.e_tot + self.fo_ene)
        # However, mycc.e_tot includes the nuclear repulsion of the fragment (if defined in es_mf.mol)
        # Usually es_mf is built with mol.nelectron reduced, but H_core includes environment potentials.
        
        # Formula 2 (Correlation Correction): E_tot = E_Global_MF + E_frag_corr
        # This is often MORE robust for PES / Energy differences because systematic errors cancel out.
        
        global_mf_energy = self.mf_or_cas.e_tot
        frag_corr_energy = mycc.e_corr
        
        e_tot_direct = mycc.e_tot + self.fo_ene
        e_tot_correction_based = global_mf_energy + frag_corr_energy
        
        self.log.info(f'Global MF Energy      = %.12f', global_mf_energy)
        self.log.info(f'Frozen Env Energy     = %.12f', self.fo_ene)
        self.log.info(f'Emb CCSD Corr Energy  = %.12f', mycc.e_corr)
        self.log.info(f'Total Energy (Direct) = %.12f (Sensitive to bath size)', e_tot_direct)
        self.log.info(f'Total Energy (Corr)   = %.12f (Recommended for PES)', e_tot_correction_based)

        # --- Electron Number Check ---
        # Check if electrons are leaking from the fragment
        try:
            rdm1 = mycc.make_rdm1()
            # The impurity orbitals are the first `nimp` orbitals in the embedded space
            nimp = len(self.imp_idx)
            nel_frag = np.trace(rdm1[:nimp, :nimp])
            self.log.info(f'Fragment Electron Number (Impurity Trace) = %.4f', nel_frag)
            
            # Warn if significant deviation from integer (optional logic, depends on system)
            # if abs(nel_frag - round(nel_frag)) > 0.1:
            #     self.log.warn("Warning: Significant charge fluctuation in fragment!")
        except Exception as e:
            self.log.warn(f"Could not calculate properties: {e}")

        # Determine which energy to return. 
        # Usually keeping the original return format is safer for compatibility, 
        # but for your research, e_tot_correction_based is likely better.
        # Here I return the Direct one to maintain compatibility, but log both.
        
        final_e_tot = e_tot_direct 

        if with_t:
            et = mycc.ccsd_t()
            e_tot_ccsd_t = final_e_tot + et
            e_tot_corr_t = e_tot_correction_based + et
            
            self.log.info('CCSD(T) correlation energy = %.12f', mycc.e_corr + et)
            self.log.info('Total CCSD(T) Energy (Direct) = %.12f', e_tot_ccsd_t)
            self.log.info('Total CCSD(T) Energy (Corr)   = %.12f', e_tot_corr_t)
            return mycc, e_tot_ccsd_t
            
        return final_e_tot
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
