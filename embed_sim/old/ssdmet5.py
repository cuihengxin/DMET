from tracemalloc import start

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
        U, S, Vh = np.linalg.svd(ldm_env_imp, full_matrices=False)
        
        # find bath orbitals
        bath_mask = S > thres
        orb_bath = U[:, bath_mask] 
        nbath = orb_bath.shape[1]
        
        P = np.eye(nenv) - orb_bath @ orb_bath.T.conj()
        
        w_proj, v_proj = np.linalg.eigh(P)
        frozen_basis_mask = w_proj > 0.5
        frozen_basis = v_proj[:, frozen_basis_mask]
        
        ldm_frozen_sub = reduce(np.dot, (frozen_basis.T.conj(), ldm_env, frozen_basis))
        w_sub, v_sub = np.linalg.eigh(ldm_frozen_sub)
        
        orb_frozen_LO = np.dot(frozen_basis, v_sub)
        
        fo_mask = w_sub > 2 - thres
        fv_mask = w_sub < thres
        
        orb_fo = orb_frozen_LO[:, fo_mask]
        orb_fv = orb_frozen_LO[:, fv_mask]
        
        orb_env = np.hstack([orb_bath, orb_fo, orb_fv])
        
        nfo = orb_fo.shape[1]
        nfv = orb_fv.shape[1]
        
        occ_env = np.zeros(nenv) 
        
    else:
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
            # keep both the per-spin density matrices and the summed density
            self.dm_pair = dm
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


    def append_bath_by_env_idx(self, env_idx):
        """
        append the bath orbital into the EO space from idx
        """
        if self.lo_cloes is None or self.caolo is None or self.cloao is None:
            raise RuntimeError("Embedded subspace not built or transforms not cached. Run build() first.")

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        indices_to_move = [] 
        
        for idx in env_idx:
            #already in EO space
            if idx < nbath:
                self.log.warn(f"Index {idx} is already in Bath (current nbath={nbath}), skipping.")
            else:                
                indices_to_move.append(idx - nbath)
        
        if not indices_to_move:
            self.log.warn("No valid Frozen orbitals selected to append.")
            return

        Q_emb = self.lo_cloes[:, :nimp+nbath]
        env_block = self.lo_cloes[:, nimp+nbath:] 

        n_shifted_fo = 0
        n_shifted_fv = 0
        
        for local_idx in indices_to_move:
            if local_idx < self.nfo:
                n_shifted_fo += 1
            else:
                n_shifted_fv += 1
        
        self.log.info(f"Appending Bath: Shifted {n_shifted_fo} from FO, {n_shifted_fv} from FV")

        mask_move = np.zeros(env_block.shape[1], dtype=bool)
        mask_move[indices_to_move] = True
        
        B_new_candidates = env_block[:, mask_move] 
        
        lo2New_bath, _ = np.linalg.qr(B_new_candidates)
        
        indices_all = np.arange(env_block.shape[1])
        indices_remain = indices_all[~mask_move]
        
        # differenciate the FO and FV idx
        idx_remain_fo = [i for i in indices_remain if i < self.nfo]
        idx_remain_fv = [i for i in indices_remain if i >= self.nfo]
        
        lo2New_core = env_block[:, idx_remain_fo]
        lo2New_vir  = env_block[:, idx_remain_fv]
        
        self.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_core, lo2New_vir])
        
        self.nes  = nimp + nbath + lo2New_bath.shape[1] 
        self.nfo -= n_shifted_fo
        self.nfv -= n_shifted_fv
        
        self.es_orb = lib.dot(self.caolo, self.lo_cloes[:, :self.nes])
        self.fo_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes : self.nes+self.nfo])
        self.fv_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes+self.nfo :])

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        dm_arg = self.dm_pair if (self.open_shell and self.dm_pair is not None) else self.dm
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, dm_arg)
        
        self.es_mf = self.ROHF()
        self.calc_fo_ene() 
        
        self.log.info(f"Bath appended. New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        self.log.info(f"Frozen Energy updated: {self.fo_ene:.6f}")        
        #self.log.info(f"Bath appended. New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        self.log.info(f'number of impurity orbitals = {nimp}')
        self.log.info(f'number of bath orbitals = {self.nes - nimp}')
        self.log.info(f'number of embedded cluster orbitals = {self.nes}')
        self.log.info(f'percentage of embedded cluster orbitals = {((self.nes)/self.mol.nao)*100:.2f}%%')
        self.log.info(f'percentage of frozen orbitals = {((self.nfo+self.nfv)/self.mol.nao)*100:.2f}%%')
    def analyze_bath_composition(self, threshold=0.1):
        if self.es_orb is None:
            self.log.warn("Embedded subspace not built.")
            return

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        bath_orb_coeff = self.es_orb[:, nimp:nimp+nbath]
        
        S = self.mol.intor_symmetric('int1e_ovlp')
        
        self.log.info(f"{'='*20} Bath Orbital Composition Analysis {'='*20}")
        
        total_atoms = [self.mol.atom_symbol(i) for i in range(self.mol.natm)]
        
        ao_labels_str = self.mol.ao_labels() 
        ao_labels_fmt = self.mol.ao_labels(fmt=None) 

        for ib in range(nbath):
            C = bath_orb_coeff[:, ib]
            # Mulliken population: P_mu = C_mu * (S @ C)_mu
            SC = np.dot(S, C)
            pop = C * SC 
            
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

            sorted_ao_idx = np.argsort(np.abs(pop))[::-1]
            orb_details = []
            
            detail_threshold = threshold 
            
            for idx in sorted_ao_idx:
                val = pop[idx]
                if abs(val) > detail_threshold:
                    lbl = ao_labels_str[idx].strip()
                    orb_details.append(f"{lbl}({val:.2f})")
            
            if orb_details:
                self.log.info(f"        [Detail]: {', '.join(orb_details)}")

        self.log.info("="*65)
    def mapping_ao(self, ref_coeff, ref_mol, threshold=0.4):
        """
        Args:
            ref_bath_coeff (ndarray): bath coeff of the reference geometry in AO basis
            ref_mol (Mole): reference geometry's Mole object (for computing overlap)
        Returns:
            list: idx for  append_bath_by_env_idx
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        n_ref_orb = ref_coeff.shape[1]
        #n_needed = n_ref_bath - nbath_current
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
        self.log.info(f"Reference Orbital Size: {n_ref_orb} | Current Bath Size: {nbath_current}")
        # get bath+env of new structure
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        #env_loc_indices = slice(nimp, nimp + nbath_current )
        #env_loc_indices = slice(nimp  + nbath_current, nimp  + nbath_current + self.nfo + self.nfv)

        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]
        S_ref = ref_mol.intor_symmetric('int1e_ovlp')
        S_ref_inv = np.linalg.inv(S_ref)
        ao_ovlp = gto.mole.intor_cross('int1e_ovlp', ref_mol, self.mol)
        env_orb_refAO = S_ref_inv @ ao_ovlp @ env_orb_AO
        # do SVD between env_orb_refAO and ref_coeff
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
            plt.savefig(f"{self.title}_overlap_matrix.png")
            plt.close()
            self.log.info(f"Saved overlap matrix heatmap to {self.title}_overlap_matrix.png")
        except ImportError:
            self.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
        
        self.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

        recommended_indices = []
        used_env = set()
        num_modes_to_check = min(len(sigma), n_ref_orb) # old version
        #num_modes_to_check = n_ref_bath # check n_ref_bath


        self.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

        for i in range(num_modes_to_check):
            s = sigma[i]
            if s < threshold:
                self.log.debug(f"Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
                continue
            weights = np.abs(U[:, i])
            sorted_env = np.argsort(weights)[::-1]
            
            best_env_idx = -1
            for idx in sorted_env:
                if idx not in used_env:
                    best_env_idx = idx
                    break
            
            if best_env_idx == -1:
                self.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            
            status = ""
            if best_env_idx < nbath_current:
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices
    def find_bath_indices_from_reference_svd(self, ref_bath_coeff, ref_mol):
        """
        Args:
            ref_bath_coeff (ndarray): bath coeff of the reference geometry in AO basis
            ref_mol (Mole): reference geometry's Mole object (for computing overlap)
        Returns:
            list: idx for  append_bath_by_env_idx
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        n_ref_bath = ref_bath_coeff.shape[1]
        #n_needed = n_ref_bath - nbath_current
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
        self.log.info(f"Reference Bath Size: {n_ref_bath} | Current Bath Size: {nbath_current}")
        # get bath+env of new structure
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        #  O = C_env^T S C_ref the S here use the lowdin orthogonalized ovlp
        #S_new = self.mol.intor_symmetric('int1e_ovlp')
        #S_old = ref_mol.intor_symmetric('int1e_ovlp')
        _, S_new_half = lowdin_orth(self.mol)
        _, S_old_half = lowdin_orth(ref_mol)
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_bath_coeff))
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
            plt.title("Overlap Matrix between Current Bath+Env and Reference Bath")
            plt.xlabel("Reference Bath Index")
            plt.ylabel("Current Bath+Env Index")
            plt.savefig(f"{self.title}_overlap_matrix.png")
            plt.close()
            self.log.info(f"Saved overlap matrix heatmap to {self.title}_overlap_matrix.png")
        except ImportError:
            self.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
        
        self.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")

        recommended_indices = []
        used_env = set()
        num_modes_to_check = min(len(sigma), n_ref_bath) # old version
        #num_modes_to_check = n_ref_bath # check n_ref_bath


        self.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

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
                self.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            
            status = ""
            if best_env_idx < nbath_current:
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices
    def find_bath_indices_from_reference_svd3(self, ref_eo_coeff, ref_mol, threshold=0.4):
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        n_ref_eo = ref_eo_coeff.shape[1]
        
        self.log.info(f"{'='*20} SVD Consistent Bath Search {'='*20}")
        self.log.info(f"Reference EO Size: {n_ref_eo} | Reference BATH Size: {n_ref_eo-nimp} | Current Bath Size: {nbath_current}")
        

        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]
        _, S_new_half = lowdin_orth(self.mol)
        _, S_old_half = lowdin_orth(ref_mol)
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_eo_coeff))
        #ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new, ref_bath_coeff))
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(np.abs(ovlp_mat), cmap="YlGnBu")
            plt.title("Overlap Matrix between Current Bath+Env and Reference Bath")
            plt.xlabel("Reference Bath Index")
            plt.ylabel("Current Bath+Env Index")
            plt.savefig(f"{self.title}_overlap_matrix.png")
            plt.close()
            self.log.info(f"Saved overlap matrix heatmap to {self.title}_overlap_matrix.png")
        except ImportError:
            self.log.warn("matplotlib or seaborn not installed, skipping heatmap.")

        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
        
        self.log.info(f"==> All Singular Values (sigma) for Reference Bath SVD: {np.round(sigma, 4)}")


        recommended_indices = []
        used_env = set()
        num_modes_to_check = min(len(sigma), n_ref_eo - nimp) # old version


        self.log.info(f"Will checking {num_modes_to_check} principal SVD modes for reference match.")

        for i in range(num_modes_to_check):
            s = sigma[i]
            if s < threshold:
                self.log.debug(f"Skipping mode {i} due to small sigma (σ={s:.4f} < {threshold})")
                continue
            weights = np.abs(U[:, i])
            sorted_env = np.argsort(weights)[::-1]
            
            best_env_idx = -1
            for idx in sorted_env:
                if idx not in used_env:
                    best_env_idx = idx
                    break
            
            if best_env_idx == -1:
                self.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            
            status = ""
            if best_env_idx < nbath_current:
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices

    def find_bath_indices_from_reference_svd2(self, ref_coeff, ref_mol):
        """
        keep the EO space consistent with reference geometry, to achieve smooth PES.
        """
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp_current = len(self.imp_idx)
        nbath_current = self.nes - nimp_current
        
        n_ref = ref_coeff.shape[1]
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

        env_loc_indices = slice(nimp_current, nimp_current + nbath_current + self.nfo + self.nfv)
        env_orb_AO = self.caolo @ self.lo_cloes[:, env_loc_indices]

        _, S_new_half = lowdin_orth(self.mol)
        _, S_old_half = lowdin_orth(ref_mol)
        print("env_orb_AO.T shape:", env_orb_AO.T.conj().shape)
        print("S_new_half shape:", S_new_half.shape)
        print("S_old_half shape:", S_old_half.shape)
        print("ref_bath_coeff shape:", ref_coeff.shape)
        ovlp_mat = reduce(lib.dot, (env_orb_AO.T.conj(), S_new_half, S_old_half, ref_coeff))

        U, sigma, Vh = np.linalg.svd(ovlp_mat, full_matrices=False)
        
        self.log.info(f"==> All Singular Values (sigma) for Reference Full EO SVD: {np.round(sigma, 4)}")

        recommended_indices = []
        used_env = set()
        num_modes_to_check = len(sigma)

        for i in range(num_modes_to_check):
            if len(recommended_indices) >= n_needed:
                break  # full size match achieved

            s = sigma[i]
            weights = np.abs(U[:, i])
            sorted_env = np.argsort(weights)[::-1]
            
            best_env_idx = -1
            for idx in sorted_env:
                if idx not in used_env:
                    best_env_idx = idx
                    break
            
            if best_env_idx == -1:
                self.log.warn(f"SVD pair {i}: No unique unused environment orbital found.")
                continue

            used_env.add(best_env_idx)
            
            status = ""
            if best_env_idx < nbath_current:
                status = "Match Current Bath (Skipped)"
            elif best_env_idx < nbath_current + self.nfo:
                status = f"Recover FO (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            else:
                status = f"Recover FV (idx {best_env_idx})"
                recommended_indices.append(best_env_idx)
            
            self.log.info(f"Mode {i} (σ={s:.4f}): matched env_idx={best_env_idx}, Status={status}")

        self.log.info(f"Indices recovered from frozen space: {recommended_indices}")
        self.log.info("="*65)
        return recommended_indices
    def concentric_localization(self, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6):
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first before localization.")
        if atoms_A is None or len(atoms_A) == 0:
            raise ValueError("atoms_A must be a non-empty list of atom indices")

        atoms_A = [int(i) for i in atoms_A]
        natm = self.mol.natm
        if any(i < 0 or i >= natm for i in atoms_A):
            raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

        # Build a fake molecule from selected atoms with a small projection basis.
        fake_mol = gto.Mole()
        fake_mol.verbose = self.verbose
        fake_mol.unit = 'Bohr'
        fake_mol.symmetry = False
        fake_mol.atom = [(self.mol.atom_symbol(i), self.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
        fake_mol.basis = proj_bas
        fake_mol.spin = 0
        fake_mol.charge = self.mol.charge # here may be error for the sake of open shell systems
        if fake_mol.nelectron % 2 != 0:
            fake_mol.spin = 1  
        fake_mol.build(False, False)
        print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
        print(f"Fake molecule: {fake_mol.atom}")
        #s_wb = self.mol.intor_symmetric('int1e_ovlp')
        s_pb = fake_mol.intor_symmetric('int1e_ovlp')
        s_cross = gto.intor_cross('int1e_ovlp', fake_mol, self.mol) 
        s_pb_inv = np.linalg.inv(s_pb)
        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        fv_AO   = self.caolo @ self.lo_cloes[:, nimp+nbath+self.nfo :]

        def svd(coeff, couple_op, coeff_ker):
            ovlp = coeff.T.conj() @ couple_op @ coeff_ker #  ovlp may use other types of coupling operators, not limited to h_core
            # MUST use full_matrices=True here to get the complete right singular vectors 

            U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
            self.log.info(f"Singular values for shell {i}: {sigma}")
            r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
            V_span  = Vh[:r, :].T.conj()
            V_ker = Vh[r:, :].T.conj()
            coeff_n1 = coeff_ker @ V_span
            coeff_ker1 = coeff_ker @ V_ker
            return coeff_n1, coeff_ker1
        c_fv_prime = s_pb_inv @ s_cross @ fv_AO
        # space sizes
        self.log.info(f"fv_AO shape: {fv_AO.shape}")
        self.log.info(f"s_cross shape: {s_cross.shape}")
        self.log.info(f"c_fv_prime shape: {c_fv_prime.shape}")

        U, sigma, Vh = np.linalg.svd(c_fv_prime.T.conj() @ s_cross @ fv_AO, full_matrices=True)
        r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        self.log.info(f"SVD on projector rank r0: {r0}, sigma size: {len(sigma)}")
        self.log.info(f"Singular values: {sigma}")
        V_span = Vh[:r0, :].T.conj()
        V_ker  = Vh[r0:, :].T.conj()
        C_0 = fv_AO @ V_span
        C_ker0 = fv_AO @ V_ker
        C_vir = []
        C_vir.append(C_0)
        C_ker = []
        C_ker.append(C_ker0)

        # pseudo-canonicalize the vir space by diagonalizing the Fock matrix in this subspace
        dm = self.mf_or_cas.make_rdm1()
        fock_ao = self.mf_or_cas.get_fock(dm=dm) 

        self.log.info(f"Shell 0: {C_0.shape[1]} vectors in vir space, {C_ker0.shape[1]} vectors in ker space.")
        for i in range(n_shell):
            if couple_op == 'hcore':
                couple_matrix = self.mf_or_cas.get_hcore()
                self.log.info(f"Using Hcore as coupling operator for shell {i+1}")
            elif couple_op == 'fock':
                couple_matrix = fock_ao
                self.log.info(f"Using Fock matrix as coupling operator for shell {i+1}")
            #elif couple_op == ''
            new_vir, new_ker = svd(C_vir[i], couple_matrix, C_ker[i])  
            C_vir.append(new_vir)
            C_ker.append(new_ker)
            print(f"Shell {i+1}: {new_vir.shape[1]} new vectors added to vir space, {new_ker.shape[1]} vectors remain in ker space.")
            print(f"======Shell {i+1} concentric localized======")
        C_vir_matrix = np.hstack(C_vir)
        
        # Export density cube files for each shell. Noting that the density is calculated consider ing the orbs are occupied just for visualization.
        if ele_density:
            try:
                from pyscf.tools import cubegen
                for idx, c_shell in enumerate(C_vir):
                    if c_shell.shape[1] > 0:
                        dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                        cube_name = f"{self.title}_shell_{idx}_density.cube"
                        self.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                        cubegen.density(self.mol, cube_name, dm_shell)
            except Exception as e:
                self.log.warn(f"Failed to export cube files: {e}")


        # Project Fock matrix into the vir subspace defined by C_vir_matrix
        # (N_vir, N_AO) @ (N_AO, N_AO) @ (N_AO, N_vir) -> (N_vir, N_vir)
        fock_sub = C_vir_matrix.T.conj() @ fock_ao @ C_vir_matrix

        # diagonalize the Fock matrix
        mo_energy, U = np.linalg.eigh(fock_sub)
        C_vir_canonical = C_vir_matrix @ U

        C_fv_new = C_ker[-1]
        fock_fv = C_fv_new.T.conj() @ fock_ao @ C_fv_new
        mo_energy_fv, U_fv = np.linalg.eigh(fock_fv)
        C_fv_canonical = C_fv_new @ U_fv

        Q_emb = self.lo_cloes[:, :nimp+nbath]
        Q_fo  = self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo]
        
        lo2New_bath = self.cloao @ C_vir_canonical
        lo2New_fv   = self.cloao @ C_fv_canonical
        
        self.lo_cloes = np.hstack([Q_emb, lo2New_bath, Q_fo, lo2New_fv])
        
        n_shifted_fv = lo2New_bath.shape[1]
        self.nes += n_shifted_fv
        self.nfv -= n_shifted_fv
        
        self.es_orb = lib.dot(self.caolo, self.lo_cloes[:, :self.nes])
        self.fo_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes : self.nes+self.nfo])
        self.fv_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes+self.nfo :])

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        dm_arg = self.dm_pair if (self.open_shell and self.dm_pair is not None) else self.dm
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, dm_arg)
        
        self.es_mf = self.ROHF()
        self.calc_fo_ene() 
        
        self.log.info(f"Concentric Shell appended. Added {n_shifted_fv} vir orbitals to bath.")
        self.log.info(f"New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        return C_vir_canonical
    def concentric_occ_spade(self, atoms_A, threshold=1e-6):
        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        c_fo_lo = self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo]
        c_fo_ao = self.caolo @ c_fo_lo
        ao_indices_A = []
        for ia in atoms_A:
            atom_id, atom_symbol, start, end = self.mol.aoslice_by_atom()[ia]
            ao_indices_A.extend(range(start, end))
        Q_A = np.zeros((self.mol.nao, self.mol.nao))
        for idx in ao_indices_A:
            Q_A[idx, idx] = 1.0
        
        c_fo_lo_A = Q_A @ c_fo_lo
        u_A, sigma_A, vh_A = np.linalg.svd(c_fo_lo_A, full_matrices=True)
        print(f"Shape of c_fo_lo_A: {c_fo_lo_A.shape}")
        print(f"Singular values for c_fo_lo_A: {sigma_A}")
        print(f"U_A shape: {u_A.shape}, vh_A shape: {vh_A.shape}")
        print(f"Shape of V_A: {vh_A.T.conj().shape}")
        C_spade = c_fo_lo @ vh_A.T.conj()
        print(f"Shape of C_spade: {C_spade.shape}")
        
    def concentric_occ(self, atoms_A, threshold=1e-6):
        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        mo_coeff = self.mf_or_cas.mo_coeff
        mo_occ = self.mf_or_cas.mo_occ
        if mo_coeff.ndim == 3:
            mo_coeff_spin = mo_coeff[0] if mo_occ[0].sum() > mo_occ[1].sum() else mo_coeff[0]
            mo_occ_spin = mo_occ[0]
            c_occ_ao = mo_coeff_spin[:, mo_occ_spin > 0]
        else:
            c_occ_ao = mo_coeff[:, mo_occ > 0]
            
        c_occ_lo = self.cloao @ c_occ_ao
        
        ao_indices_A = []
        for ia in atoms_A:
            atom_id, atom_symbol, start, end = self.mol.aoslice_by_atom()[ia]
            ao_indices_A.extend(range(start, end))
            
        Q_A = np.zeros((self.mol.nao, self.mol.nao))
        for idx in ao_indices_A:
            Q_A[idx, idx] = 1.0
            
        c_occ_lo_A = Q_A @ c_occ_lo
        u_A, sigma_A, vh_A = np.linalg.svd(c_occ_lo_A, full_matrices=False)
        print(f"Shape of c_occ_lo_A: {c_occ_lo_A.shape}")
        print(f"Singular values for c_occ_lo_A: {sigma_A}")
        delta_sigma = []
        for i in range(1, len(sigma_A)):
            delta_sigma.append(sigma_A[i-1] - sigma_A[i])
        print(f"Delta sigma: {delta_sigma}")
        max_delta_idx = np.argmax(delta_sigma) if delta_sigma else -1
        print(f"Max delta sigma index: {max_delta_idx}, Max delta sigma value: {delta_sigma[max_delta_idx] if max_delta_idx >= 0 else 'N/A'}")
        
        mask = sigma_A > threshold
        if not np.any(mask):
            self.log.info("No occupied orbitals selected above threshold.")
            return None
            
        c_target = c_occ_lo @ vh_A[mask, :].T.conj()
        
        c_fo_lo = self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo]
        
        ovlp_fo_target = c_fo_lo.T.conj() @ c_target
        u_M, sigma_M, vh_M = np.linalg.svd(ovlp_fo_target, full_matrices=True)
        
        move_mask = np.zeros(u_M.shape[1], dtype=bool)
        move_mask[:len(sigma_M)] = (sigma_M > threshold)
        
        n_shift = np.sum(move_mask)
        
        if n_shift == 0:
            self.log.info("The selected target orbitals are already completely inside Imp+Bath. No FO to move.")
            return None
            
        lo2New_bath = c_fo_lo @ u_M[:, move_mask]
        lo2New_fo   = c_fo_lo @ u_M[:, ~move_mask]
        
        Q_emb = self.lo_cloes[:, :nimp+nbath]
        Q_fv  = self.lo_cloes[:, nimp+nbath+self.nfo:]
        
        self.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_fo, Q_fv])
        
        self.nes += n_shift
        self.nfo -= n_shift
        
        self.es_orb = lib.dot(self.caolo, self.lo_cloes[:, :self.nes])
        self.fo_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes : self.nes+self.nfo])
        self.fv_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes+self.nfo :])

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        dm_arg = self.dm_pair if (self.open_shell and self.dm_pair is not None) else self.dm
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, dm_arg)
        
        self.es_mf = self.ROHF()
        self.calc_fo_ene() 
        
        self.log.info(f"Concentric OCC appended. Compared with Bath, recovered {n_shift} missing OCC orbitals from FO.")
        self.log.info(f"New sizes: NES={self.nes}, NFO={self.nfo}, NFV={self.nfv}")
        return lo2New_bath

    def concentric_occ_localization(self, proj_bas, n_shell, atoms_A, couple_op='hcore', ele_density = False, threshold=1e-6):
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first before localization.")
        if atoms_A is None or len(atoms_A) == 0:
            raise ValueError("atoms_A must be a non-empty list of atom indices")

        atoms_A = [int(i) for i in atoms_A]
        natm = self.mol.natm
        if any(i < 0 or i >= natm for i in atoms_A):
            raise ValueError(f"atoms_A out of range. Valid atom index: 0..{natm-1}")

        # Build a fake molecule from selected atoms with a small projection basis.
        fake_mol = gto.Mole()
        fake_mol.verbose = self.verbose
        fake_mol.unit = 'Bohr'
        fake_mol.symmetry = False
        fake_mol.atom = [(self.mol.atom_symbol(i), self.mol.atom_coord(i, unit='Bohr')) for i in atoms_A]
        fake_mol.basis = proj_bas
        fake_mol.spin = self.mol.spin
        fake_mol.charge = self.mol.charge # here may be error for the sake of open shell systems
        if fake_mol.nelectron % 2 != 0:
            fake_mol.spin = 1
        fake_mol.build(False, False)
        print(f"Fake molecule built with {fake_mol.natm} atoms and {fake_mol.nao} AOs.")
        
        s_pb = fake_mol.intor_symmetric('int1e_ovlp')
        s_cross = gto.intor_cross('int1e_ovlp', fake_mol, self.mol) 
        s_pb_inv = np.linalg.inv(s_pb)
        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        # NOTE: Here we operate on the frozen occupied (FO) orbitals instead of FV
        fo_AO   = self.caolo @ self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo]

        def svd(coeff, couple_op, coeff_ker):
            ovlp = coeff.T.conj() @ couple_op @ coeff_ker
            U, sigma, Vh = np.linalg.svd(ovlp, full_matrices=True)
            r = np.sum(sigma > threshold) if len(sigma) > 0 else 0
            V_span  = Vh[:r, :].T.conj()
            V_ker = Vh[r:, :].T.conj()
            coeff_n1 = coeff_ker @ V_span
            coeff_ker1 = coeff_ker @ V_ker
            return coeff_n1, coeff_ker1
            
        c_fo_prime = s_pb_inv @ s_cross @ fo_AO

        U, sigma, Vh = np.linalg.svd(c_fo_prime.T.conj() @ s_cross @ fo_AO, full_matrices=True)
        r0 = np.sum(sigma > threshold) if len(sigma) > 0 else 0
        V_span = Vh[:r0, :].T.conj()
        V_ker  = Vh[r0:, :].T.conj()
        C_0 = fo_AO @ V_span
        C_ker0 = fo_AO @ V_ker
        C_occ = [C_0]
        C_ker = [C_ker0]

        dm = self.mf_or_cas.make_rdm1()
        fock_ao = self.mf_or_cas.get_fock(dm=dm) 

        for i in range(n_shell):
            if couple_op == 'hcore':
                couple_matrix = self.mf_or_cas.get_hcore()
            elif couple_op == 'fock':
                couple_matrix = fock_ao
                
            new_occ, new_ker = svd(C_occ[i], couple_matrix, C_ker[i])  
            C_occ.append(new_occ)
            C_ker.append(new_ker)
            print(f"Shell {i+1}: {new_occ.shape[1]} new vectors added to occ space")
            
        if ele_density:
            try:
                from pyscf.tools import cubegen
                for idx, c_shell in enumerate(C_occ):
                    if c_shell.shape[1] > 0:
                        dm_shell = 2.0 * (c_shell @ c_shell.T.conj())
                        cube_name = f"{self.title}_shell_{idx}_density.cube"
                        self.log.info(f"Exporting electron density of Shell {idx} ({c_shell.shape[1]} orbitals) to {cube_name}")
                        cubegen.density(self.mol, cube_name, dm_shell)
            except Exception as e:
                self.log.warn(f"Failed to export cube files: {e}")

        C_occ_matrix = np.hstack(C_occ)
        
        fock_sub = C_occ_matrix.T.conj() @ fock_ao @ C_occ_matrix
        mo_energy, U = np.linalg.eigh(fock_sub)
        C_occ_canonical = C_occ_matrix @ U

        C_fo_new = C_ker[-1]
        fock_fo = C_fo_new.T.conj() @ fock_ao @ C_fo_new
        mo_energy_fo, U_fo = np.linalg.eigh(fock_fo)
        C_fo_canonical = C_fo_new @ U_fo

        Q_emb = self.lo_cloes[:, :nimp+nbath]
        Q_fv  = self.lo_cloes[:, nimp+nbath+self.nfo :]
        
        lo2New_bath = self.cloao @ C_occ_canonical
        lo2New_fo   = self.cloao @ C_fo_canonical
        
        # Reassemble logic for FO
        # Sequence: [Emb, Target_FO (shifted to bath), Remaining_FO, FV]
        self.lo_cloes = np.hstack([Q_emb, lo2New_bath, lo2New_fo, Q_fv])
        
        n_shifted_fo = lo2New_bath.shape[1]
        self.nes += n_shifted_fo
        self.nfo -= n_shifted_fo
        
        self.es_orb = lib.dot(self.caolo, self.lo_cloes[:, :self.nes])
        self.fo_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes : self.nes+self.nfo])
        self.fv_orb = lib.dot(self.caolo, self.lo_cloes[:, self.nes+self.nfo :])

        self.es_int1e = self.make_es_int1e()
        self.es_int2e = self.make_es_int2e()
        dm_arg = self.dm_pair if (self.open_shell and self.dm_pair is not None) else self.dm
        self.es_dm = self.make_es_dm(self.open_shell, self.lo_cloes[:, :self.nes], self.cloao, dm_arg)
        
        self.es_mf = self.ROHF()
        self.calc_fo_ene() 
        
        self.log.info(f"Concentric FO Shell appended. Added {n_shifted_fo} occ orbitals to bath.")
        return C_occ_canonical

    def localize_environment_spaces(self, method='boys'):
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first before localization.")            
        self.log.info(f"Performing {method.upper()} localization on Env subspaces (Bath, FO, FV)...")
        
        nimp = len(self.imp_idx)
        nbath = self.nes - nimp
        
        # MO coeff in AO bases
        bath_AO = self.caolo @ self.lo_cloes[:, nimp : nimp+nbath]
        fo_AO   = self.caolo @ self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo]
        fv_AO   = self.caolo @ self.lo_cloes[:, nimp+nbath+self.nfo :]
        # note that, bath space may have some occ orbitals, may be some issues.
        def localize_subspace(coeff_AO, name):
            if coeff_AO.shape[1] <= 1:
                return coeff_AO  
            try:
                if method.lower() == 'boys':
                    loc_obj = lo.Boys(self.mol, coeff_AO)
                elif method.lower() == 'pm':
                    loc_obj = lo.PipekMezey(self.mol, coeff_AO)
                elif method.lower() == 'er':
                    loc_obj = lo.EdmistonRuedenberg(self.mol, coeff_AO)
                else:
                    self.log.warn(f"Unknown localization method {method}, skipping localization for {name}.")
                    return coeff_AO
                    
                loc_obj.verbose = 0
                return loc_obj.kernel()
            except Exception as e:
                self.log.warn(f"Localization failed for {name} subspace using {method}: {str(e)}")
                return coeff_AO
        
        bath_loc_AO = localize_subspace(bath_AO, "Bath")
        fo_loc_AO   = localize_subspace(fo_AO, "FO")
        fv_loc_AO   = localize_subspace(fv_AO, "FV")
        # from AO basis to LO basis
        self.lo_cloes[:, nimp : nimp+nbath] = self.cloao @ bath_loc_AO
        self.lo_cloes[:, nimp+nbath : nimp+nbath+self.nfo] = self.cloao @ fo_loc_AO
        self.lo_cloes[:, nimp+nbath+self.nfo :] = self.cloao @ fv_loc_AO
        # and refresh the fo fv orbitals in AO basis
        self.es_orb = self.caolo @ self.lo_cloes[:, :self.nes]
        self.fo_orb = fo_loc_AO
        self.fv_orb = fv_loc_AO
        
        self.log.info("Environment subspaces localized successfully.")
        return 

    def iao_analysis(self):
        if self.lo_cloes is None:
            raise RuntimeError("Run build() first.")

        nimp = len(self.imp_idx)
        nbath_current = self.nes - nimp
        
        self.log.info(f" Current Bath Size: {nbath_current}")
        caoes = self.caolo @ self.lo_cloes
        # bath + fo + fv
        env_loc_indices = slice(nimp, nimp + nbath_current + self.nfo + self.nfv)
        env_orb_AO = caoes[:, env_loc_indices]
        # bath 
        bath_orb = caoes[:, nimp:nimp+nbath_current]
        fo_orb = caoes[:, nimp+nbath_current : nimp+nbath_current+self.nfo]
        fv_orb = caoes[:, nimp+nbath_current+self.nfo : nimp+nbath_current+self.nfo+self.nfv]
        imp_orb = self.caolo @ self.lo_cloes[:, :nimp]
        mo_occ  = self.mf_or_cas.mo_coeff[:, self.mf_or_cas.mo_occ>1e-3]
        a = lo.iao.iao(self.mol, mo_occ)
        a = lo.vec_lowdin(a, self.mol.intor_symmetric('int1e_ovlp'))
        mo_occ = reduce(np.dot, (a.T, self.mol.intor_symmetric('int1e_ovlp'), mo_occ))
        # still the RHF 
        dm = np.dot(mo_occ, mo_occ.T) * 2
        assert(abs(dm.trace() - self.mol.nelectron) < 1e-13)
        pmol = self.mol.copy()
        pmol.build(False, False, basis='minao')
        self.mf_or_cas.mulliken_pop(pmol, dm, s=np.eye(pmol.nao_nr()))
        # above for the whole system
        t_ao = self.mol.intor('int1e_kin')
        def get_orb_kinetic_energy(idx):
            c_i = self.mf_or_cas.mo_coeff[:, idx]
                #  < psi_i | T | psi_i > = C_i^T * T_ao * C_i
            t_mo = c_i.T @ t_ao @ c_i
            print(f"Kinetic Energy for MO index {idx} : {t_mo:.8f} Hartree")
            return t_mo

        def analyze_orb_iao(env_orb_AO, idx, a, mf, pmol):
            target_env = env_orb_AO[:, idx:idx+1]
            
            target_env_iao = reduce(np.dot, (a.T, mf.get_ovlp(), target_env))
            
            dm_target = np.dot(target_env_iao, target_env_iao.T)
            
            # IAO pop
            print(f"\n--- IAO Population for Environment/Bath Orbital index {idx} ---")
            pop, charge = mf.mulliken_pop(pmol, dm_target, s=np.eye(pmol.nao_nr()))
            return pop    
        

        kinetic_energy =[]
        iao_pop = []
        print("\nImpurity orbitals:")
        for idx in range(imp_orb.shape[1]):
            iao_pop.append(analyze_orb_iao(imp_orb, idx, a, self.mf_or_cas, pmol))
            kinetic_energy.append(get_orb_kinetic_energy(idx))
        #print(iao_pop)
        print(f"\nBath orbitals:")
        for idx in range(bath_orb.shape[1]):
            iao_pop.append(analyze_orb_iao(bath_orb, idx, a, self.mf_or_cas, pmol))
            kinetic_energy.append(get_orb_kinetic_energy(idx))
        print(f"\nFrozen occupied orbitals:")
        for idx in range(fo_orb.shape[1]):
            iao_pop.append(analyze_orb_iao(fo_orb, idx, a, self.mf_or_cas, pmol))
            kinetic_energy.append(get_orb_kinetic_energy(idx))
        print("\nFrozen virtual orbitals:")
        for idx in range(fv_orb.shape[1]):
            iao_pop.append(analyze_orb_iao(fv_orb, idx, a, self.mf_or_cas, pmol))
            kinetic_energy.append(get_orb_kinetic_energy(idx))
        return np.array(iao_pop), np.array(kinetic_energy)
        #ref.mf
        # this function is used to generate the IAO analysis of all the orbitals
        


    
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
        # check spin 
        

        self.log.info('Running MP2 for EO space')
        self.log.info('='*60)

        mymp2 = mp.MP2(self.es_mf)
        mymp2.verbose = self.verbose
        mymp2.max_memory = self.max_mem
        mymp2.kernel()
        e_tot_mp2 = mymp2.e_tot + self.fo_ene
        self.log.info('MP2 correlation energy = %.12f', mymp2.e_corr)
        self.log.info('Total MP2 energy = %.12f', e_tot_mp2)
        return e_tot_mp2
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
            return mycc, e_tot_ccsd_t
            
        return final_e_tot
    def export_molden(self, filename, mode='es'):
        from pyscf import tools
        if self.es_orb is None:
            self.log.warn("Embedded subspace not built. Cannot export orbitals. Run build() first.")
            return

        nimp = len(self.imp_idx)
        nbath = self.nes - nimp

        self.log.info(f'Exporting orbitals to {filename} (mode={mode})')
        
        if mode == 'es':
            self.log.info(f'  [1 - {nimp}]: Impurity Orbitals')
            self.log.info(f'  [{nimp+1} - {nimp+nbath}]: Bath Orbitals')
            tools.molden.from_mo(self.mol, filename, self.es_orb)
            
        elif mode == 'all':
            mo_coeff = np.hstack((self.fo_orb, self.es_orb, self.fv_orb))
            
            occ = np.zeros(mo_coeff.shape[1])
            occ[:self.nfo] = 2.0
            
            self.log.info(f'  [1 - {self.nfo}]: Frozen Occupied ')
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
