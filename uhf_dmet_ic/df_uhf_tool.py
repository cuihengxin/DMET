import numpy as np
import os
import h5py
from copy import copy
from functools import reduce

from pyscf import lib, df, mp, cc, gto
from pyscf.cc import dfuccsd
from pyscf.mp import dfump2
from pyscf.ao2mo.incore import _conc_mos
from pyscf.ao2mo import _ao2mo
from uhf_dmet_ic import uhf_tool


def _ao2mo_df(mycc, mo_coeff=None):
    """DF AO2MO for embedded UHF with mixed representations.

    - Use AO->MO (caomo) for DF 3-center/4-center integral transformation.
    - Use ES->MO (mo_coeff) for _common_init_ so Fock/mo_energy are evaluated
      in embedded-space representation.
    """
    print('Overwriting AO2MO for UHF embedding...density fitting')
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff

    # AO->MO coefficients are required by dfuccsd integral contractions.
    caomo = mycc._scf.get_caomo(mo_coeff)

    # dfuccsd._make_df_eris always calls _common_init_(..., mo_coeff). We need
    # _common_init_ to use ES->MO while keeping AO->MO for DF contractions.
    # Patch _common_init_ only within this call.
    orig_common_init = dfuccsd._ChemistsERIs._common_init_

    def _common_init_mixed(self_eris, mycc_inner, mo_coeff_inner=None):
        out = orig_common_init(self_eris, mycc_inner, mo_coeff)
        # Keep AO->MO for subsequent DF contractions in _make_df_eris.
        self_eris.mo_coeff = caomo
        return out

    dfuccsd._ChemistsERIs._common_init_ = _common_init_mixed
    try:
        eris = dfuccsd._make_df_eris(mycc, mo_coeff=caomo)
    finally:
        dfuccsd._ChemistsERIs._common_init_ = orig_common_init

    # Expose canonical ES->MO coefficients to downstream routines.
    eris.mo_coeff = mo_coeff
    return eris


def _ao2mo_df_mp2(mymp2, mo_coeff=None):
    """DF AO2MO for embedded UHF MP2 using AO->MO coefficients in DF transform."""
    print('Overwriting AO2MO for UHF embedding MP2...density fitting')
    if mo_coeff is None:
        mo_coeff = mymp2.mo_coeff
    caomo = mymp2._scf.get_caomo(mo_coeff)

    # dfump2._make_df_eris uses mymp2.mo_coeff internally for MO partitioning.
    # Temporarily switch to AO->MO for integral transformation and restore after.
    mo_coeff_ = mymp2.mo_coeff
    mymp2.mo_coeff = caomo
    try:
        eris = dfump2._make_df_eris(mymp2)
    finally:
        mymp2.mo_coeff = mo_coeff_
    return eris

def make_es_cderi(title, es_orb, with_df, spin_tag):
    erifile = f"{title}_es_cderi_{spin_tag}.h5"
    dataname = 'j3c'
    feri = df.outcore._create_h5file(erifile, dataname)
    ijmosym, nij_pair, moij, ijslice = _conc_mos(es_orb, es_orb, True)
    naux = with_df.get_naoaux()
    neo = es_orb.shape[-1]
    nao_pair = neo*(neo+1)//2
    label = '%s/%d'%(dataname, 0)
    feri[label] = np.zeros((naux,nao_pair),dtype=np.float64)
    nij = 0
    for eri1 in with_df.loop():
        Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym)
        nrow = Lij.shape[0]
        feri[label][nij:nij+nrow] = Lij
        nij += nrow
    return erifile

class DF_UHF_EMB(uhf_tool.UHF_EMB):
    def __init__(self, mol, mf, es_orb, fo_orb, es_dm, es_cderi):
        self.mf_emb = mf
        self.es_orb = es_orb
        self.fo_orb = fo_orb
        self.dm_emb = es_dm
        self.es_cderi = es_cderi
        super(uhf_tool.UHF_EMB, self).__init__(mol)  # skip UHF_EMB init to do it ourselves

        self.veff = self.get_veff()
        self.fock = self.get_fock()
        self.mo_energy, self.mo_coeff = self.eig(fock = self.fock)
        self.mo_occ = self.get_occ()

    def _get_jk(self, dm, es_orb):
        mo_a, mo_b = es_orb
        dm_a, dm_b = dm
        dm_ao_a = lib.einsum('pi,ij,qj->pq', mo_a, dm_a, mo_a)
        dm_ao_b = lib.einsum('pi,ij,qj->pq', mo_b, dm_b, mo_b)

        vj, vk = self.mf_emb.get_jk(self.mf_emb.mol, (dm_ao_a, dm_ao_b), hermi=1)
        vj_ao = vj[0] + vj[1]

        vj_mo_a = lib.einsum('pi,pq,qj->ij', mo_a, vj_ao, mo_a)
        vj_mo_b = lib.einsum('pi,pq,qj->ij', mo_b, vj_ao, mo_b)

        vk_mo_a = lib.einsum('pi,pq,qj->ij', mo_a, vk[0], mo_a)
        vk_mo_b = lib.einsum('pi,pq,qj->ij', mo_b, vk[1], mo_b)

        return (vj_mo_a, vj_mo_b), (vk_mo_a, vk_mo_b)

class DFSSDMET_uhf(uhf_tool.SSDMET_uhf):
    """
    Density fitting extension for single-shot UHF DMET.
    """
    def __init__(self, mf_or_cas, title='untitled', imp_idx=None, threshold=1e-8, bath_option=None, es_method='svd', with_df=None, verbose=lib.logger.INFO):
        super().__init__(mf_or_cas, title=title, imp_idx=imp_idx, threshold=threshold, bath_option=bath_option, es_method=es_method, verbose=verbose)
        self.with_df = with_df if with_df is not None else getattr(self.mf_or_cas, 'with_df', None)
        if self.with_df is None:
            self.with_df = df.DF(self.mol)
            self.with_df.build()
            
        self.es_cderi = None

    def make_es_cderi(self):
        cderi_a = make_es_cderi(self.title, self.es_orb[0], self.with_df, 'a')
        cderi_b = make_es_cderi(self.title, self.es_orb[1], self.with_df, 'b')
        return (cderi_a, cderi_b)

    def load_chk(self, chk_fname):
        try:
            if not '_df_uhf_chk.h5' in chk_fname:
                chk_fname = chk_fname + '_df_uhf_chk.h5'
            if not os.path.isfile(chk_fname):
                return False
        except:
            return False

        self.log.info(f'load chk file {chk_fname}')
        with h5py.File(chk_fname, 'r') as fh5:
            dm_check = (np.allclose(self.dm[0], fh5['dm_a'][:], atol=1e-5) and
                        np.allclose(self.dm[1], fh5['dm_b'][:], atol=1e-5))
            imp_idx_check = np.array_equal(np.array(self.imp_idx), fh5['imp_idx'][:])
            threshold_check = self.threshold == fh5['threshold'][()]
            if dm_check and imp_idx_check and threshold_check:
                self.es_orb = (fh5['es_orb_a'][:], fh5['es_orb_b'][:])
                self.fo_orb = (fh5['fo_orb_a'][:], fh5['fo_orb_b'][:])
                self.fv_orb = (fh5['fv_orb_a'][:], fh5['fv_orb_b'][:])
                self.es_dm = (fh5['es_dm_a'][:], fh5['es_dm_b'][:])
                self.nes = tuple(fh5['nes'][:])
                self.nfo = tuple(fh5['nfo'][:])
                self.es_cderi = (fh5['es_cderi_a'][()].decode(), fh5['es_cderi_b'][()].decode())
                return True
            else:
                self.log.info(f'density matrix check {dm_check}')
                self.log.info(f'impurity index check {imp_idx_check}')
                self.log.info(f'threshold check {threshold_check}')
                self.log.info(f'build df uhf embedding with imp idx {self.imp_idx} threshold {self.threshold}')
                return False

    def save_chk(self, chk_fname):
        if not '_df_uhf_chk.h5' in chk_fname:
            chk_fname = chk_fname + '_df_uhf_chk.h5'
        with h5py.File(chk_fname, 'w') as fh5:
            fh5['dm_a'] = self.dm[0]
            fh5['dm_b'] = self.dm[1]
            fh5['imp_idx'] = np.array(self.imp_idx)
            fh5['threshold'] = self.threshold
            fh5['es_orb_a'] = self.es_orb[0]
            fh5['es_orb_b'] = self.es_orb[1]
            fh5['fo_orb_a'] = self.fo_orb[0]
            fh5['fo_orb_b'] = self.fo_orb[1]
            fh5['fv_orb_a'] = self.fv_orb[0]
            fh5['fv_orb_b'] = self.fv_orb[1]
            fh5['es_dm_a'] = self.es_dm[0]
            fh5['es_dm_b'] = self.es_dm[1]
            fh5['nes'] = np.array(self.nes)
            fh5['nfo'] = np.array(self.nfo)
            fh5['es_cderi_a'] = np.string_(self.es_cderi[0])
            fh5['es_cderi_b'] = np.string_(self.es_cderi[1])

    def build(self, restore_imp = False, aodmet = False, chk_fname_load='', save_chk=True):
        if not hasattr(self.mf_or_cas, 'with_df') or self.mf_or_cas.with_df is None:
            self.mf_or_cas.with_df = self.with_df
        
        super().build(restore_imp=restore_imp, chk_fname_load=chk_fname_load, aodmet = aodmet, save_chk=save_chk)

        if save_chk:
            self.save_chk(self.title)

    def UHF(self):
        if getattr(self, 'es_cderi', None) is None:
            self.es_cderi = self.make_es_cderi()
            
        mol = gto.M()
        mol.max_memory = self.mf_or_cas.max_memory
        mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.nelec = (self.mol.nelec[0] - self.nfo[0], self.mol.nelec[1] - self.nfo[1])
        mol.build()
        mf_emb = DF_UHF_EMB(mol, self.mf_or_cas, self.es_orb, self.fo_orb, self.es_dm, self.es_cderi)
        mf_emb.max_memory = self.mf_or_cas.max_memory
        
        # X2C is propagated via UHF_EMB.__init__ -> with_x2c attribute;
        # the 1e Hamiltonian already carries x2c corrections through get_hcore().

        mf_emb.with_df = df.DF(mol)
        mf_emb.with_df._cderi = self.es_cderi[0]  # will need to overwrite _make_eris for postHF
        self.es_e = mf_emb.energy_elec()[0]
        return mf_emb

    def get_mf_for_post_hf(self):
        es_mf_post = copy(self.es_mf)
        # Keep ES->MO coefficients. The custom ao2mo in uhf_tool will do
        # AO<-ES and ES<-MO composition via get_caomo; pre-transforming to
        # AO->MO here would apply the AO<-ES map twice.
        es_mf_post.mo_coeff = (
            np.array(self.es_mf.mo_coeff[0], copy=True),
            np.array(self.es_mf.mo_coeff[1], copy=True),
        )
        return es_mf_post

    def uccsd(self):
        # DF-based UCCSD for embedded space.
        es_mf_post = self.get_mf_for_post_hf()
        mycc = cc.UCCSD(es_mf_post).density_fit()
        mycc.max_memory = getattr(self.mf_or_cas, 'max_memory', 4000)
        # Use full-system DF object in AO basis; _ao2mo_df handles AO->MO transformation.
        mycc.with_df = self.with_df
        e_hf_fixed = self.es_e
        mycc.get_e_hf = lambda mo_coeff=None: e_hf_fixed
        mycc.ao2mo = uhf_tool.types.MethodType(_ao2mo_df, mycc)
        return mycc

    def ump2(self):
        # DF-based UMP2 for embedded space.
        es_mf_post = self.get_mf_for_post_hf()
        # DFUMP2 does not implement iterative update_amps.
        # Ensure MP2.kernel takes the direct init_amps branch.
        es_mf_post.converged = True
        mymp2 = mp.UMP2(es_mf_post).density_fit()
        mymp2.max_memory = getattr(self.mf_or_cas, 'max_memory', 4000)
        mymp2.with_df = self.with_df
        e_hf_fixed = self.es_e
        mymp2.get_e_hf = lambda mo_coeff=None: e_hf_fixed
        mymp2.ao2mo = uhf_tool.types.MethodType(_ao2mo_df_mp2, mymp2)
        return mymp2

def density_fit(self, with_df=None):
    df_dmet = DFSSDMET_uhf(self.mf_or_cas, self.title, imp_idx=self.imp_idx, threshold=self.threshold,
                    bath_option=self.bath_option, es_method=self.es_method, with_df=with_df, verbose=self.verbose)
    df_dmet.__dict__.update(self.__dict__)
    return df_dmet

uhf_tool.SSDMET_uhf.density_fit = density_fit
