import numpy as np
from scipy.linalg import null_space

from pyscf.lo.orth import lowdin
from pyscf.lib import logger
from pyscf import dft, lib

from embed_sim import ssdmet, spin_utils
from functools import reduce

class SPADECL(ssdmet.SSDMET):
    """
    SPADE+CL
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-12, es_natorb=True, bath_option=None, bath_norb=None, readmp2=False, bath_core_cutoff=0.5, verbose=logger.INFO):
        ssdmet.SSDMET.__init__(self, mf_or_cas=mf_or_cas,title=title,imp_idx=imp_idx, threshold=threshold, es_natorb=es_natorb, bath_option=bath_option, bath_norb=bath_norb, readmp2=readmp2, bath_core_cutoff=bath_core_cutoff, verbose=verbose)

    def build(self, chk_fname_load='', save_chk=True, xc='hf'):
        self.dump_flags()
        dm = ssdmet.mf_or_cas_make_rdm1s(self.mf_or_cas)
        if dm.ndim == 3: # ROHF density matrix have dimension (2, nao, nao)
            self.dm = dm[0] + dm[1]
            open_shell = True
        else:
            self.dm = dm
            open_shell = False

        loaded = self.load_chk(chk_fname_load)
        
        if not loaded:
            nimp = len(self.imp_idx)

            mo_coeff = self.mf_or_cas.mo_coeff
            mo_occ = self.mf_or_cas.mo_occ
            nv = np.sum(mo_occ <  self.threshold)
            na = np.sum((mo_occ >= self.threshold) & (mo_occ <= 2-self.threshold))
            no = np.sum(mo_occ > 2-self.threshold)
            mo_coeff_v = mo_coeff[:, no+na:nv+no+na]
            mo_coeff_a = mo_coeff[:, no:no+na]
            mo_coeff_o = mo_coeff[:, :no]

            s = self.mol.intor_symmetric('int1e_ovlp')
            mo_coeff_o_A = (lowdin(s) @ s)[self.imp_idx,:] @ mo_coeff_o
            U, Sigma, Vh = np.linalg.svd(mo_coeff_o_A, full_matrices=False)
            delta_Sigma = [Sigma[i] - Sigma[i+1] for i in range(len(Sigma)-1)]
            n_act = np.argpartition(delta_Sigma, -1)[-1] + 1
            mo_coeff_o_SPADE = mo_coeff_o @ Vh.T.conj()[:,:n_act]
            mo_coeff_fo_SPADE = mo_coeff_o @ null_space(Vh[:n_act,:])
            # B0 = mo_coeff_o.T.conj() @ s[:, self.imp_idx] @ np.linalg.inv(s[self.imp_idx, :][:, self.imp_idx]) @ s[self.imp_idx, :] @ mo_coeff_o
            # U, Sigma, Vh = np.linalg.svd(B0)
            # delta_Sigma = [Sigma[i] - Sigma[i+1] for i in range(len(Sigma)-1)]
            # n_act = np.argpartition(delta_Sigma, -1)[-1] + 1
            # mo_coeff_o_SPADE = mo_coeff_o @ Vh.T.conj()[:,:n_act]
            # mo_coeff_fo_SPADE = mo_coeff_o @ Vh.T.conj()[:,n_act:]

            neso = mo_coeff_o_SPADE.shape[-1]
            nfo = no - neso

            A0 = mo_coeff_v.T.conj() @ s[:, self.imp_idx] @ np.linalg.inv(s[self.imp_idx, :][:, self.imp_idx]) @ s[self.imp_idx, :] @ mo_coeff_v
            U, Sigma, Vh = np.linalg.svd(A0)
            mo_coeff_v_CL0 = mo_coeff_v @ Vh.T.conj()[:,Sigma > self.threshold]
            mo_coeff_v_CL0_ker = mo_coeff_v @ Vh.T.conj()[:,Sigma <= self.threshold]

            mo_coeff_v_CL = mo_coeff_v_CL0.copy()
            mo_coeff_v_CLi = mo_coeff_v_CL0.copy()
            mo_coeff_v_CLi_ker = mo_coeff_v_CL0_ker.copy()
            hcore = self.mf_or_cas.get_hcore()
            for i in range(self.bath_option+1):
                if i >= 1:
                    mo_coeff_v_CLisub1 = mo_coeff_v_CLi.copy()
                    mo_coeff_v_CLisub1_ker = mo_coeff_v_CLi_ker.copy()
                    Ai = mo_coeff_v_CLisub1.T.conj() @ hcore @ mo_coeff_v_CLisub1_ker
                    U, Sigma, Vh = np.linalg.svd(Ai, full_matrices=False)
                    mo_coeff_v_CLi = mo_coeff_v_CLisub1_ker @ Vh.T.conj()
                    mo_coeff_v_CLi_ker = mo_coeff_v_CLisub1_ker @ null_space(Vh)
                    mo_coeff_v_CL = np.hstack([mo_coeff_v_CL, mo_coeff_v_CLi])

            nesv = mo_coeff_v_CL.shape[-1]
            nfv = mo_coeff_v_CLi_ker.shape[-1]

            self.fo_orb = mo_coeff_fo_SPADE
            self.fv_orb = mo_coeff_v_CLi_ker
            self.es_orb = np.hstack([mo_coeff_o_SPADE, mo_coeff_a, mo_coeff_v_CL])
        
            self.nfo = nfo
            self.nfv = nfv
            self.nes = neso + na + nesv

            nesactive_a, nesactive_b = spin_utils.unpack_nelec(round(np.sum(mo_occ[no:no+na])), self.mol.spin)
            es_occactive = np.zeros(na)
            es_occactive[:nesactive_a] += 1
            es_occactive[:nesactive_b] += 1
            self.es_occ = np.hstack([np.ones(neso)*2, es_occactive, np.zeros(nesv)])

            self.log.info(f'number of impurity orbitals = {nimp}')
            self.log.info(f'number of embedded cluster orbitals = {self.nes}')
            self.log.info(f'number of frozen occupied orbitals = {nfo}')
            self.log.info(f'number of frozen virtual orbitals = {nfv}')
            self.log.info(f'number of frozen orbitals = {nfo+nfv}')
            self.log.info(f'percentage of embedded cluster orbitals = {((self.nes)/self.mol.nao)*100:.2f}%%')
            self.log.info(f'percentage of frozen orbitals = {((nfo+nfv)/self.mol.nao)*100:.2f}%%')

            self.get_corr(xc=xc)
            self.es_int1e = self.make_es_int1e()
            self.es_int2e = self.make_es_int2e()

            self.es_dm = np.zeros((2, self.nes, self.nes))
            self.es_dm[0] = np.diag(np.int32(self.es_occ>1-1e-3))
            self.es_dm[1] = np.diag(np.int32(self.es_occ>2-1e-3))

        self.es_mf = self.ROHF()

        if save_chk:
            chk_fname_save = self.title + '_dmet_chk.h5'
            self.save_chk(chk_fname_save)
        return self.es_mf

    def get_corr(self, xc='hf'):
        # HF J/K from env frozen occupied orbital
        fo_dm = np.array([self.fo_orb @ self.fo_orb.T.conj(), 
                          self.fo_orb @ self.fo_orb.T.conj()])
        es_dm = np.array([self.es_orb @ np.diag(np.int32(self.es_occ>1-1e-3)) @ self.es_orb.T.conj(), 
                          self.es_orb @ np.diag(np.int32(self.es_occ>2-1e-3)) @ self.es_orb.T.conj()])

        dft_temp = dft.roks.ROKS(self.mf_or_cas.mol).density_fit().x2c()
        dft_temp.xc = xc
        hcore = dft_temp.get_hcore()

        j_ae = dft_temp.get_j(dm=fo_dm+es_dm)
        j_fo = dft_temp.get_j(dm=fo_dm)
        j_es = dft_temp.get_j(dm=es_dm)

        veff_ae = dft_temp.get_veff(dm=fo_dm+es_dm)
        veff_fo = dft_temp.get_veff(dm=fo_dm)
        veff_es = dft_temp.get_veff(dm=es_dm)

        exc_ae = veff_ae.exc
        exc_fo = veff_fo.exc
        exc_es = veff_es.exc

        exc_cross = exc_ae - exc_fo - exc_es
        print(exc_cross)
        ej_cross = (np.einsum('ij,ji', es_dm[0]+es_dm[1], j_fo[0]+j_fo[1]) +
                    np.einsum('ij,ji', fo_dm[0]+fo_dm[1], j_es[0]+j_es[1])) / 2 #
        e_cross = exc_cross + ej_cross

        e_fo = np.einsum('ij,ji', fo_dm[0]+fo_dm[1], hcore + (j_fo[0]+j_fo[1]) * 0.5) + exc_fo
        e_es = np.einsum('ij,ji', es_dm[0]+es_dm[1], hcore + (j_es[0]+j_es[1]) * 0.5) + exc_es
        e_ae = np.einsum('ij,ji', fo_dm[0]+fo_dm[1] + es_dm[0]+es_dm[1], hcore + (j_ae[0]+j_ae[1]) * 0.5) + exc_ae

        e_nuc = self.mol.energy_nuc()
        print(e_fo, e_es, e_ae, e_ae+e_nuc)

        self.es_dm0 = es_dm
        veff_emb = veff_ae - veff_es
        self.veff_emb = (veff_emb[0] + veff_emb[1]) * 0.5
        self.e_cross = e_cross
        self.e_nuc = e_nuc
        self.e_fo = e_fo

    def make_es_int1e(self):
        hcore = self.mf_or_cas.get_hcore()
        fock = hcore + self.veff_emb

        es_int1e = reduce(np.dot, (self.es_orb.T.conj(), fock, self.es_orb)) # AO to embedded space
        return es_int1e

    def get_e_mf(self):
        e_dm_es = self.es_mf.make_rdm1()
        hcore = self.mf_or_cas.get_hcore()
        hcore_es = reduce(np.dot, (self.es_orb.T.conj(), hcore, self.es_orb))
        veff = self.es_mf.get_veff(dm=e_dm_es)
        e_mf = np.einsum('ij,ji', e_dm_es[0]+e_dm_es[1], hcore_es) + (np.einsum('ij,ji', e_dm_es[0], veff[0]) + np.einsum('ij,ji', e_dm_es[1], veff[1])) * 0.5
        return e_mf
