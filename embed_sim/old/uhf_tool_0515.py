import numpy as np
from scipy.linalg import eigh, svd, block_diag
import h5py
import os

import types
import time

from pyscf.lo.orth import lowdin
from pyscf import gto, scf, cc, ao2mo

"""
UHF Embedding Tools
"""
from pyscf import lib
from pyscf.lib import logger

def _ao2mo(self, mo_coeff):
    """
    Overwrite CCSD ao2mo.  Note that 4 copy of eris in ES is needed so most cases will be outcore.
    """
    nmoa, nmob = self.get_nmo() # If frozen set this already subtracts the frozen orbitals
    nao = self.mo_coeff[0].shape[0]
    nmo_pair = nmoa*(nmoa+1)//2
    nao_pair = nao*(nao+1)//2
    mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8 * 4/1e6
    mem_now = lib.current_memory()[0]
    is_incore = getattr(self, 'incore_complete', True)
    if mem_incore+mem_now < self.max_memory and is_incore:
        # Eris is calculated from AO -> MO 
        # C[AO->MO] = C[AO->ES] C[ES->MO](can use frozen)

        return _make_eris_incore(self, mo_coeff)
    else:
        return _make_eris_outcore(self, mo_coeff)

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    """
    Use directly the ERI(aa,bb,ab) to get ERI in mo for subsequent calculation.
    """
    print('Overwriting AO2MO for UHF embedding...incore')
    eris = cc.uccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    eri_aa, eri_bb, eri_ab = mycc._scf.get_eri(eris.mo_coeff)  # fix frozen in _common_init_

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eri_aa = None

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eri_bb = None

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)
    eri_ab = None
    assert(eri_ba is not None)

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    eri_ba = None

    ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
    eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
    eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

    OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
    eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
    eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

    ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
    eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
    vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
    idxa = np.tril_indices(nvira)
    idxb = np.tril_indices(nvirb)
    eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

    OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
    eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    """
    Outcore version for UHF embedding.
    """

    print('Overwriting AO2MO for UHF embedding...outcore')
    eris = cc.uccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    # ---- Get AO->MO coefficients from the embedded UHF ----
    caomo_a, caomo_b = mycc._scf.get_caomo(eris.mo_coeff)
    mol = mycc._scf.mf_emb.mol

    orboa = caomo_a[:, :nocca]
    orbob = caomo_b[:, :noccb]
    orbva = caomo_a[:, nocca:]
    orbvb = caomo_b[:, noccb:]

    # ---- Create HDF5 temp file for outcore storage ----
    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocca,nocca,nocca,nocca), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocca,nvira,nocca,nocca), 'f8')
    eris.ovov = eris.feri.create_dataset('ovov', (nocca,nvira,nocca,nvira), 'f8')
    eris.oovv = eris.feri.create_dataset('oovv', (nocca,nocca,nvira,nvira), 'f8')
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocca,nvira,nvira,nocca), 'f8')
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocca,nvira,nvira*(nvira+1)//2), 'f8')
    eris.OOOO = eris.feri.create_dataset('OOOO', (noccb,noccb,noccb,noccb), 'f8')
    eris.OVOO = eris.feri.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), 'f8')
    eris.OVOV = eris.feri.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), 'f8')
    eris.OOVV = eris.feri.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), 'f8')
    eris.OVVO = eris.feri.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), 'f8')
    eris.OVVV = eris.feri.create_dataset('OVVV', (noccb,nvirb,nvirb*(nvirb+1)//2), 'f8')
    eris.ooOO = eris.feri.create_dataset('ooOO', (nocca,nocca,noccb,noccb), 'f8')
    eris.ovOO = eris.feri.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8')
    eris.ovOV = eris.feri.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), 'f8')
    eris.ooVV = eris.feri.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), 'f8')
    eris.ovVO = eris.feri.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), 'f8')
    eris.ovVV = eris.feri.create_dataset('ovVV', (nocca,nvira,nvirb*(nvirb+1)//2), 'f8')
    eris.OVoo = eris.feri.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8')
    eris.OOvv = eris.feri.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8')
    eris.OVvo = eris.feri.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), 'f8')
    eris.OVvv = eris.feri.create_dataset('OVvv', (noccb,nvirb,nvira*(nvira+1)//2), 'f8')

    tmpf = lib.H5TmpFile()

    def _load_and_slice(tmpf_key, nocc_i, nmo_bra, nmo_ket):
        """Load a block from tmpf, handling both compact and non-compact storage."""
        data = tmpf[tmpf_key]
        ncol = data.shape[1]
        compact_ncol = nmo_ket * (nmo_ket + 1) // 2
        is_compact = (ncol == compact_ncol)
        return data, is_compact

    def _get_buf(data, i, nmo_bra, nmo_ket, is_compact):
        """Get a (nmo_bra, nmo_ket, nmo_ket) buffer for the i-th occupied orbital."""
        row = np.asarray(data[i*nmo_bra:(i+1)*nmo_bra])
        if is_compact:
            buf = np.empty((nmo_bra, nmo_ket, nmo_ket))
            lib.unpack_tril(row, out=buf)
        else:
            buf = row.reshape(nmo_bra, nmo_ket, nmo_ket)
        return buf

    # ---- aa block: (oa, moa | moa, moa) ----
    if nocca > 0:
        ao2mo.general(mol, (orboa, caomo_a, caomo_a, caomo_a), tmpf, 'aa')
        data_aa, compact_aa = _load_and_slice('aa', nocca, nmoa, nmoa)
        for i in range(nocca):
            buf = _get_buf(data_aa, i, nmoa, nmoa, compact_aa)
            eris.oooo[i] = buf[:nocca, :nocca, :nocca]
            eris.ovoo[i] = buf[nocca:, :nocca, :nocca]
            eris.ovov[i] = buf[nocca:, :nocca, nocca:]
            eris.oovv[i] = buf[:nocca, nocca:, nocca:]
            eris.ovvo[i] = buf[nocca:, nocca:, :nocca]
            eris.ovvv[i] = lib.pack_tril(buf[nocca:, nocca:, nocca:])
        del tmpf['aa']

    # ---- bb block: (ob, mob | mob, mob) ----
    if noccb > 0:
        ao2mo.general(mol, (orbob, caomo_b, caomo_b, caomo_b), tmpf, 'bb')
        data_bb, compact_bb = _load_and_slice('bb', noccb, nmob, nmob)
        for i in range(noccb):
            buf = _get_buf(data_bb, i, nmob, nmob, compact_bb)
            eris.OOOO[i] = buf[:noccb, :noccb, :noccb]
            eris.OVOO[i] = buf[noccb:, :noccb, :noccb]
            eris.OVOV[i] = buf[noccb:, :noccb, noccb:]
            eris.OOVV[i] = buf[:noccb, noccb:, noccb:]
            eris.OVVO[i] = buf[noccb:, noccb:, :noccb]
            eris.OVVV[i] = lib.pack_tril(buf[noccb:, noccb:, noccb:])
        del tmpf['bb']

    # ---- ab block: (oa, moa | mob, mob) ----
    if nocca > 0:
        ao2mo.general(mol, (orboa, caomo_a, caomo_b, caomo_b), tmpf, 'ab')
        data_ab, compact_ab = _load_and_slice('ab', nocca, nmoa, nmob)
        for i in range(nocca):
            buf = _get_buf(data_ab, i, nmoa, nmob, compact_ab)
            eris.ooOO[i] = buf[:nocca, :noccb, :noccb]
            eris.ovOO[i] = buf[nocca:, :noccb, :noccb]
            eris.ovOV[i] = buf[nocca:, :noccb, noccb:]
            eris.ooVV[i] = buf[:nocca, noccb:, noccb:]
            eris.ovVO[i] = buf[nocca:, noccb:, :noccb]
            eris.ovVV[i] = lib.pack_tril(buf[nocca:, noccb:, noccb:])
        del tmpf['ab']

    # ---- ba block: (ob, mob | moa, moa) ----
    if noccb > 0:
        ao2mo.general(mol, (orbob, caomo_b, caomo_a, caomo_a), tmpf, 'ba')
        data_ba, compact_ba = _load_and_slice('ba', noccb, nmob, nmoa)
        for i in range(noccb):
            buf = _get_buf(data_ba, i, nmob, nmoa, compact_ba)
            eris.OVoo[i] = buf[noccb:, :nocca, :nocca]
            eris.OOvv[i] = buf[:noccb, nocca:, nocca:]
            eris.OVvo[i] = buf[noccb:, nocca:, :nocca]
            eris.OVvv[i] = lib.pack_tril(buf[noccb:, nocca:, nocca:])
        del tmpf['ba']
    buf = None

    # ---- vvvv blocks ----
    if not mycc.direct:
        ao2mo.full(mol, orbva, eris.feri, dataname='vvvv')
        ao2mo.full(mol, orbvb, eris.feri, dataname='VVVV')
        ao2mo.general(mol, (orbva, orbva, orbvb, orbvb),
                      eris.feri, dataname='vvVV')
        eris.vvvv = eris.feri['vvvv']
        eris.VVVV = eris.feri['VVVV']
        eris.vvVV = eris.feri['vvVV']

    return eris


def _ao2mo_mp2(self, mo_coeff):
    """
    Overwrite UMP2 ao2mo for UHF embedding.
    UMP2 only needs ovov, ovOV, OVOV blocks (Chemists' notation (ia|jb)).
    """
    nmoa, nmob = self.get_nmo()
    nao = self.mo_coeff[0].shape[0]
    nmo_pair_a = nmoa * (nmoa + 1) // 2
    nao_pair = nao * (nao + 1) // 2
    mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair_a**2) * 8 * 4 / 1e6
    mem_now = lib.current_memory()[0]
    is_incore = getattr(self, 'incore_complete', True)
    if mem_incore + mem_now < self.max_memory and is_incore:
        return _make_mp2_eris_incore(self, mo_coeff)
    else:
        return _make_mp2_eris_outcore(self, mo_coeff)


def _make_mp2_eris_incore(mp, mo_coeff=None):
    """
    In-core ERI transformation for UMP2 embedding.
    Only computes ovov, ovOV, OVOV blocks.
    """
    from pyscf.mp import ump2
    print('Overwriting AO2MO for UHF embedding MP2...incore')
    eris = ump2._ChemistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb

    caomo_a, caomo_b = mp._scf.get_caomo(eris.mo_coeff)
    mol = mp._scf.mf_emb.mol

    orboa = caomo_a[:, :nocca]
    orbob = caomo_b[:, :noccb]
    orbva = caomo_a[:, nocca:]
    orbvb = caomo_b[:, noccb:]

    if nocca * nvira > 0:
        eris.ovov = ao2mo.general(mol, (orboa, orbva, orboa, orbva))
        eris.ovov = eris.ovov.reshape(nocca * nvira, nocca * nvira)
    else:
        eris.ovov = np.zeros((nocca * nvira, nocca * nvira))

    if nocca * nvira * noccb * nvirb > 0:
        eris.ovOV = ao2mo.general(mol, (orboa, orbva, orbob, orbvb))
        eris.ovOV = eris.ovOV.reshape(nocca * nvira, noccb * nvirb)
    else:
        eris.ovOV = np.zeros((nocca * nvira, noccb * nvirb))

    if noccb * nvirb > 0:
        eris.OVOV = ao2mo.general(mol, (orbob, orbvb, orbob, orbvb))
        eris.OVOV = eris.OVOV.reshape(noccb * nvirb, noccb * nvirb)
    else:
        eris.OVOV = np.zeros((noccb * nvirb, noccb * nvirb))

    return eris


def _make_mp2_eris_outcore(mp, mo_coeff=None):
    """
    Outcore ERI transformation for UMP2 embedding.
    Only computes ovov, ovOV, OVOV blocks (Chemists' notation (ia|jb)).
    Stores flat 2D arrays in HDF5.
    """
    from pyscf.mp import ump2
    print('Overwriting AO2MO for UHF embedding MP2...outcore')
    eris = ump2._ChemistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa - nocca, nmob - noccb

    caomo_a, caomo_b = mp._scf.get_caomo(eris.mo_coeff)
    mol = mp._scf.mf_emb.mol

    orboa = caomo_a[:, :nocca]
    orbob = caomo_b[:, :noccb]
    orbva = caomo_a[:, nocca:]
    orbvb = caomo_b[:, noccb:]

    eris.feri = lib.H5TmpFile()

    if nocca * nvira > 0:
        ao2mo.general(mol, (orboa, orbva, orboa, orbva),
                       eris.feri, dataname='ovov', max_memory=max(2000, mp.max_memory))
        eris.ovov = eris.feri['ovov']
    else:
        eris.ovov = np.zeros((nocca * nvira, nocca * nvira))

    if nocca * nvira * noccb * nvirb > 0:
        ao2mo.general(mol, (orboa, orbva, orbob, orbvb),
                       eris.feri, dataname='ovOV', max_memory=max(2000, mp.max_memory))
        eris.ovOV = eris.feri['ovOV']
    else:
        eris.ovOV = np.zeros((nocca * nvira, noccb * nvirb))

    if noccb * nvirb > 0:
        ao2mo.general(mol, (orbob, orbvb, orbob, orbvb),
                       eris.feri, dataname='OVOV', max_memory=max(2000, mp.max_memory))
        eris.OVOV = eris.feri['OVOV']
    else:
        eris.OVOV = np.zeros((noccb * nvirb, noccb * nvirb))

    return eris


class SSDMET_uhf:
    """
    UHF embedding
        bath generation
            svd : if no threshold will add in small eigenvalues 1e-16/1e-17 poisoning results
                  can set higher thres as 1e-6/1e-7
            eig : recommended
            
        Embedded MeanFieldObject
            self written version for different embedding space size.

        Embedded Space High Level Solver
            self written ccsd class overwriting AO2MO
    """
    def __init__(self,mf_or_cas,title='untitled',imp_idx=None, threshold=1e-8, bath_option=None, es_method='svd',verbose=logger.INFO):
        self.mf_or_cas = mf_or_cas
        self.mol = self.mf_or_cas.mol
        self.title = title
        self.max_mem = mf_or_cas.max_memory 
        self.verbose = verbose 
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
        self.bath_option = bath_option
        if es_method not in ['svd', 'eig']:
            raise ValueError(f"es_method must be 'svd' or 'eig', got {es_method}")
        self.es_method = es_method

        # NOT inputs
        self.es_orb = None
        self.fo_orb = None
        self.fv_orb = None
        self.nes    = None  # tuple (nes_a, nes_b) for UHF
        self.nfo    = None  # tuple (nfo_a, nfo_b)
        self.nfv    = None  # tuple (nfv_a, nfv_b)
        self.es_mf  = None

        # LO-basis attributes (needed by concentric localization)
        self.caolo = None
        self.cloao = None
        self.lo_cloes = None  # tuple (lo_cloes_a, lo_cloes_b) for UHF
        self.open_shell = True
        self.dm_pair = None
        self.es_int1e = None  # tuple (es_int1e_a, es_int1e_b) for UHF
        self.es_int2e = None  # tuple (es_int2e_a, es_int2e_b) for UHF

    def svd_build_embedded_space(self,ldm,imp_idx,threshold=1e-8):
        """ Use SVD of Imp-Env block to generate bath for single-spin
        Returns
            Coeff from LO basis to EO/FO/FV
            Nimp, Nbath, Nfo, Nfv
        """
        env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]
        ldm_env_imp = ldm[env_idx,:][:, imp_idx]
        u,s,vh = svd(ldm_env_imp, full_matrices=False) # generating bath from SVD of imp-env block
        
        idx = np.where(s > threshold)[0]
        s = s[idx]
        u = u[:,idx]
        nimp = len(imp_idx)
        nbath = len(s)

        ldm_eo = u @ u.T
        ldm_fov = ldm[env_idx,:][:,env_idx] - ldm_eo

        e,v = eigh(ldm_fov)
        nfo = np.sum(e>1 - threshold)
        nfv = ldm.shape[0] - nimp - nbath - nfo

        fo_idx = np.nonzero(e > 1 - threshold)[0]
        fo_orb = v[:, fo_idx]

        # fv_orb: among non-fo eigenvectors, pick the nfv with largest eigenvalues
        non_fo_idx = np.array([i for i in range(len(e)) if i not in fo_idx])
        fv_idx = non_fo_idx[-nfv:] if nfv > 0 else np.array([], dtype=int)
        fv_orb = v[:, fv_idx]

        cloes = np.zeros((ldm.shape[0], nimp+nbath+nfo+nfv))
        cloes[:nimp,:nimp] = np.eye(nimp)
        cloes[nimp:,nimp:nimp+nbath] = u
        cloes[nimp:,nimp+nbath:nimp+nbath+nfo] = fo_orb
        cloes[nimp:,nimp+nbath+nfo:] = fv_orb

        rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
        cloes = cloes[rearange_idx,:]        

        return cloes, nimp, nbath, nfo, nfv

    
    def eig_build_embedded_space(self, ldm, imp_idx, lo_meth='lowdin', thres=1e-12):

        env_idx = [x for x in range(ldm.shape[0]) if x not in imp_idx]
        ldm_imp = ldm[imp_idx,:][:,imp_idx]
        ldm_env = ldm[env_idx,:][:,env_idx]
        ldm_imp_env = ldm[imp_idx,:][:,env_idx]
        ldm_env_imp = ldm[env_idx,:][:,imp_idx]

        occ_env, orb_env = np.linalg.eigh(ldm_env) # occupation and orbitals on environment

        nimp = len(imp_idx)
        nfv = np.sum(occ_env <  thres) # frozen virtual 
        nbath = np.sum((occ_env >= thres) & (occ_env <= 1-thres)) # bath orbital
        nfo = np.sum(occ_env > 1-thres) # frozen occupied

        # defined w.r.t enviroment orbital index
        fv_idx = np.nonzero(occ_env <  thres)[0]
        bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 1-thres))[0]
        fo_idx = np.nonzero(occ_env > 1-thres)[0]

        orb_env = np.hstack((orb_env[:, bath_idx], orb_env[:, fo_idx], orb_env[:, fv_idx]))
        
        es_occ = None
        cloes = block_diag(np.eye(nimp), orb_env)
        
        rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
        cloes = cloes[rearange_idx,:]

        return cloes, nimp, nbath, nfo, nfv
    # 20260515
    def lowdin_orth(self, restore_imp = False):
        # lowdin orthonormalize of DM for different spin.
        caolo = lowdin(self.mf_or_cas.get_ovlp())
        cloao = caolo @ self.mf_or_cas.get_ovlp()
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
        ldm = np.einsum('ij,sjk,kl->sil',cloao, self.dm, cloao.conj().T)
        return ldm[0],ldm[1], caolo, cloao

    def load_chk(self, chk_fname):
        try:
            if not '_uhf_chk.h5' in chk_fname:
                chk_fname = chk_fname + '_uhf_chk.h5'
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
                es_orb_a = fh5['es_orb_a'][:]
                es_orb_b = fh5['es_orb_b'][:]
                self.es_orb = (es_orb_a, es_orb_b)

                fo_orb_a = fh5['fo_orb_a'][:]
                fo_orb_b = fh5['fo_orb_b'][:]
                self.fo_orb = (fo_orb_a, fo_orb_b)

                fv_orb_a = fh5['fv_orb_a'][:]
                fv_orb_b = fh5['fv_orb_b'][:]
                self.fv_orb = (fv_orb_a, fv_orb_b)

                es_dm_a = fh5['es_dm_a'][:]
                es_dm_b = fh5['es_dm_b'][:]
                self.es_dm = (es_dm_a, es_dm_b)

                self.nes = tuple(fh5['nes'][:])
                self.nfo = tuple(fh5['nfo'][:])
                return True
            else:
                self.log.info(f'density matrix check {dm_check}')
                self.log.info(f'impurity index check {imp_idx_check}')
                self.log.info(f'threshold check {threshold_check}')
                self.log.info(f'build uhf embedding with imp idx {self.imp_idx} threshold {self.threshold}')
                return False

    def save_chk(self, chk_fname):
        if not '_uhf_chk.h5' in chk_fname:
            chk_fname = chk_fname + '_uhf_chk.h5'
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
        return

    def build(self, restore_imp = False , chk_fname_load='', save_chk=True):
        '''
        Build Embedding Space and Embedding Space Mean Field Object.
        For UHF 2 different spin space is needed and ERI is not stored.
        '''
        self.dm = self.mf_or_cas.make_rdm1()
        self.dm_pair = self.dm  # UHF always has spin-resolved dm
        loaded = self.load_chk(chk_fname_load)

        if not loaded:
            ldm0 ,ldm1, caolo, cloao = self.lowdin_orth(restore_imp)
            if self.es_method == 'svd':
                cloes0, nimp, nbath0, nfo0, nfv0 = self.svd_build_embedded_space(ldm0,self.imp_idx,self.threshold)
                cloes1, _   , nbath1, nfo1, nfv1 = self.svd_build_embedded_space(ldm1,self.imp_idx,self.threshold)
            else:  # es_method == 'eig'
                cloes0, nimp, nbath0, nfo0, nfv0 = self.eig_build_embedded_space(ldm0,self.imp_idx)
                cloes1, _   , nbath1, nfo1, nfv1 = self.eig_build_embedded_space(ldm1,self.imp_idx)

            self.nes = (nimp+nbath0,nimp+nbath1)
            self.nfo = (nfo0,nfo1)
            self.nfv = (nfv0,nfv1)

            # Save LO-basis attributes for concentric localization
            self.caolo = caolo
            self.cloao = cloao
            self.lo_cloes = (cloes0, cloes1)

            self.caoes = (caolo @ cloes0, caolo @ cloes1)
            self.es_orb = (self.caoes[0][:,:self.nes[0]], self.caoes[1][:,:self.nes[1]])
            self.fo_orb = (self.caoes[0][:,self.nes[0]:self.nes[0]+self.nfo[0]], self.caoes[1][:,self.nes[1]:self.nes[1]+self.nfo[1]])
            self.fv_orb = (self.caoes[0][:,self.nes[0]+self.nfo[0]:], self.caoes[1][:,self.nes[1]+self.nfo[1]:])
            
            self.es_dm = self.make_es_dm((cloes0[:,:self.nes[0]], cloes1[:,:self.nes[1]]), (ldm0,ldm1))

            self.log.info(f"Number of Imp. {nimp}")
            self.log.info(f"Number of bath {nbath0}, {nbath1}")
            self.log.info(f"Number of fo {nfo0}, {nfo1}")
            self.log.info(f"Number of fv {nfv0}, {nfv1}")
            self.log.info(f"Embedding space {self.nes[0]}({100*self.nes[0]/self.mol.nao:.2f}%%), {self.nes[1]}({100*self.nes[1]/self.mol.nao:.2f}%%)")
            self.log.info('\n')

        # Log X2C status (propagated from parent MF)
        self.log.info('with_x2c = %s', getattr(self.mf_or_cas, 'with_x2c', None))

        self.es_mf = self.UHF()
        self.fo_ene = self._fo_ene()
        self.log.info(f'Energy from frozen orbs {self.fo_ene}')
        self.log.info(f'Energy from embedding space {self.es_e}')
        self.log.info(f'Energy from exact condiation {self.mf_or_cas.e_tot}')
        self.log.info(f'Deviation between exact condiation is {self.es_e + self.fo_ene - self.mf_or_cas.e_tot}')
        self.log.info('\n')
        if save_chk:
            chk_fname_save = self.title
            self.save_chk(chk_fname_save)


    def make_es_int1e(self):
        """Build embedded 1e Hamiltonian for each spin (AO->ES projection of Fock)."""
        hcore = self.mf_or_cas.get_hcore()
        dm_fo0 = self.fo_orb[0] @ self.fo_orb[0].T
        dm_fo1 = self.fo_orb[1] @ self.fo_orb[1].T

        vj0, vk0 = self.mf_or_cas.get_jk(mol=self.mf_or_cas.mol, dm=dm_fo0)
        vj1, vk1 = self.mf_or_cas.get_jk(mol=self.mf_or_cas.mol, dm=dm_fo1)

        fock0 = hcore + vj0 + vj1 - vk0
        fock1 = hcore + vj0 + vj1 - vk1

        es_int1e0 = lib.einsum('pi,ij,jq->pq', self.es_orb[0].conj().T, fock0, self.es_orb[0])
        es_int1e1 = lib.einsum('pi,ij,jq->pq', self.es_orb[1].conj().T, fock1, self.es_orb[1])
        self.es_int1e = (es_int1e0, es_int1e1)
        return self.es_int1e

    def make_es_int2e(self):
        """Build full 4-index ERI in embedded space for each spin block."""
        es_int2e_aa = ao2mo.restore(8, ao2mo.full(self.mf_or_cas.mol, self.es_orb[0]), self.nes[0])
        es_int2e_bb = ao2mo.restore(8, ao2mo.full(self.mf_or_cas.mol, self.es_orb[1]), self.nes[1])
        es_int2e_ab = ao2mo.general(self.mf_or_cas.mol,
                                     (self.es_orb[0], self.es_orb[0], self.es_orb[1], self.es_orb[1]),
                                     compact=False)
        self.es_int2e = (es_int2e_aa, es_int2e_bb, es_int2e_ab)
        return self.es_int2e

    def make_es_dm(self, lo2es, ldm, use_ao=False):
        """Build DM in embedding space.

        Parameters
        ----------
        lo2es : tuple of ndarray
            Embedding-space coefficients: either in LO basis (from
            lo_cloes) or in AO basis (from es_orb).
        ldm : tuple of ndarray
            Density matrix in the matching basis:
            - LO basis: Lowdin-orthogonalized DM from lowdin_orth()
            - AO basis: AO DM from mf_or_cas.make_rdm1()
        use_ao : bool
            If True, use AO-basis projection (S-convention).
            If False, use the original LO-basis formula.
        """
        if use_ao:
            S = self.mf_or_cas.get_ovlp()
            es_dm0 = lo2es[0].T @ S @ ldm[0] @ S @ lo2es[0]
            es_dm1 = lo2es[1].T @ S @ ldm[1] @ S @ lo2es[1]
        else:
            es_dm0 = np.einsum('ij,jk,kl->il', lo2es[0].T, ldm[0], lo2es[0])
            es_dm1 = np.einsum('ij,jk,kl->il', lo2es[1].T, ldm[1], lo2es[1])
        return (es_dm0, es_dm1)
    
    def UHF(self):
        '''
        Embedding UHF mean field object.
        X2C corrections, if present in the parent, flow through
        get_hcore() (which returns the parent's x2c-corrected 1e
        Hamiltonian).  with_x2c is propagated in UHF_EMB.__init__
        so that downstream post-HF code stays consistent.
        '''
        mol = gto.M()
        mol.nelec = (self.mol.nelec[0] - self.nfo[0], self.mol.nelec[1] - self.nfo[1])
        mol.build()
        mf_emb = UHF_EMB(mol, self.mf_or_cas, self.es_orb, self.fo_orb, self.es_dm)
        self.es_e = mf_emb.energy_elec()[0]
        return mf_emb

    def _fo_ene(self):
        """Internal: compute FO energy (called by build)."""
        return self.calc_fo_ene()

    def calc_fo_ene(self, e_nuc=True):
        """Compute energy of frozen occupied orbitals (+ nuclear repulsion) for UHF."""
        dm_fo0 = self.fo_orb[0] @ self.fo_orb[0].T
        dm_fo1 = self.fo_orb[1] @ self.fo_orb[1].T

        hcore = self.mf_or_cas.get_hcore()
        vj0, vk0 = self.mf_or_cas.get_jk(mol=self.mf_or_cas.mol, dm=dm_fo0)
        vj1, vk1 = self.mf_or_cas.get_jk(mol=self.mf_or_cas.mol, dm=dm_fo1)
        vhf0 = vj0 + vj1 - vk0
        vhf1 = vj0 + vj1 - vk1

        e1  = np.einsum('ij,ji->', hcore, dm_fo0)
        e1 += np.einsum('ij,ji->', hcore, dm_fo1)
        e_coul  = 0.5 * np.einsum('ij,ji->', vhf0, dm_fo0)
        e_coul += 0.5 * np.einsum('ij,ji->', vhf1, dm_fo1)
        e_elec = e1 + e_coul
        fo_ene = e_elec
        if e_nuc:
            fo_ene += self.mf_or_cas.energy_nuc()
        self.fo_ene = fo_ene
        return fo_ene

    def uccsd(self):
        '''
        Modified Unrestrited CCSD solver for Embedding Space.
        '''
        mycc = cc.UCCSD(self.es_mf)
        e_hf_fixed = self.es_e
        mycc.get_e_hf = lambda mo_coeff=None: e_hf_fixed
        mycc.ao2mo = types.MethodType(_ao2mo, mycc)
        return mycc
    def ump2(self):
        from pyscf import mp
        mymp2 = mp.UMP2(self.es_mf)
        e_hf_fixed = self.es_e
        mymp2.get_e_hf = lambda mo_coeff=None: e_hf_fixed
        mymp2.ao2mo = types.MethodType(_ao2mo_mp2, mymp2)
        return mymp2

    def density_fit(self, with_df=None):
        """Return a density-fitting enabled UHF DMET object."""
        from embed_sim.df_uhf_tool import DFSSDMET_uhf
        df_dmet = DFSSDMET_uhf(self.mf_or_cas, self.title, imp_idx=self.imp_idx, threshold=self.threshold,
                           bath_option=self.bath_option, es_method=self.es_method, with_df=with_df,
                           verbose=self.verbose)
        df_dmet.__dict__.update(self.__dict__)
        if with_df is not None:
            df_dmet.with_df = with_df
        return df_dmet

class UHF_EMB(scf.uhf.UHF):
    """
    A class inherited from UHF with modified get_veff and get_fock... enough for subsequent high level calculation.

    Inputs
        mol     : Only indicate nelectron can be optimized. TODO
        mf      : Mean field object of the full molecule
        es_orb  : Orbital coefficients from AO to ES
        fo_orb  : Orbital coefficients from AO to FO
        es_dm   : Density matrix of embedding space
    
    NOTE Energy calculation will not store ERI in ES. 

    Properties
        mo_coeff : from ES to MO(ES) after eig of fock.  
                    need special care when subsequent high level calculation is performed
        
    """
    def __init__(self, mol, mf, es_orb, fo_orb, es_dm):
        super().__init__(mol)

        self.mf_emb = mf
        self.es_orb = es_orb
        self.fo_orb = fo_orb
        self.dm_emb = es_dm

        # Propagate X2C from parent mean-field object.
        # x2c modifies the 1e Hamiltonian; since get_hcore() is overridden to use
        # the parent's (already x2c-corrected) hcore, we only need to retain the
        # with_x2c attribute so that energy_nuc() and downstream code stay consistent.
        self.with_x2c = getattr(mf, 'with_x2c', None)

        self.veff = self.get_veff()
        self.fock = self.get_fock()
        self.mo_energy, self.mo_coeff = self.eig(fock = self.fock)
        self.mo_occ = self.get_occ()


    def get_ovlp(self):
        s = self.mf_emb.get_ovlp()
        s0 = self.es_orb[0].T @ s @ self.es_orb[0]
        s1 = self.es_orb[1].T @ s @ self.es_orb[1]
        return (s0, s1)

    def get_veff(self,mol=None,dm=None):# don't delete. AO2MO call this fucntion with these 2 args.
        
        vj,vk = self._get_jk(self.dm_emb, self.es_orb)
        return vj[0]-vk[0], vj[1]-vk[1]

    def get_hcore(self):
        h = self.mf_emb.get_hcore()
        dm_fo0, dm_fo1 = self.fo_orb[0] @ self.fo_orb[0].T , self.fo_orb[1] @ self.fo_orb[1].T

        vj0,vk0 = self.mf_emb.get_jk(mol = self.mf_emb.mol, dm = dm_fo0)
        vj1,vk1 = self.mf_emb.get_jk(mol = self.mf_emb.mol, dm = dm_fo1)

        f0 = h +  vj0 + vj1 - vk0
        f1 = h +  vj0 + vj1 - vk1 
        h0 = lib.einsum('pi,ij,jq->pq',self.es_orb[0].conj().T, f0, self.es_orb[0])
        h1 = lib.einsum('pi,ij,jq->pq',self.es_orb[1].conj().T, f1, self.es_orb[1])
        
        return h0, h1

    def get_fock(self,vhf=None, dm=None):
        h1e0,h1e1 = self.get_hcore()
        if self.veff is None:
            self.veff = self.get_veff()
        f0, f1 = h1e0 + self.veff[0], h1e1 + self.veff[1]
        return f0, f1
    
    def eig(self,fock=None,s=None):
        if fock is None: fock = self.get_fock()
        s0, s1 = self.get_ovlp()
        e0, v0 = eigh(fock[0], s0)
        e1, v1 = eigh(fock[1], s1)
        return (e0, e1), (v0,v1)
    
    def get_occ(self):
        nelec0, nelec1 = self.mol.nelec
        occ0, occ1 = np.zeros_like(self.mo_energy[0]), np.zeros_like(self.mo_energy[1])
        occ0[:nelec0] = 1
        occ1[:nelec1] = 1
        return (occ0,occ1)
    
    def energy_elec(self,dm=None, h1e=None, vhf=None):
        h0, h1 = self.get_hcore()
        v0, v1 = self.veff
        e1 = lib.einsum('pq,qp->',h0,self.dm_emb[0])+lib.einsum('pq,qp->',h1,self.dm_emb[1])
        e2 = 0.5 * (lib.einsum('pq,qp->', v0,self.dm_emb[0]) + lib.einsum('pq,qp->',v1, self.dm_emb[1]))
        e = e1+e2
        return e,e2

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]
        dm_a = np.dot(mo_a*mo_occ[0], mo_a.conj().T)
        dm_b = np.dot(mo_b*mo_occ[1], mo_b.conj().T)
        return (dm_a, dm_b)

    def _get_jk(self, dm, es_orb):

        mo_a, mo_b = es_orb
        dm_a, dm_b = dm
        dm_ao_a = lib.einsum('pi,ij,qj->pq', mo_a, dm_a, mo_a)
        dm_ao_b = lib.einsum('pi,ij,qj->pq', mo_b, dm_b, mo_b)
        
        dm_tot = dm_ao_a + dm_ao_b

        vj_ao = self.get_j(self.mf_emb.mol, dm_tot, hermi=1)
        vk_ao = self.get_k(self.mf_emb.mol, (dm_ao_a, dm_ao_b), hermi=1)
        

        vj_mo_a = lib.einsum('pi,pq,qj->ij', mo_a, vj_ao, mo_a)
        vj_mo_b = lib.einsum('pi,pq,qj->ij', mo_b, vj_ao, mo_b)

        vk_mo_a = lib.einsum('pi,pq,qj->ij', mo_a, vk_ao[0], mo_a)
        vk_mo_b = lib.einsum('pi,pq,qj->ij', mo_b, vk_ao[1], mo_b)

        return (vj_mo_a, vj_mo_b), (vk_mo_a, vk_mo_b)
    
    def get_caomo(self, mo_coeff=None):
        """
        Return AO -> MO(ES) coefficients: caomo = es_orb @ mo_coeff.
        These go directly from AO basis to the canonical MOs of the embedded space. 
        Suitable for MP/CC ERI transformation
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        caomo_a = self.es_orb[0] @ mo_coeff[0]
        caomo_b = self.es_orb[1] @ mo_coeff[1]
        return caomo_a, caomo_b

    def get_eri(self,mo_coeff=None):
        """Return full MO ERIs (aa, bb, ab) as 4D arrays."""
        caomo_a, caomo_b = self.get_caomo(mo_coeff)
        aa = ao2mo.restore(1, ao2mo.full(self.mf_emb.mol, caomo_a, compact=True), caomo_a.shape[1])
        bb = ao2mo.restore(1, ao2mo.full(self.mf_emb.mol, caomo_b, compact=True), caomo_b.shape[1])
        ab = ao2mo.general(self.mf_emb.mol, (caomo_a, caomo_a, caomo_b, caomo_b), compact=False)
        return aa, bb, ab
