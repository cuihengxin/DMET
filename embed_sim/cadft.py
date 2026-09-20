"""Configuration-averaged DFT (CA-DFT) references for DMET.

Ensemble/average-of-configuration mean field with FIXED fractional occupations
on an (ncas, nelecas) active space (equal-weight average over all
configurations of the active space):

    E_CA = sum_i f_i <phi_i|h|phi_i> + 1/2 J[rho] + E_xc[rho_a, rho_b] + E_nn
    rho_s(r) = sum_i f_i^s |phi_i(r)|^2,   v_xc evaluated on the
             fractional-occupation density (Gross-Oliveira-Kohn ensemble DFT;
             no CAHF-style lambda_J/lambda_K operator corrections are needed)

Two schemes:
  CADFT_RKS : spin-averaged ensemble, RKS mo_occ convention; requires
              mol.spin == 0 and nelecas even (same parity as mol.nelectron).
  CADFT_UKS : fixed-Sz ensemble, f_a = na/ncas, f_b = nb/ncas with
              na = (nelecas+spin)/2, nb = (nelecas-spin)/2;
              requires spin == mol.spin.

Occupations are assigned by orbital POSITION (first ncore orbitals closed,
next ncas orbitals carry the fractional occupation), the same convention as
embed_sim.cahf.CAHF, and are never reordered by Aufbau during SCF.  The
active occupation is nelecas/ncas exactly, so the energy is continuous in
nelecas and Janak's theorem dE/dN_active = <eps_active> holds at
convergence; the reference reduces exactly to ordinary RKS/UKS when
nelecas = 0 or 2*ncas.  For nelecas of the same parity as the molecular
electron count the ensemble carries an integer number of electrons; a
nelecas probe with odd parity defines a fractional-N ensemble (Janak's
chemical-potential construction) and raises no error, only a warning.

Usage:
    mf = cadft.CADFT_RKS(mol, xc='pbe', ncas=4, nelecas=4).run()
    mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx='O').build()
"""

import numpy as np
from pyscf import scf
from pyscf.dft import rks, uks
from pyscf.lib import logger


class CADFT_RKS(rks.RKS):
    """Spin-averaged configuration-averaged DFT (fractional-occupation RKS)."""

    def __init__(self, mol, xc='pbe', ncas=None, nelecas=None):
        if mol.spin != 0:
            raise ValueError('CADFT_RKS is the spin-averaged ensemble; '
                             'mol.spin must be 0 (use CADFT_UKS otherwise)')
        if ncas is None or nelecas is None:
            raise ValueError('ncas and nelecas must be given')
        if not 0 <= nelecas <= 2 * ncas:
            raise ValueError(f'nelecas={nelecas} out of range [0, {2*ncas}]')
        rks.RKS.__init__(self, mol, xc)
        self.ncas = int(ncas)
        self.nelecas = float(nelecas)
        self.conv_check = False

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        nmo = mo_energy.size
        ncore = int(np.round((self.mol.nelectron - self.nelecas) / 2.0))
        act_occ = self.nelecas / self.ncas
        if (self.mol.nelectron - self.nelecas) % 2 > 1e-9:
            logger.warn(self, 'nelecas=%g has odd parity vs mol.nelectron=%d; '
                        'the ensemble carries a fractional electron count '
                        '(fine for Janak probes, use CADFT_UKS otherwise)',
                        self.nelecas, self.mol.nelectron)
        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[ncore:ncore + self.ncas] = act_occ

        if self.verbose >= logger.INFO and ncore + self.ncas < nmo:
            ehomo = mo_energy[mo_occ > 1e-9].max()
            elumo = mo_energy[mo_occ <= 1e-9].min()
            if ehomo + 1e-3 > elumo:
                logger.warn(self, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
            else:
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
            logger.info(self, '  CA-DFT active occ = %.6f x %d orbitals (ncore=%d)',
                        act_occ, self.ncas, ncore)
        return mo_occ


class CADFT_UKS(uks.UKS):
    """Fixed-Sz configuration-averaged DFT (fractional-occupation UKS)."""

    def __init__(self, mol, xc='pbe', ncas=None, nelecas=None, spin=0):
        if ncas is None or nelecas is None:
            raise ValueError('ncas and nelecas must be given')
        if not 0 <= nelecas <= 2 * ncas:
            raise ValueError(f'nelecas={nelecas} out of range [0, {2*ncas}]')
        if spin != mol.spin:
            raise ValueError(f'active spin {spin} must equal mol.spin {mol.spin}')
        uks.UKS.__init__(self, mol, xc)
        self.ncas = int(ncas)
        self.nelecas = float(nelecas)
        self.spin = int(spin)
        self.conv_check = False

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        nmo = mo_energy[0].size
        na_act = (self.nelecas + self.spin) / 2.0
        nb_act = (self.nelecas - self.spin) / 2.0
        mo_occ = np.zeros((2, nmo))
        act_occs = []
        for s, (n_tot, n_act) in enumerate([(self.mol.nelec[0], na_act),
                                            (self.mol.nelec[1], nb_act)]):
            ncore = int(np.round(n_tot - n_act))
            occ = n_act / self.ncas
            act_occs.append(occ)
            mo_occ[s, :ncore] = 1.0
            mo_occ[s, ncore:ncore + self.ncas] = occ

        if self.verbose >= logger.INFO and max(
                int(np.round(self.mol.nelec[0] - na_act)),
                int(np.round(self.mol.nelec[1] - nb_act))) + self.ncas < nmo:
            ehomo = max(mo_energy[0][mo_occ[0] > 1e-9].max(),
                        mo_energy[1][mo_occ[1] > 1e-9].max())
            elumo = min(mo_energy[0][mo_occ[0] <= 1e-9].min(),
                        mo_energy[1][mo_occ[1] <= 1e-9].min())
            if ehomo + 1e-3 > elumo:
                logger.warn(self, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
            else:
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
            logger.info(self, '  CA-DFT active occ (a,b) = %.6f/%.6f x %d orbitals',
                        act_occs[0], act_occs[1], self.ncas)
        return mo_occ
