#!/usr/bin/env python
"""Analytic nuclear gradients for one-shot SSDMET (``embed_sim.ssdmet.SSDMET``).

Scope of this first stage
-------------------------
* global reference  : converged closed-shell ``scf.RHF`` (no x2c, no density
                      fitting, plain Lowdin localization)
* embedded solver   : the embedded mean field ``mydmet.es_mf`` (HF-in-HF)
* bath              : threshold-based or ``bath_norb`` fixed-size selection
                      (``bath_option`` MP2 bath expansion is *not* supported)

Why this is simpler than democratic-partitioning DMET
-----------------------------------------------------
``SSDMET`` uses a single fragment and partitions the environment into
bath / frozen-occupied / frozen-virtual, so the total energy is a plain
CASCI-type functional

    E = E_nuc + Tr[h D_c] + 1/2 Tr[v(D_c) D_c]
              + Tr[r1 h1] + 1/2 r2:g_es ,
    D_c = 2 Cc Cc^T,   h1 = C^T (h + v(D_c)) C,   g_es = (CC|CC),

with ``v(D) = J(D) - K(D)/2``.  There is no democratic weight and no
back-projected density, hence the functional is *stationary* with respect to
the embedded wavefunction whenever the embedded solver is variational
(RHF / FCI / CASSCF).  The impurity adjoint (CP-CI / CP-HF) layer of the
democratic-partitioning formulation is therefore not needed here and the
Hellmann-Feynman theorem applies inside the cluster: ``r1`` and ``r2`` are held
fixed while differentiating.

Response chain that *does* survive
----------------------------------
The orbitals are ``C = X L`` and ``Cc = X Lc`` with ``X = S^{-1/2}`` and
``L / Lc`` built from the eigenvectors of the environment block of

    gamma = S^{1/2} P S^{1/2}   (P = global RHF AO density) .

So

    dE = dE|_explicit integrals
       + Tr[B_X^T dX]                       (Lowdin channel)
       + Tr[G_gamma dgamma],
    dgamma = dY P Y + Y dP Y + Y P dY,      Y = S^{1/2} ,

which splits into an overlap (Pulay) channel and a mean-field density channel.
The latter is eliminated with one global CPHF / Z-vector solve.  Everything is
finally contracted with ``dh/dR``, ``dg/dR`` and ``dS/dR``.

Reference for the democratic-partitioning counterpart of these formulas:
``8dmet4reac/DMET-preview/dmet/grad/`` and
``8dmet4reac/oneshot_dmet解析梯度推导与pyscf实现计划.md``.
"""

from functools import reduce

import numpy as np

from pyscf import lib, scf
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
from pyscf.scf import cphf

from embed_sim.bath_selection import (count_imp_env_bonds,
                                      partition_env_by_bath_count)
from embed_sim.grad.linalg import eig_subspace_grad, matfun_grad

einsum = lib.einsum


def symmetrize_rdm2(r2):
    """Symmetrize a 2-RDM in the (p<->q), (r<->s) index pairs.

    ``make_rdm12`` output already satisfies ``r2[pqrs] = r2[rspq]``.  Adding the
    remaining two permutations lets every AO-derivative index position be
    treated on the same footing (a single contraction times 4).
    """
    r2 = 0.5 * (r2 + r2.transpose(1, 0, 2, 3))
    r2 = 0.5 * (r2 + r2.transpose(0, 1, 3, 2))
    return r2


class SSDMETGradients(lib.StreamObject):
    """Analytic nuclear gradient of a built :class:`embed_sim.ssdmet.SSDMET`."""

    def __init__(self, base, conv_tol_cphf=1e-9, max_cycle_cphf=50,
                 verbose=None):
        self.base = base
        self.mol = base.mol
        self.mf = base.mf_or_cas
        self.conv_tol_cphf = conv_tol_cphf
        self.max_cycle_cphf = max_cycle_cphf
        self.verbose = base.verbose if verbose is None else verbose
        self.log = logger.new_logger(self.mol, self.verbose)

        self.mfgrad = self.mf.nuc_grad_method()
        # A plain RHF gradient helper, used only for the closed-shell veff
        # derivative d/dR [J(D) - K(D)/2].  Keeping it separate from
        # ``self.mfgrad`` means the ROHF branch does not accidentally pick up
        # uhf_grad.get_veff (which expects spin-resolved densities).
        self._veff_helper = rhf_grad.Gradients(scf.hf.RHF(self.mol))
        self.openshell = False

        self.de = None
        self.e_tot = None

    def _veff_deriv(self, dm):
        """d/dR of the closed-shell effective potential v(D) = J(D) - K(D)/2."""
        return self._veff_helper.get_veff(self.mol, dm)

    # ------------------------------------------------------------------
    # 0. what we can and cannot differentiate
    # ------------------------------------------------------------------
    def check_support(self):
        base, mf, mol = self.base, self.mf, self.mol

        if base.es_mf is None or base.es_orb is None:
            raise RuntimeError('run mydmet.build() before asking for gradients')
        if base.bath_option is not None:
            raise NotImplementedError(
                'gradients for the MP2 bath expansion (bath_option) are not '
                'implemented; use the plain/fixed-size bath')
        if getattr(mf, 'with_df', None) is not None:
            raise NotImplementedError('density fitting is not supported yet')
        if getattr(mf, 'with_x2c', None) is not None:
            raise NotImplementedError('x2c reference is not supported yet')
        if isinstance(mf, scf.uhf.UHF):
            raise NotImplementedError(
                'a UHF reference is not supported; use RHF or ROHF')
        if not isinstance(mf, scf.hf.RHF):
            raise NotImplementedError(
                'only RHF (closed shell) and ROHF references are supported')
        # ROHF is a subclass of RHF in PySCF; the open-shell branch is selected
        # here and used by make_densities / solve_z / make_sbar.
        self.openshell = isinstance(mf, scf.rohf.ROHF)
        if self.openshell and mol.spin == 0:
            self.openshell = False      # ROHF object but closed shell
        if not self.openshell and mol.spin != 0:
            raise NotImplementedError(
                'spin != 0 requires an ROHF reference object')
        if not mf.converged:
            raise RuntimeError('global mean field is not converged')
        if not base.es_mf.converged:
            raise RuntimeError('embedded mean field is not converged')
        if np.ndim(base.dm) != 2:
            raise RuntimeError(
                'base.dm should be the total (spin-summed) AO density')

    # ------------------------------------------------------------------
    # 1. rebuild the LO decomposition, keeping the intermediates
    # ------------------------------------------------------------------
    def decompose(self):
        """Redo ``build_embeded_subspace`` while retaining eigen-intermediates.

        Sets ``w_s/u_s`` (eigen-decomposition of S), ``X``, ``Y``, the
        environment eigenpairs ``w/V``, the bath/frozen index groups, and the
        LO->EO coefficient blocks in the *raw* (un-rotated) eigenvector gauge.
        """
        base, mf = self.base, self.mf
        ovlp = mf.get_ovlp()
        self.ovlp = ovlp
        nao = ovlp.shape[0]

        w_s, u_s = np.linalg.eigh(ovlp)
        if w_s.min() < 1e-12:
            raise RuntimeError('AO overlap is (near) singular; the Lowdin '
                               'derivative would be meaningless')
        self.w_s, self.u_s = w_s, u_s
        self.X = (u_s * w_s ** -0.5) @ u_s.T          # S^{-1/2}
        self.Y = (u_s * w_s ** 0.5) @ u_s.T           # S^{1/2}

        if base.caolo is not None and \
                not np.allclose(base.caolo, self.X, atol=1e-8):
            raise NotImplementedError(
                'the DMET was built with a non-Lowdin localization '
                '(restore_imp / iaopao); only plain Lowdin is supported')

        imp_idx = np.asarray(base.imp_idx, dtype=int)
        env_idx = np.array([x for x in range(nao) if x not in imp_idx],
                           dtype=int)
        self.imp_idx, self.env_idx = imp_idx, env_idx
        nimp, nenv = imp_idx.size, env_idx.size

        gamma = self.Y @ base.dm @ self.Y
        self.gamma = gamma
        w, V = np.linalg.eigh(gamma[np.ix_(env_idx, env_idx)])
        self.w, self.V = w, V

        bath_norb = base.bath_norb
        if isinstance(bath_norb, str):
            bath_norb = count_imp_env_bonds(self.mol, base.imp_idx)
        thres = base.threshold
        if bath_norb is None:
            fv_sel = np.nonzero(w < thres)[0]
            bath_sel = np.nonzero((w >= thres) & (w <= 2 - thres))[0]
            fo_sel = np.nonzero(w > 2 - thres)[0]
        else:
            bath_sel, fo_sel, fv_sel = partition_env_by_bath_count(
                w, bath_norb, thres=thres, core_cutoff=base.bath_core_cutoff)
        self.bath_sel, self.fo_sel, self.fv_sel = bath_sel, fo_sel, fv_sel

        nbath, nfo, nfv = bath_sel.size, fo_sel.size, fv_sel.size
        nes = nimp + nbath
        if (nes, nfo, nfv) != (base.nes, base.nfo, base.nfv):
            raise RuntimeError(
                'bath partition could not be reproduced: got '
                f'(nes,nfo,nfv)=({nes},{nfo},{nfv}) vs stored '
                f'({base.nes},{base.nfo},{base.nfv}). Rebuild the DMET without '
                'a checkpoint file.')
        self.nimp, self.nbath, self.nfo, self.nfv, self.nes = \
            nimp, nbath, nfo, nfv, nes

        # group label per environment eigenvector: 0 bath, 1 frozen-occ, 2 frozen-vir
        group = np.empty(nenv, dtype=int)
        group[bath_sel] = 0
        group[fo_sel] = 1
        group[fv_sel] = 2
        self.group = group

        # raw (un-rotated) LO->EO blocks
        L_raw = np.zeros((nao, nes))
        L_raw[imp_idx, np.arange(nimp)] = 1.0
        L_raw[np.ix_(env_idx, np.arange(nimp, nes))] = V[:, bath_sel]
        Lc_raw = np.zeros((nao, nfo))
        Lc_raw[np.ix_(env_idx, np.arange(nfo))] = V[:, fo_sel]
        self.L_raw, self.Lc_raw = L_raw, Lc_raw

        # the ES basis actually used by the solver (es_natorb rotation included)
        L_act = self.Y @ base.es_orb
        self.L_act = L_act
        # U maps the raw gauge onto the solver gauge: L_act = L_raw @ U.
        # When the bath (or frozen) eigenvalues are (near) degenerate the raw
        # eigenvectors V are only determined up to a rotation within the block;
        # the energy and gradient are invariant to that rotation, so U is not
        # unique and the *projected* consistency below is the right check.
        U = L_raw.T @ L_act
        proj = L_raw @ (L_raw.T @ L_act)
        if not np.allclose(proj, L_act, atol=1e-6, rtol=1e-6):
            raise RuntimeError('embedded subspace does not match the '
                               'recomputed bath; rebuild without checkpoint')
        self.U = U

        # frozen-occupied space: any orthonormal basis gives the same D_c
        self.Ccore = self.X @ Lc_raw
        dc_ref = base.fo_orb @ base.fo_orb.T
        if not np.allclose(self.Ccore @ self.Ccore.T, dc_ref, atol=1e-7):
            raise RuntimeError('frozen-occupied subspace does not match the '
                               'recomputed one; rebuild without checkpoint')

    # ------------------------------------------------------------------
    # 2. densities and the embedded 1-/2-RDM
    # ------------------------------------------------------------------
    def make_densities(self):
        base, mf, mol = self.base, self.mf, self.mol
        C = base.es_orb
        self.C = C
        Ccore = self.Ccore

        self.dm_core = 2.0 * (Ccore @ Ccore.T)
        r1s = base.es_mf.make_rdm1()
        if np.ndim(r1s) == 3:           # ROHF / open shell: (dma, dmb)
            r1a, r1b = r1s[0], r1s[1]
        else:                           # RHF: total density, equal spins
            r1a = r1b = 0.5 * r1s
        r1 = r1a + r1b
        self.r1, self.r1a, self.r1b = r1, r1a, r1b
        # HF-in-HF: the cluster 2-RDM of a single determinant.  The exchange
        # part is spin resolved (same-spin only); for RHF r1a = r1b = r1/2
        # reproduces the closed-shell  r1 (x) r1 - 1/2 r1[ux] r1[vw].
        self.r2 = symmetrize_rdm2(
            einsum('uv,wx->uvwx', r1, r1)
            - einsum('ux,vw->uvwx', r1a, r1a)
            - einsum('ux,vw->uvwx', r1b, r1b))
        self.dm_act = C @ r1 @ C.T

        vj, vk = mf.get_jk(mol, np.asarray((self.dm_core, self.dm_act)))
        self.veff_core = vj[0] - 0.5 * vk[0]
        self.veff_act = vj[1] - 0.5 * vk[1]
        self.hcore = mf.get_hcore()
        self.fock_core = self.hcore + self.veff_core          # h + v(D_c)
        self.fock_tot = self.fock_core + self.veff_act        # dE/dD_c

        # consistency with what the solver actually used
        h1 = C.T @ self.fock_core @ C
        if not np.allclose(h1, base.es_int1e, atol=1e-7):
            self.log.warn('recomputed embedded h1 differs from es_int1e by '
                          '%.3e', abs(h1 - base.es_int1e).max())
        self.e_tot = base.es_mf.e_tot + base.fo_ene

    # ------------------------------------------------------------------
    # 3. dE/dC and dE/dCc at fixed r1, r2
    # ------------------------------------------------------------------
    def orb_grad(self):
        mol, C = self.mol, self.C
        nao, nes = C.shape

        # (mu q|r s) with three indices in the embedded space, contracted
        # directly out of the AO integrals.
        eri = mol.intor('int2e').reshape(nao, nao, nao, nao)
        eri_aeee = einsum('uvwx,va,wb,xc->uabc', eri, C, C, C)
        self.eri_aeee = eri_aeee
        del eri

        # dE/dC = 2 F_c C r1 + 2 (mu q|r s) r2[p,q,r,s]
        g2 = eri_aeee.reshape(nao, -1) @ self.r2.reshape(nes, -1).T
        self.Ge = 2.0 * (self.fock_core @ C @ self.r1) + 2.0 * g2
        # dE/dCc with D_c = 2 Cc Cc^T
        self.Gc = 4.0 * (self.fock_tot @ self.Ccore)

        # Stationarity check: C^T dE/dC must be symmetric for a variational
        # cluster solver.  Its antisymmetric part is exactly the piece an
        # adjoint (Lambda) equation would have to supply.
        gen_fock = C.T @ self.Ge
        asym = abs(gen_fock - gen_fock.T).max()
        scale = max(abs(gen_fock).max(), 1.0)
        if asym > 1e-5 * scale:
            self.log.warn('cluster generalized Fock is not symmetric '
                          '(|asym| = %.3e): the embedded solver is not '
                          'stationary, the gradient will be wrong', asym)
        else:
            self.log.debug('cluster generalized Fock asymmetry %.3e', asym)

    # ------------------------------------------------------------------
    # 4. chain to the Lowdin matrix and the environment eigenvectors
    # ------------------------------------------------------------------
    def lo_grad(self):
        X, Y = self.X, self.Y
        env_idx = self.env_idx
        nimp, nes = self.nimp, self.nes

        # dE/dX at fixed L, Lc
        self.B_X = self.Ge @ self.L_act.T + self.Gc @ self.Lc_raw.T

        # dE/dL in the solver gauge, then rotated back to the raw eigenvectors.
        # The gauge (es_natorb) term drops out because C^T dE/dC is symmetric.
        b_es = (X @ self.Ge) @ self.U.T
        b_fo = X @ self.Gc

        nenv = env_idx.size
        bmat = np.zeros((nenv, nenv))
        bmat[:, self.bath_sel] = b_es[np.ix_(env_idx, np.arange(nimp, nes))]
        bmat[:, self.fo_sel] = b_fo[env_idx]

        G_M = eig_subspace_grad(self.w, self.V, bmat, self.group, log=self.log)

        nao = X.shape[0]
        G_gamma = np.zeros((nao, nao))
        G_gamma[np.ix_(env_idx, env_idx)] = G_M
        self.G_gamma = G_gamma

        # gamma = Y P Y  ->  density channel and overlap channel
        self.G_P = Y @ G_gamma @ Y
        B_Y = G_gamma @ Y @ self.base.dm
        self.B_Y = B_Y + B_Y.T

    # ------------------------------------------------------------------
    # 5. global CPHF / Z-vector for the mean-field density response
    # ------------------------------------------------------------------
    def solve_z(self):
        """Z-vector for the mean-field density response.

        Closed shell: one RHF CPHF solve (pyscf.scf.cphf).
        Open shell:   ROHF rotation space {vir-docc, vir-socc, socc-docc}, the
                      orbital Hessian from ``newton_ah.gen_g_hop_rohf`` (which
                      is a *constrained* UHF: one orbital set for both spins).
        """
        if self.openshell:
            return self._solve_z_rohf()
        return self._solve_z_rhf()

    def _solve_z_rhf(self):
        mf, mol = self.mf, self.mol
        mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
        occidx = mo_occ > 0
        viridx = ~occidx
        orbo, orbv = mo_coeff[:, occidx], mo_coeff[:, viridx]

        moFbar = mo_coeff.T @ self.G_P @ mo_coeff
        self.moFbar = moFbar
        # Lagrangian L = Tr[G_P P], P = 2 sum_i C_i C_i^T.  The CPHF Z-vector
        # RHS is dL/dkappa = 2 * moFbar_ai (the factor 2 is the singlet
        # response convention carried by cphf.solve + the folding below;
        # verified against central differences of Tr[G_P dP/dR]).
        xvo = 2.0 * moFbar[np.ix_(viridx, occidx)]

        def fvind(x):
            x = x.reshape(xvo.shape)
            dm = orbv @ x @ orbo.T
            v = mf.get_veff(mol, dm + dm.T)
            return (orbv.T @ v @ orbo).ravel() * 2

        if self.conv_tol_cphf is not None and self.conv_tol_cphf > 0:
            z = cphf.solve(fvind, mo_energy, mo_occ, xvo,
                           max_cycle=self.max_cycle_cphf,
                           tol=self.conv_tol_cphf)[0]
        else:  # frozen mean-field density (built-in approximation)
            z = np.zeros_like(xvo)
        self.Z = z.reshape(xvo.shape)

        nmo = mo_coeff.shape[1]
        zvec = np.zeros((nmo, nmo))
        zvec[np.ix_(viridx, occidx)] = self.Z
        self.zvec_ao = mo_coeff @ (zvec + zvec.T) @ mo_coeff.T
        self.zeta = mo_coeff @ (zvec * mo_energy[None, :]) @ mo_coeff.T
        p_occ = orbo @ orbo.T
        vj, vk = mf.get_jk(mol, self.zvec_ao)
        self.vhf_s1occ = p_occ @ (vj - 0.5 * vk) @ p_occ

    def _rohf_rotation_masks(self):
        """The non-redundant ROHF rotation blocks, matching gen_g_hop_rohf."""
        mo_occ = self.mf.mo_occ
        occidxa = mo_occ > 0        # docc + socc  (alpha occupied)
        occidxb = mo_occ == 2       # docc         (beta  occupied)
        uniq_var_a = (~occidxa)[:, None] & occidxa      # vir x (docc+socc)
        uniq_var_b = (~occidxb)[:, None] & occidxb      # (socc+vir) x docc
        return occidxa, occidxb, uniq_var_a, uniq_var_b, uniq_var_a | uniq_var_b

    def _solve_z_rohf(self):
        from pyscf.soscf import newton_ah

        mf, mol = self.mf, self.mol
        mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
        nmo = mo_coeff.shape[1]

        moFbar = mo_coeff.T @ self.G_P @ mo_coeff
        self.moFbar = moFbar

        occa, occb, var_a, var_b, uniq_ab = self._rohf_rotation_masks()
        nocca = int(np.count_nonzero(occa))
        nvira = nmo - nocca

        def sum_ab(x):
            """Fold the (alpha, beta) packing onto the shared ROHF variables."""
            x1 = np.zeros((nmo, nmo), dtype=x.dtype)
            x1[var_a] = x[:nvira * nocca]
            x1[var_b] += x[nvira * nocca:]
            return x1[uniq_ab]

        g, h_op, h_diag = newton_ah.gen_g_hop_rohf(mf, mo_coeff, mo_occ)

        # dL/dtheta with L = Tr[G_P P], P = Pa + Pb and one orbital set.
        # Both spins see the same G_P (the DMET construction only uses the
        # total density), so a single moFbar feeds both branches:
        #     dL/dtheta_pq = 2 (n_q - n_p) moFbar_pq
        # which in the (alpha, beta) packing is 2*moFbar on each branch.
        #
        # NOTE gen_g_hop_rohf returns HALF the true orbital Hessian -- measured
        # d2E/dtheta2 / (x.h_op(x)) = 2.0000 in
        # examples/test_example/calib_rohf_hessian.py.  The Z-vector equation
        # H_true Z = -dL/dtheta therefore reads  h_op(Z) = -1/2 dL/dtheta,
        # i.e. the packed RHS carries 1*moFbar rather than 2*moFbar.
        rhs_uhf = np.hstack([moFbar[var_a], moFbar[var_b]])
        rhs = sum_ab(rhs_uhf)

        if self.conv_tol_cphf is not None and self.conv_tol_cphf > 0:
            import scipy.sparse.linalg as sla
            n = rhs.size
            op = sla.LinearOperator((n, n), matvec=h_op)
            precond = np.where(np.abs(h_diag) > 1e-8, h_diag, 1.0)
            M = sla.LinearOperator((n, n), matvec=lambda x: x / precond)
            z, info = sla.gmres(op, -rhs, M=M, rtol=self.conv_tol_cphf,
                                atol=0.0, maxiter=self.max_cycle_cphf * 10)
            if info != 0:
                self.log.warn('ROHF Z-vector GMRES did not converge (info=%d)',
                              info)
        else:
            z = np.zeros(rhs.size)
        self.Z = z

        # Scatter back to the two spin branches.  NOTE a (vir, docc) pair
        # belongs to BOTH var_a and var_b, so it contributes to the alpha and
        # the beta density response -- weight 2, matching the occupation
        # difference (n_i - n_a) = 2 there (1 for vir-socc and socc-docc).
        kappa = np.zeros((nmo, nmo))
        kappa[uniq_ab] = z
        self.kappa = kappa
        kappa_a = np.where(var_a, kappa, 0.0)
        kappa_b = np.where(var_b, kappa, 0.0)

        Zmat_a = mo_coeff @ kappa_a @ mo_coeff.T
        Zmat_b = mo_coeff @ kappa_b @ mo_coeff.T
        self.dPa = Zmat_a + Zmat_a.T
        self.dPb = Zmat_b + Zmat_b.T
        self.zvec_ao = self.dPa + self.dPb          # total density response

        # Energy-weighted piece: the same P_s F_s P_s structure PySCF uses for
        # the ROHF energy-weighted density (pyscf/grad/rohf.py::make_rdm1e).
        # Calibrated coefficient -1 (calib_rohf_folding.py -> -1.0396).
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
        fa = getattr(fock, 'focka', fock)
        fb = getattr(fock, 'fockb', fock)
        mo_a = mo_coeff[:, occa]
        mo_b = mo_coeff[:, occb]
        Pa = mo_a @ mo_a.T
        Pb = mo_b @ mo_b.T
        self.Pa, self.Pb = Pa, Pb
        self.zeta = Zmat_a @ fa @ Pa + Zmat_b @ fb @ Pb

        # Two-electron response of the spin-resolved ROHF Fock to the Z-vector
        # density, projected on the occupied spaces.  Calibrated coefficient
        # -1/2 (calib_rohf_folding.py -> -0.4753).
        vj, vk = mf.get_jk(mol, np.asarray((self.dPa, self.dPb)))
        veff_a = vj[0] + vj[1] - vk[0]
        veff_b = vj[0] + vj[1] - vk[1]
        self.vhf_s1occ = 0.5 * (Pa @ veff_a @ Pa + Pb @ veff_b @ Pb)

    # ------------------------------------------------------------------
    # 6. total dE/dS (Pulay channel)
    # ------------------------------------------------------------------
    def make_sbar(self):
        w_s, u_s = self.w_s, self.u_s
        # X = S^{-1/2}
        sbar = matfun_grad(w_s, u_s, self.B_X,
                           lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
        # Y = S^{+1/2}
        sbar += matfun_grad(w_s, u_s, self.B_Y,
                            lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5)

        # Orbital orthonormality: U_pq carries a symmetric -1/2 S^x_pq piece.
        # Tr[G_P dP]|_S^x = -1/2 Tr[dS (P G_P S^{-1} + S^{-1} G_P P)]  (correct
        # Pulay, includes S^{-1}).  Its occupied-occupied block in MO basis is
        #   -1/2 sum_ij Smo_ij moFbar_ij (n_i + n_j)
        # so the sbar contribution (with contract factor 2 and sym folding) is
        #   -1/4 sym(C_occ (moFbar_oo . (n_i + n_j)) C_occ^T).
        # For RHF (n = 2) this reduces to -2 C_occ moFbar_oo C_occ^T, exactly
        # the validated RHF oo term.
        mo_coeff, mo_occ = self.mf.mo_coeff, self.mf.mo_occ
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        n_occ = mo_occ[occidx]
        w = n_occ[:, None] + n_occ[None, :]
        moFbar_oo = self.moFbar[np.ix_(occidx, occidx)]
        oo = orbo @ (moFbar_oo * w) @ orbo.T
        sbar -= 0.25 * (oo + oo.T)

        # Z-vector Pulay pieces (same structure as pyscf.grad.casci)
        sbar -= self.zeta + self.zeta.T
        sbar -= self.vhf_s1occ + self.vhf_s1occ.T

        self.sbar = sbar

    # ------------------------------------------------------------------
    # 7. contraction with the AO integral derivatives
    # ------------------------------------------------------------------
    def contract(self, atmlst=None):
        mol, mf, mfgrad = self.mol, self.mf, self.mfgrad
        C, nes = self.C, self.nes
        nao = mol.nao

        dm_core, dm_act = self.dm_core, self.dm_act
        zvec_ao, aodm = self.zvec_ao, self.base.dm

        hcore_deriv = mfgrad.hcore_generator(mol)
        s1 = mfgrad.get_ovlp(mol)
        # Closed-shell veff derivative d/dR [J(D) - K(D)/2], independent of the
        # reference type.  For ROHF ``mfgrad.get_veff`` is uhf_grad.get_veff
        # (spin-resolved), which is NOT what these channels need: the frozen
        # core is doubly occupied, so both the core-core and the core-active
        # interaction are spin independent (see DMET_ROHF梯度推导.md §2).
        vR = self._veff_deriv
        vR_core = vR(dm_core)
        vR_act = vR(dm_act)
        if self.openshell:
            # The Z-vector couples to the *ROHF* Fock, which is spin resolved
            # (veff_s = J(Pa+Pb) - K(P_s)).  uhf_grad.get_veff builds exactly
            # that pair, so use mfgrad here rather than the closed-shell helper.
            dm_pair = np.asarray(self.mf.make_rdm1())
            vR_z_s = mfgrad.get_veff(mol, np.asarray((self.dPa, self.dPb)))
            vR_p_s = mfgrad.get_veff(mol, dm_pair)
        else:
            vR_z = vR(zvec_ao)
            vR_p = vR(aodm)

        # d(mu q|r s)/dR on the bra index (int2e_ip1 already returns the full
        # 5-index tensor in this PySCF build).
        eriR = mol.intor('int2e_ip1')
        eriR = einsum('Ruvwx,va,wb,xc->Ruabc', eriR, C, C, C)
        g2R = eriR.reshape(3, nao, -1) @ self.r2.reshape(nes, -1).T
        # sign convention: + (d/dR (mu q|r s)) r2 gives the force from the
        # cluster electrons (verified against central differences).
        g2R = -2.0 * einsum('xui,ui->xu', g2R, C)
        del eriR

        dm_hcore = dm_core + dm_act + zvec_ao
        dm_cas = dm_core + dm_act

        if atmlst is None:
            atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            de[k] += einsum('xij,ij->x', hcore_deriv(ia), dm_hcore)
            # mean-field-like two-electron terms (x2 folds bra and ket)
            de[k] += einsum('xij,ij->x', vR_core[:, p0:p1], dm_cas[p0:p1]) * 2
            de[k] += einsum('xij,ij->x', vR_act[:, p0:p1], dm_core[p0:p1]) * 2
            # Z-vector two-electron cross terms
            if self.openshell:
                de[k] += einsum('xij,ij->x', vR_z_s[0][:, p0:p1],
                                dm_pair[0][p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vR_z_s[1][:, p0:p1],
                                dm_pair[1][p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vR_p_s[0][:, p0:p1],
                                self.dPa[p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vR_p_s[1][:, p0:p1],
                                self.dPb[p0:p1]) * 2
            else:
                de[k] += einsum('xij,ij->x', vR_z[:, p0:p1], aodm[p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vR_p[:, p0:p1], zvec_ao[p0:p1]) * 2
            # cluster 2-RDM
            de[k] += g2R[:, p0:p1].sum(axis=1)
            # overlap / Pulay
            de[k] += einsum('xij,ij->x', s1[:, p0:p1], self.sbar[p0:p1]) * 2

        self.de = de
        return de

    # ------------------------------------------------------------------
    def kernel(self, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.check_support()
        self.decompose()
        self.make_densities()
        self.orb_grad()
        self.lo_grad()
        self.solve_z()
        self.make_sbar()
        de = self.contract(atmlst)
        de = de + self.mfgrad.grad_nuc(atmlst=atmlst)
        self.de = de
        self.log.timer('SSDMET nuclear gradient', *cput0)
        if self.verbose >= logger.INFO:
            self._write(de, atmlst)
        return de

    def _write(self, de, atmlst=None):
        mol = self.mol
        if atmlst is None:
            atmlst = range(mol.natm)
        self.log.info('--------- SSDMET gradients ---------')
        self.log.info('%-5s %-14s %-14s %-14s', 'atom', 'x', 'y', 'z')
        for k, ia in enumerate(atmlst):
            self.log.info('%-3d %-3s %14.9f %14.9f %14.9f',
                          ia, mol.atom_symbol(ia), de[k, 0], de[k, 1], de[k, 2])
        self.log.info('------------------------------------')


Gradients = SSDMETGradients
