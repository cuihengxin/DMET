'''
Pair-resolved (PNO) bath expansion wired into single-shot DMET.

This module adds `embed_sim.pno_bath` support to `SSDMET` (and its density
fitting / AO variants) *without touching* `ssdmet.py` or `BNO_bath.py`.

The expansion is applied as a post-processing step on an already built DMET
object: the plain 1-RDM bath is constructed first, the embedded mean field is
solved, and the frozen occupied / virtual spaces are then rotated so that the
pair-selected orbitals move into the cluster.  Because the rotations returned by
`pno_bath.build_pno_rotations` are orthogonal *within* each frozen space, the
total span is preserved and the DMET exactness relation

    E(embedded mean field) + E(frozen occupied) == E(full mean field)

still holds after the expansion.  `check_exactness` below verifies this.

Usage
-----
    from embed_sim.ssdmet_pno import SSDMET_PNO

    mydmet = SSDMET_PNO(mf, title='h2o', imp_idx='0 O', es_natorb=False,
                        pno_option=dict(t_pno=1e-7, t_pair=1e-5))
    mydmet.build()

or, on an existing object built by any SSDMET flavour:

    from embed_sim.ssdmet_pno import expand_bath_pno
    expand_bath_pno(mydmet, t_pno=1e-7)
'''

from functools import reduce

import numpy as np

from pyscf import lib

from embed_sim.ssdmet import SSDMET
from embed_sim.pno_bath import build_pno_rotations


def _require_lo(mydmet):
    '''The expansion works in the LO basis, so a fresh build() is required.'''
    for attr in ('caolo', 'cloao', 'lo_cloes'):
        if getattr(mydmet, attr, None) is None:
            raise RuntimeError(
                'PNO bath expansion needs the Lowdin transformation matrices, '
                'which are not stored in the checkpoint file. Rebuild with '
                'build(chk_fname_load=None) before expanding.')


def expand_bath_pno(mydmet, verbose=None, **pno_kwargs):
    '''Expand the bath of a built DMET object with pair natural orbitals.

    Parameters
    ----------
    mydmet : SSDMET (or subclass)
        An object on which `build()` has already run in this session.
    **pno_kwargs
        Forwarded to `pno_bath.build_pno_rotations` (`t_pno`, `t_pair`,
        `nbath_occ`, `nbath_vir`, `e_target`, `score`, `occ_side`, `vir_side`).

    Returns
    -------
    info : dict
        Diagnostics from the selection, plus 'nes_before' / 'nes_after' and the
        DMET exactness deviation after the expansion.
    '''
    _require_lo(mydmet)
    log = mydmet.log
    if mydmet.es_natorb:
        raise RuntimeError('es_natorb must be False when expanding the bath')
    if mydmet.es_mf is None:
        raise RuntimeError('run build() before expanding the bath')

    cloao, caolo = mydmet.cloao, mydmet.caolo
    nes_before, nfo_before, nfv_before = mydmet.nes, mydmet.nfo, mydmet.nfv

    lo2es = lib.dot(cloao, mydmet.es_orb)
    lo2core = lib.dot(cloao, mydmet.fo_orb)
    lo2vir = lib.dot(cloao, mydmet.fv_orb)

    rot, info = build_pno_rotations(
        mydmet.mf_or_cas, mydmet.es_mf,
        mydmet.es_orb, mydmet.fo_orb, mydmet.fv_orb,
        verbose=verbose if verbose is not None else mydmet.verbose,
        **pno_kwargs)

    lo2bath = np.hstack((lib.dot(lo2core, rot['core_bath']),
                         lib.dot(lo2vir, rot['vir_bath'])))
    lo2eo = np.hstack((lo2es, lo2bath))

    mydmet.es_orb = lib.dot(caolo, lo2eo)
    mydmet.fo_orb = lib.dot(caolo, lib.dot(lo2core, rot['core_rest']))
    mydmet.fv_orb = lib.dot(caolo, lib.dot(lo2vir, rot['vir_rest']))

    mydmet.nes = mydmet.es_orb.shape[-1]
    mydmet.nfo = mydmet.fo_orb.shape[-1]
    mydmet.nfv = mydmet.fv_orb.shape[-1]

    nao = mydmet.mol.nao
    log.info('embedded cluster orbitals %d -> %d (%.2f%% -> %.2f%% of %d AOs)',
             nes_before, mydmet.nes, 100 * nes_before / nao,
             100 * mydmet.nes / nao, nao)
    log.info('frozen occupied %d -> %d, frozen virtual %d -> %d',
             nfo_before, mydmet.nfo, nfv_before, mydmet.nfv)

    # rebuild everything that depends on the partition
    mydmet.es_int1e = mydmet.make_es_int1e()
    mydmet.es_int2e = mydmet.make_es_int2e()
    mydmet.es_dm = mydmet.make_es_dm(mydmet.open_shell, lo2eo, cloao, mydmet.dm)
    mydmet.es_mf = mydmet.ROHF()
    mydmet.calc_fo_ene()

    dev = mydmet.es_mf.e_tot + mydmet.fo_ene - mydmet.mf_or_cas.e_tot
    log.info('energy from frozen occupied orbitals = %.12f', mydmet.fo_ene)
    log.info('deviation from DMET exact condition = %.3e', dev)
    if abs(dev) > 1e-7:
        log.warn('DMET exactness broken after PNO expansion (%.3e); '
                 'the frozen/bath rotation may have lost orthonormality', dev)

    info.update(nes_before=nes_before, nes_after=mydmet.nes,
                nfo_after=mydmet.nfo, nfv_after=mydmet.nfv,
                exactness_deviation=dev)
    mydmet.pno_info = info
    return info


def check_exactness(mydmet, tol=1e-8):
    '''Orthonormality and span checks on the current DMET partition.

    Returns a dict of the measured deviations; raises nothing so it can be used
    as a diagnostic inside test scripts.
    '''
    S = mydmet.mol.intor_symmetric('int1e_ovlp')
    C = np.hstack((mydmet.es_orb, mydmet.fo_orb, mydmet.fv_orb))
    n = C.shape[1]
    orth = np.abs(reduce(np.dot, (C.conj().T, S, C)) - np.eye(n)).max()
    complete = abs(n - mydmet.mol.nao)
    dev = mydmet.es_mf.e_tot + mydmet.fo_ene - mydmet.mf_or_cas.e_tot
    return dict(orthonormality=float(orth),
                missing_orbitals=int(complete),
                exactness_deviation=float(dev),
                passed=bool(orth < tol and complete == 0 and abs(dev) < 1e-7))


class SSDMET_PNO(SSDMET):
    '''SSDMET with an optional pair-resolved (PNO) bath expansion.

    `pno_option` is a dict forwarded to `pno_bath.build_pno_rotations`; set it
    to None to fall back to plain SSDMET behaviour.  It is mutually exclusive
    with the `bath_option` (BNO) machinery of the base class.
    '''

    def __init__(self, *args, pno_option=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pno_option = pno_option
        self.pno_info = None
        if pno_option is not None:
            if self.bath_option is not None:
                raise ValueError('pno_option and bath_option are mutually '
                                 'exclusive; pick one bath expansion scheme')
            if self.es_natorb:
                raise ValueError('es_natorb must be False when using pno_option')

    def build(self, restore_imp=False, iaopao=False, chk_fname_load=None,
              save_chk=True, xc=None):
        '''Build the embedded space, then expand the bath with PNOs.

        Unlike the base class this does *not* load a checkpoint by default: the
        expansion needs the Lowdin transformation matrices, which the checkpoint
        does not store, and an already expanded checkpoint is indistinguishable
        from a plain one.  Checkpoints are written to `<title>_pno_dmet_chk.h5`
        so they never collide with plain SSDMET runs.
        '''
        # stage 1: plain 1-RDM bath (checkpoint written only after stage 2)
        super().build(restore_imp=restore_imp, iaopao=iaopao,
                      chk_fname_load=chk_fname_load, save_chk=False, xc=xc)

        # stage 2: pair-resolved expansion
        if self.pno_option is not None:
            self.log.info('')
            self.log.info('=' * 60)
            self.log.info('pair-resolved (PNO) bath expansion')
            self.log.info('=' * 60)
            expand_bath_pno(self, **self.pno_option)

        if save_chk:
            self.save_chk(self.title + '_pno_dmet_chk.h5')
        return self
