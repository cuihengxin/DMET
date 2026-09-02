"""Export the DMET embedded-space orbitals (impurity / bath / frozen) to Molden.

Works for every DMET object in this package that shares the attributes built
by ``build()``:

- ``SSDMET``   (``embed_sim.ssdmet.SSDMET``)
- ``DFSSDMET`` (``embed_sim.df.DFSSDMET``)
- ``AODMET``   (``embed_sim.aodmet.AODMET``)
- ``DFAODMET`` (``embed_sim.df.DFAODMET``)

The embedded subspace is split into four blocks, and each orbital is labelled
through the Molden ``Sym`` field so that any Molden viewer (Chemcraft / VMD /
IQmol / Molden) shows whether it is an impurity (``imp``), bath (``bath``),
frozen occupied (``fo``) or frozen virtual (``fv``) orbital.

Notes
-----
- ``es_natorb`` (default True) rotates the impurity+bath block into embedded
  natural orbitals and therefore mixes impurity and bath.  This module
  reconstructs the *pure* impurity and bath orbitals from the stored 1-RDM and
  the Löwdin transform, so the labels are meaningful regardless of
  ``es_natorb``.
- ``fo_orb`` / ``fv_orb`` are taken directly from the object (they are already
  the frozen-occupied / frozen-virtual orbitals in the AO basis).
- If ``bath_option`` (MP2 bath expansion) was used, the extra MP2 bath orbitals
  are *not* reconstructed here; only the mean-field imp/bath/fo/fv blocks are
  exported.

Example
-------
    from embed_sim import ssdmet, orbital_export
    mydmet = ssdmet.SSDMET(mf, title='h2o', imp_idx=[0], bath_norb='per_bond')
    mydmet.build()
    orbital_export.export_embedding_molden(mydmet)  # -> h2o_embedding.molden
"""

import numpy as np
from pyscf import tools

from embed_sim.bath_selection import partition_env_by_bath_count


def _get_lowdin(dmet):
    """Return ``(caolo, cloao)``: the AO <-> Löwdin-orthonormal transforms.

    Prefer the transforms cached on the object (``SSDMET`` / ``DFSSDMET``);
    for ``AODMET`` / ``DFAODMET`` they are not cached, so recompute them via
    ``lowdin_orth()``.
    """
    caolo = getattr(dmet, 'caolo', None)
    cloao = getattr(dmet, 'cloao', None)
    if caolo is None or cloao is None:
        # SSDMET.lowdin_orth -> (ldm, caolo, cloao)
        # AODMET.lowdin_orth -> (ldm, caolo, cloao, ovlp)
        out = dmet.lowdin_orth()
        caolo, cloao = out[1], out[2]
    return caolo, cloao


def _select_bath_idx(occ_env, dmet, nbath):
    """Environment-natural-orbital indices selected as bath.

    Mirrors the selection in ``build_embeded_subspace``: threshold-based when
    ``bath_norb is None``, fixed-size ("one bath per bond") otherwise.
    """
    thres = dmet.threshold
    if dmet.bath_norb is None:
        bath_idx = np.nonzero((occ_env >= thres) & (occ_env <= 2.0 - thres))[0]
    else:
        bath_idx, _, _ = partition_env_by_bath_count(
            occ_env, nbath, thres=thres, core_cutoff=dmet.bath_core_cutoff)
    return bath_idx


def collect_embedding_orbitals(dmet):
    """Return the four orbital blocks in the AO basis + occupations + labels.

    Returns
    -------
    mo_coeff : (nao, ntot) ndarray
        concatenated ``[fo | imp | bath | fv]`` orbitals in the AO basis.
    occ : (ntot,) ndarray
        occupations: ``fo`` = 2, ``fv`` = 0; ``imp`` and ``bath`` carry their
        true natural occupations (impurity block / environment block), so
        occupied vs. virtual impurity orbitals are distinguished.
    labels : list[str]
        ``'fo'`` / ``'imp'`` / ``'bath'`` / ``'fv'`` per orbital.
    counts : dict
        number of orbitals in each block.
    """
    mol = dmet.mol
    nao = mol.nao
    imp_idx = np.asarray(dmet.imp_idx, dtype=int)
    env_idx = np.array([i for i in range(nao) if i not in imp_idx], dtype=int)
    nimp = len(imp_idx)
    nbath = dmet.nes - nimp

    caolo, cloao = _get_lowdin(dmet)

    # Environment natural orbitals in the Löwdin basis.
    ldm = cloao @ dmet.dm @ cloao.conj().T
    occ_env, orb_env = np.linalg.eigh(ldm[env_idx][:, env_idx])

    bath_idx = _select_bath_idx(occ_env, dmet, nbath)

    # Impurity natural orbitals: eigenvectors of the impurity block of the
    # 1-RDM (Löwdin basis), sorted descending so occupied (occ ~ 2) impurity
    # orbitals come first and virtual (occ ~ 0) ones last.
    occ_imp, orb_imp = np.linalg.eigh(ldm[imp_idx][:, imp_idx])
    occ_imp, orb_imp = occ_imp[::-1], orb_imp[:, ::-1]
    imp_ao = caolo[:, imp_idx] @ orb_imp

    # Pure bath orbitals: environment natural orbitals -> AO.
    bath_ao = caolo[:, env_idx] @ orb_env[:, bath_idx]

    fo = dmet.fo_orb
    fv = dmet.fv_orb

    mo_coeff = np.hstack([fo, imp_ao, bath_ao, fv])
    occ = np.concatenate([
        np.full(fo.shape[1], 2.0),
        occ_imp,
        occ_env[bath_idx],
        np.zeros(fv.shape[1]),
    ])
    # clip numerical noise (X2C complex arithmetic can give e.g. -1e-16).
    occ = np.clip(occ, 0.0, 2.0)
    labels = (['fo'] * fo.shape[1] + ['imp'] * imp_ao.shape[1]
              + ['bath'] * bath_ao.shape[1] + ['fv'] * fv.shape[1])
    counts = {'fo': fo.shape[1], 'imp': imp_ao.shape[1],
              'bath': bath_ao.shape[1], 'fv': fv.shape[1]}
    return mo_coeff, occ, labels, counts


def export_embedding_molden(dmet, filename=None, verbose=True):
    """Write the DMET embedded-space orbitals to a Molden file.

    Parameters
    ----------
    dmet : SSDMET | DFSSDMET | AODMET | DFAODMET
        a DMET object whose ``build()`` has been called.
    filename : str, optional
        output path; defaults to ``<title>_embedding.molden``.
    verbose : bool
        print the block -> MO-index mapping.

    Returns
    -------
    filename : str
    """
    mo_coeff, occ, labels, counts = collect_embedding_orbitals(dmet)
    if filename is None:
        filename = dmet.title + '_embedding.molden'

    tools.molden.from_mo(dmet.mol, filename, mo_coeff,
                         symm=labels, occ=occ)

    if verbose:
        print(f'Exported {filename}:')
        start = 0
        for block in ('fo', 'imp', 'bath', 'fv'):
            n = counts[block]
            if n:
                print(f'  {block:>4s}: MO {start+1:4d} .. {start+n:4d}  ({n} orbitals)')
            start += n
    return filename
