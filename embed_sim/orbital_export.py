"""Export final DMET embedding orbitals and their spatial coverage to Molden.

The exporter works with ``SSDMET``, ``DFSSDMET``, ``AODMET`` and
``DFAODMET`` objects after ``build()``.  If MP2/BNO or PNO bath expansion was
requested, the added orbitals are read from the *final* ``dmet.es_orb`` and are
therefore included in the output.

``export_embedding_molden`` writes a complete orbital partition with reference
occupations.  Its possible Molden ``Sym`` labels are ``fo``, ``imp``, ``bath``,
``mp2_occ``, ``mp2_vir`` and ``fv``.

``export_embedding_projector_molden`` is intended for comparing IPLO, IAO and
AVAS impurity definitions.  It writes only the final embedded space and gives
every orbital occupation 1.0.  The total density displayed by a Molden viewer
is consequently the rotation-invariant subspace coverage

    rho_ES(r) = sum_{p in ES} |phi_p(r)|^2.

That quantity describes where the selected embedding space lives.  It is not a
physical MP2 electron density; the MP2 calculation here selects bath natural
orbitals but does not supply a correlated density for the final impurity
solver.

``export_embedding_density_cube`` evaluates that sum on a real-space grid and
writes one scalar ``.cube`` file.  Use this function when the viewer otherwise
shows the Molden orbitals one at a time.

Example
-------
    from embed_sim import orbital_export

    mydmet.build()
    orbital_export.export_embedding_molden(mydmet)
    orbital_export.export_embedding_projector_molden(mydmet)
    orbital_export.export_embedding_density_cube(mydmet)
"""

import numpy as np
from pyscf import tools

from embed_sim.bath_selection import (count_imp_env_bonds,
                                      partition_env_by_bath_count)


def _require_built(dmet):
    if getattr(dmet, 'es_orb', None) is None:
        raise RuntimeError('embedded subspace is not built; run dmet.build() first')


def _get_lowdin(dmet):
    """Return the AO <- LO and LO <- AO transformations used by ``build``."""
    caolo = getattr(dmet, 'caolo', None)
    cloao = getattr(dmet, 'cloao', None)
    if caolo is None or cloao is None:
        # SSDMET.lowdin_orth -> (ldm, caolo, cloao)
        # AODMET.lowdin_orth -> (ldm, caolo, cloao, ovlp)
        out = dmet.lowdin_orth()
        caolo, cloao = out[1], out[2]
    return caolo, cloao


def _select_bath_idx(occ_env, dmet):
    """Mirror the pre-MP2 mean-field bath selection used by ``build``."""
    thres = dmet.threshold
    bath_norb = getattr(dmet, 'bath_norb', None)
    if bath_norb is None:
        return np.nonzero(
            (occ_env >= thres) & (occ_env <= 2.0 - thres))[0]

    if isinstance(bath_norb, str):
        if bath_norb.lower() not in ('per_bond', 'perbond', 'one_per_bond'):
            raise ValueError('unknown bath_norb string: %r' % bath_norb)
        bath_norb = count_imp_env_bonds(dmet.mol, dmet.imp_idx)
    bath_idx, _, _ = partition_env_by_bath_count(
        occ_env, int(bath_norb), thres=thres,
        core_cutoff=getattr(dmet, 'bath_core_cutoff', 0.5))
    return bath_idx


def _orbital_occupations(dmet, mo_coeff):
    """Return reference 1-RDM populations of orthonormal AO-basis orbitals."""
    if mo_coeff.shape[1] == 0:
        return np.zeros(0)
    dm = np.asarray(dmet.dm)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    ovlp = dmet.mol.intor_symmetric('int1e_ovlp')
    smo = ovlp @ mo_coeff
    occ = np.einsum('pi,pq,qi->i', smo.conj(), dm, smo, optimize=True)
    return np.clip(np.real_if_close(occ).real, 0.0, 2.0)


def _mean_field_partition(dmet):
    """Reconstruct pure impurity and bath orbitals before MP2 expansion."""
    nao = dmet.mol.nao
    imp_idx = np.asarray(dmet.imp_idx, dtype=int)
    env_idx = np.array([i for i in range(nao) if i not in imp_idx], dtype=int)
    caolo, cloao = _get_lowdin(dmet)
    ldm = cloao @ dmet.dm @ cloao.conj().T

    occ_env, orb_env = np.linalg.eigh(ldm[np.ix_(env_idx, env_idx)])
    bath_idx = _select_bath_idx(occ_env, dmet)

    # The rotation is internal to the impurity span, so its projector is
    # unchanged but individual orbitals are easier to inspect.
    occ_imp, orb_imp = np.linalg.eigh(ldm[np.ix_(imp_idx, imp_idx)])
    occ_imp, orb_imp = occ_imp[::-1], orb_imp[:, ::-1]
    imp_ao = caolo[:, imp_idx] @ orb_imp
    bath_ao = caolo[:, env_idx] @ orb_env[:, bath_idx]
    return imp_ao, occ_imp, bath_ao, occ_env[bath_idx]


def _split_added_bath(dmet, added_ao):
    """Split MP2-added orbitals by their parent occupied/virtual space."""
    occ = _orbital_occupations(dmet, added_ao)
    occ_mask = occ > 1.0
    return (added_ao[:, occ_mask], occ[occ_mask],
            added_ao[:, ~occ_mask], occ[~occ_mask])


def _embedding_labels(dmet):
    """Labels in the native column order of final ``dmet.es_orb``."""
    _require_built(dmet)
    nimp = len(dmet.imp_idx)
    nes = dmet.es_orb.shape[1]
    if getattr(dmet, 'es_natorb', False):
        # Embedded natural orbitals mix the impurity and bath columns.
        return ['es_nat'] * nes

    labels = ['imp'] * nimp
    try:
        _, _, bath_ao, _ = _mean_field_partition(dmet)
        nbath = bath_ao.shape[1]
    except Exception:
        # Checkpoints do not retain all localization settings.  The final span
        # can still be exported exactly; classify expanded columns from their
        # reference occupations when reconstruction is unavailable.
        occ = _orbital_occupations(dmet, dmet.es_orb[:, nimp:])
        expanded = (getattr(dmet, 'bath_option', None) is not None or
                    getattr(dmet, 'pno_info', None) is not None)
        if not expanded:
            return labels + ['bath'] * (nes - nimp)
        tol = max(10.0 * getattr(dmet, 'threshold', 1e-12), 1e-8)
        for value in occ:
            if value > 2.0 - tol:
                labels.append('mp2_occ')
            elif value < tol:
                labels.append('mp2_vir')
            else:
                labels.append('bath')
        return labels

    if nimp + nbath > nes:
        raise RuntimeError('final embedded space is smaller than its mean-field space')
    labels += ['bath'] * nbath
    added_occ = _orbital_occupations(dmet, dmet.es_orb[:, nimp + nbath:])
    labels += ['mp2_occ' if value > 1.0 else 'mp2_vir'
               for value in added_occ]
    return labels


def collect_embedding_orbitals(dmet):
    """Return the complete post-expansion orbital partition in the AO basis.

    Returns
    -------
    mo_coeff : (nao, nao) ndarray
        Orbitals ordered as ``[fo | imp | bath | mp2_occ | mp2_vir | fv]``.
    occ : (nao,) ndarray
        Mean-field reference occupations in the exported orbital basis.
    labels : list[str]
        Molden symmetry label for each orbital.
    counts : dict[str, int]
        Number of orbitals in each labelled block.
    """
    _require_built(dmet)
    imp_ao, occ_imp, bath_ao, occ_bath = _mean_field_partition(dmet)
    nbase = imp_ao.shape[1] + bath_ao.shape[1]
    if dmet.es_orb.shape[1] < nbase:
        raise RuntimeError('final embedded space is smaller than its mean-field space')

    added_ao = dmet.es_orb[:, nbase:]
    mp2_occ_ao, mp2_occ, mp2_vir_ao, mp2_vir = _split_added_bath(
        dmet, added_ao)
    fo = dmet.fo_orb
    fv = dmet.fv_orb

    mo_coeff = np.hstack([fo, imp_ao, bath_ao,
                          mp2_occ_ao, mp2_vir_ao, fv])
    occ = np.concatenate([
        np.full(fo.shape[1], 2.0), occ_imp, occ_bath,
        mp2_occ, mp2_vir, np.zeros(fv.shape[1]),
    ])
    occ = np.clip(occ, 0.0, 2.0)
    labels = (['fo'] * fo.shape[1] + ['imp'] * imp_ao.shape[1]
              + ['bath'] * bath_ao.shape[1]
              + ['mp2_occ'] * mp2_occ_ao.shape[1]
              + ['mp2_vir'] * mp2_vir_ao.shape[1]
              + ['fv'] * fv.shape[1])
    counts = {'fo': fo.shape[1], 'imp': imp_ao.shape[1],
              'bath': bath_ao.shape[1],
              'mp2_occ': mp2_occ_ao.shape[1],
              'mp2_vir': mp2_vir_ao.shape[1],
              'fv': fv.shape[1]}

    if mo_coeff.shape[1] != dmet.mol.nao:
        raise RuntimeError(
            'exported partition is incomplete: got %d of %d orbitals'
            % (mo_coeff.shape[1], dmet.mol.nao))
    return mo_coeff, occ, labels, counts


def export_embedding_molden(dmet, filename=None, verbose=True):
    """Write the complete final DMET orbital partition to a Molden file."""
    mo_coeff, occ, labels, counts = collect_embedding_orbitals(dmet)
    if filename is None:
        filename = dmet.title + '_embedding.molden'

    tools.molden.from_mo(dmet.mol, filename, mo_coeff,
                         symm=labels, occ=occ)

    if verbose:
        print('Exported %s:' % filename)
        start = 0
        for block in ('fo', 'imp', 'bath', 'mp2_occ', 'mp2_vir', 'fv'):
            n = counts[block]
            if n:
                print('  %8s: MO %4d .. %4d  (%d orbitals)' %
                      (block, start + 1, start + n, n))
            start += n
    return filename


def export_embedding_projector_molden(dmet, filename=None, normalize=False,
                                      verbose=True):
    """Export the final embedding-space coverage with unit occupations.

    In a Molden viewer, display the total density of this file to compare the
    spatial coverage selected by IPLO, IAO or AVAS.  Because all occupations
    are nonzero, virtual-side MP2 bath orbitals remain visible in that density.

    Set ``normalize=True`` when the methods produce different numbers of
    embedded orbitals and the comparison should emphasize distribution shape
    rather than subspace size.  It changes every occupation from 1 to
    ``1 / nes``, giving every file the same integrated coverage density.
    """
    _require_built(dmet)
    if filename is None:
        filename = dmet.title + '_embedding_projector.molden'

    labels = _embedding_labels(dmet)
    occ_value = 1.0 / dmet.es_orb.shape[1] if normalize else 1.0
    tools.molden.from_mo(dmet.mol, filename, dmet.es_orb,
                         symm=labels,
                         occ=np.full(dmet.es_orb.shape[1], occ_value))

    if verbose:
        counts = {label: labels.count(label) for label in dict.fromkeys(labels)}
        summary = ', '.join('%s=%d' % item for item in counts.items())
        print('Exported %s: %d final embedded orbitals (%s)' %
              (filename, dmet.es_orb.shape[1], summary))
        if normalize:
            print('  Occ=1/nes is normalized coverage, not an MP2 occupation.')
        else:
            print('  Occ=1 is the subspace-projector density, not an MP2 occupation.')
    return filename


def _select_embedding_component(dmet, component):
    """Select columns of the final embedded space for a combined density."""
    labels = np.asarray(_embedding_labels(dmet))
    component = component.lower().replace('-', '_')
    aliases = {
        'all': 'all',
        'embedding': 'all',
        'es': 'all',
        'imp': 'imp',
        'impurity': 'imp',
        'bath': 'bath',
        'mf_bath': 'mf_bath',
        'mean_field_bath': 'mf_bath',
        'mp2': 'mp2',
        'mp2_added': 'mp2',
        'mp2_occ': 'mp2_occ',
        'mp2_vir': 'mp2_vir',
    }
    if component not in aliases:
        raise ValueError(
            'unknown component %r; choose all, imp, bath, mf_bath, mp2, '
            'mp2_occ or mp2_vir' % component)
    component = aliases[component]

    if component == 'all':
        mask = np.ones(labels.size, dtype=bool)
    elif component == 'bath':
        mask = np.isin(labels, ('bath', 'mp2_occ', 'mp2_vir'))
    elif component == 'mf_bath':
        mask = labels == 'bath'
    elif component == 'mp2':
        mask = np.isin(labels, ('mp2_occ', 'mp2_vir'))
    else:
        mask = labels == component

    if not np.any(mask):
        if np.all(labels == 'es_nat'):
            reason = ' (es_natorb=True mixes impurity and bath orbitals)'
        else:
            reason = ''
        raise ValueError('embedding component %r is empty%s' %
                         (component, reason))
    return dmet.es_orb[:, mask], labels[mask].tolist()


def embedding_projector_dm(dmet, component='all', normalize=False):
    """Build the AO density matrix for a combined embedding-space coverage.

    The returned matrix is ``C C^H`` for the selected orthonormal orbitals.
    It is a subspace projector represented in the AO basis, not an MP2 1-RDM.
    With ``normalize=True`` it is divided by the number of selected orbitals so
    that different-size spaces have equal integrated density.
    """
    _require_built(dmet)
    coeff, _ = _select_embedding_component(dmet, component)
    dm = coeff @ coeff.conj().T
    if normalize:
        dm /= coeff.shape[1]
    return np.real_if_close(dm)


def export_embedding_density_cube(dmet, filename=None, component='all',
                                  normalize=False, nx=80, ny=80, nz=80,
                                  resolution=None, margin=3.0, verbose=True):
    """Write the combined spatial distribution of selected orbitals to Cube.

    Parameters
    ----------
    component : str
        ``'all'`` for the final MP2-expanded embedding space; ``'imp'`` for
        impurity only; ``'bath'`` for mean-field plus MP2 bath; ``'mf_bath'``
        for the pre-MP2 bath; or ``'mp2'``, ``'mp2_occ'`` and ``'mp2_vir'`` for
        the added MP2 bath orbitals.
    normalize : bool
        Divide by the number of selected orbitals.  Recommended when comparing
        IPLO, IAO and AVAS spaces of different dimensions.
    nx, ny, nz, resolution, margin
        Real-space grid controls forwarded to ``pyscf.tools.cubegen.density``.

    Returns
    -------
    filename : str
        Path of the scalar Cube file.  Opening it displays the sum over all
        selected orbitals rather than individual molecular orbitals.
    """
    coeff, labels = _select_embedding_component(dmet, component)
    if filename is None:
        suffix = '' if component.lower() in ('all', 'embedding', 'es') \
            else '_' + component.lower().replace('-', '_')
        filename = dmet.title + '_embedding%s_density.cube' % suffix

    dm = coeff @ coeff.conj().T
    if normalize:
        dm /= coeff.shape[1]
    dm = np.real_if_close(dm)
    tools.cubegen.density(dmet.mol, filename, dm, nx=nx, ny=ny, nz=nz,
                          resolution=resolution, margin=margin)

    if verbose:
        unique = {label: labels.count(label) for label in dict.fromkeys(labels)}
        summary = ', '.join('%s=%d' % item for item in unique.items())
        norm_text = ', normalized to integral 1' if normalize else ''
        print('Exported %s: combined %s density from %d orbitals (%s%s)' %
              (filename, component, coeff.shape[1], summary, norm_text))
    return filename
