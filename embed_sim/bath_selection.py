"""Bath-orbital selection helpers for DMET.

This module implements the *one bath orbital per bond* bath-orbital selection
scheme (Sun & Chan, JCTC 10, 3784 (2014)), originally used in the QC-DMET SN2
test.  Instead of taking *all* environment natural orbitals with fractional
occupation (the default threshold-based selection), the user fixes the number
of bath orbitals ``nbath`` and the scheme keeps the ``nbath`` environment
natural orbitals whose occupation is closest to 1 (i.e. the most entangled
ones).  For closed-shell fragments this number is naturally chosen as the
number of bonds connecting the impurity to the environment.

Two public functions are provided:

- :func:`count_imp_env_bonds`: number of impurity-environment bonds from the
  molecular geometry (covalent-radii connectivity).
- :func:`partition_env_by_bath_count`: split the environment natural orbitals
  into bath / frozen occupied / frozen virtual for a fixed bath count.
"""

import numpy as np


# Covalent radii in Angstrom (Cordero et al., J. Chem. Phys. 128, 014102 (2008)).
# Only the elements needed by the small-molecule tests are listed; extend the
# table when testing other elements.
COVALENT_RADII = {
    'H': 0.31, 'Li': 1.28, 'B': 0.84, 'C': 0.76, 'N': 0.71,
    'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03,
    'Ca': 1.76, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32,
    'Zn': 1.22, 'Br': 1.20, 'I': 1.39,
}


def imp_atom_indices(mol, imp_idx):
    """Map impurity AO indices to the atom indices they belong to.

    Args:
        mol: PySCF molecule.
        imp_idx: list of AO indices defining the impurity.

    Returns:
        sorted list of unique atom indices carrying impurity AOs.
    """
    aoslices = mol.aoslice_by_atom()
    # columns 2:4 are the AO (not shell) index ranges of each atom
    ao_ranges = aoslices[:, 2:4]
    atom_ids = set()
    for ao in imp_idx:
        for iatom, (start, end) in enumerate(ao_ranges):
            if start <= ao < end:
                atom_ids.add(iatom)
                break
    return sorted(atom_ids)


def count_imp_env_bonds(mol, imp_idx, tol=1.3):
    """Count bonds between the impurity atoms and the environment atoms.

    Two atoms are considered bonded when their distance is smaller than
    ``tol`` times the sum of their covalent radii.  The count is the number of
    impurity-environment *atom pairs* (for a triple bond such as N2 the pair
    still counts as one bond; see the README for the multiple-bond caveat).

    Args:
        mol: PySCF molecule.
        imp_idx: list of impurity AO indices (or atom indices; resolved
            automatically).
        tol: connectivity tolerance factor.

    Returns:
        number of impurity-environment bonds (int).
    """
    if len(imp_idx) == 0:
        return 0
    imp_atoms = imp_atom_indices(mol, imp_idx)
    try:
        coords = np.asarray(mol.atom_coords(unit='Angstrom'))
    except TypeError:  # older PySCF: convert Bohr -> Angstrom
        coords = np.asarray(mol.atom_coords()) * 0.529177210903
    nbond = 0
    for i in imp_atoms:
        r_i = COVALENT_RADII.get(mol.atom_symbol(i).title(), 1.0)
        for j in range(mol.natm):
            if j in imp_atoms:
                continue
            r_j = COVALENT_RADII.get(mol.atom_symbol(j).title(), 1.0)
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < tol * (r_i + r_j):
                nbond += 1
    return nbond


def partition_env_by_bath_count(occ_env, bath_norb, thres=1e-12,
                                core_cutoff=0.5):
    """Partition environment natural orbitals for a fixed bath count.

    The ``bath_norb`` environment natural orbitals with occupation closest to 1
    are selected as bath; the remaining orbitals are classified as frozen
    occupied (occupation > ``2 - core_cutoff``) or frozen virtual (occupation
    < ``core_cutoff``).  A remaining orbital with occupation in
    ``[core_cutoff, 2-core_cutoff]`` means the requested bath count is too
    small to freeze the rest of the environment consistently; a
    :class:`ValueError` is raised (mirroring the QC-DMET assertion).

    Args:
        occ_env: 1-D array of environment natural-orbital occupations
            (ascending).
        bath_norb: desired number of bath orbitals.
        thres: occupation threshold used by the default (threshold-based)
            selection, kept for API compatibility.
        core_cutoff: occupations of the non-bath orbitals must lie outside
            ``[core_cutoff, 2-core_cutoff]``.

    Returns:
        tuple ``(bath_idx, fo_idx, fv_idx)`` of integer index arrays, where
        ``bath_idx`` contains the selected bath orbitals (most entangled
        first) and ``fo_idx``/``fv_idx`` the frozen occupied/virtual ones.
    """
    occ_env = np.asarray(occ_env)
    nenv = occ_env.size
    if bath_norb < 0 or bath_norb > nenv:
        raise ValueError(
            f'bath_norb={bath_norb} out of range [0, {nenv}] (environment size)')

    # distance of each occupation from the closest integer (0 or 2):
    # min(occ, 2-occ) is maximal for occupations closest to 1.
    frac = np.minimum(occ_env, 2.0 - occ_env)
    order = np.argsort(-frac, kind='stable')
    bath_idx = order[:bath_norb]
    rest_idx = order[bath_norb:]

    occ_rest = occ_env[rest_idx]
    fo_idx = rest_idx[occ_rest > 2.0 - core_cutoff]
    fv_idx = rest_idx[occ_rest < core_cutoff]
    bad_idx = rest_idx[(occ_rest >= core_cutoff) &
                       (occ_rest <= 2.0 - core_cutoff)]
    if bad_idx.size > 0:
        bad_occ = ', '.join(f'{occ_env[i]:.3f}' for i in bad_idx)
        raise ValueError(
            f'bath_norb={bath_norb} too small: environment orbitals {bad_idx.tolist()} '
            f'with occupations [{bad_occ}] are too entangled to be frozen. '
            'Increase bath_norb (or use the threshold-based selection).')
    return bath_idx, fo_idx, fv_idx
