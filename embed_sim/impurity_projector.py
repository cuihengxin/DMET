"""Build DMET impurity orbitals from an independent projection basis.

The parent molecular AO basis is kept for the Hamiltonian.  A smaller
reference basis (for example, user-contracted Co 3d or Ce 4f/5d pGTOs) is
projected into that parent AO space and only its selected functions are used
to define the impurity projector.  This is deliberately different from AVAS:
AVAS returns rotations internal to the occupied and virtual MO spaces,
whereas a DMET fragment basis must generally mix those spaces to generate a
mean-field Schmidt bath.

All coefficient matrices in this module use the PySCF convention: rows are
parent AOs and columns are orbitals.
"""

import numpy as np
import scipy.linalg

from pyscf import gto
from pyscf.lo.orth import lowdin


def _as_column_matrix(coeff, nao, name):
    coeff = np.asarray(coeff)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    if coeff.ndim != 2 or coeff.shape[0] != nao:
        raise ValueError(
            f'{name} must have shape (nao, norb); got {coeff.shape}, '
            f'nao={nao}')
    if coeff.shape[1] == 0:
        raise ValueError(f'{name} contains no orbitals')
    if not np.isfinite(coeff).all():
        raise ValueError(f'{name} contains non-finite values')
    return coeff


def orthonormalize_impurity_orbitals(imp_orb, ovlp, lindep=1e-10):
    """Symmetrically S-orthonormalize a fixed impurity span.

    Linear dependencies are rejected rather than silently deleting columns:
    changing their number would change the requested impurity definition.

    Returns
    -------
    imp_orth : ndarray
        S-orthonormal AO coefficients spanning the input columns.
    info : dict
        Rank/conditioning diagnostics suitable for JSON output.
    """
    ovlp = np.asarray(ovlp)
    if ovlp.ndim != 2 or ovlp.shape[0] != ovlp.shape[1]:
        raise ValueError('ovlp must be a square matrix')
    imp_orb = _as_column_matrix(imp_orb, ovlp.shape[0], 'imp_orb')
    gram = imp_orb.conj().T @ ovlp @ imp_orb
    gram = (gram + gram.conj().T) * 0.5
    eig, vec = scipy.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    cutoff = lindep * scale
    rank = int(np.count_nonzero(eig > cutoff))
    if rank != imp_orb.shape[1]:
        raise ValueError(
            'impurity projector is linearly dependent in the AO metric: '
            f'rank={rank}, columns={imp_orb.shape[1]}, '
            f'min_eigenvalue={eig.min():.3e}, cutoff={cutoff:.3e}')
    imp_orth = imp_orb @ ((vec * eig**-0.5) @ vec.conj().T)
    error = np.linalg.norm(
        imp_orth.conj().T @ ovlp @ imp_orth
        - np.eye(imp_orth.shape[1]))
    return imp_orth, {
        'nimp': int(imp_orth.shape[1]),
        'input_gram_eigenvalues': np.real_if_close(eig).real.tolist(),
        'input_gram_condition': float(eig.max() / eig.min()),
        'impurity_orth_error': float(error),
    }


def complete_impurity_basis(imp_orb, ovlp, lindep=1e-10):
    """Complete impurity orbitals with their S-orthogonal environment.

    The impurity columns are placed first.  The remaining columns span the
    unique S-orthogonal complement, so rotations chosen while completing the
    environment cannot change the DMET bath spectrum.

    Returns
    -------
    caolo : (nao, nao) ndarray
        AO <- orthonormal impurity/environment basis transformation.
    cloao : (nao, nao) ndarray
        Its inverse, equal to ``caolo.T.conj() @ ovlp``.
    info : dict
        Orthonormality diagnostics.
    """
    ovlp = np.asarray(ovlp)
    imp_orth, info = orthonormalize_impurity_orbitals(
        imp_orb, ovlp, lindep=lindep)
    nao, nimp = imp_orth.shape
    if nimp > nao:
        raise ValueError(f'nimp={nimp} exceeds nao={nao}')

    # X is AO <- symmetric-Lowdin-OAO.  X^H S C gives coefficients of C
    # in that Euclidean orthonormal coordinate system.
    x = lowdin(ovlp)
    imp_oao = x.conj().T @ ovlp @ imp_orth
    q_full, _ = scipy.linalg.qr(imp_oao, mode='full')
    env_orb = x @ q_full[:, nimp:]
    caolo = np.hstack((imp_orth, env_orb))
    cloao = caolo.conj().T @ ovlp

    ident = caolo.conj().T @ ovlp @ caolo
    full_error = np.linalg.norm(ident - np.eye(nao))
    cross_error = np.linalg.norm(imp_orth.conj().T @ ovlp @ env_orb)
    inverse_error = np.linalg.norm(cloao @ caolo - np.eye(nao))
    if full_error > 1e-7 or inverse_error > 1e-7:
        raise RuntimeError(
            'failed to complete the impurity projector to an orthonormal '
            f'basis: orth_error={full_error:.3e}, '
            f'inverse_error={inverse_error:.3e}')
    info.update({
        'basis_orth_error': float(full_error),
        'impurity_environment_overlap': float(cross_error),
        'basis_inverse_error': float(inverse_error),
    })
    return caolo, cloao, info


def subspace_singular_values(left, right, ovlp):
    """Principal-overlap singular values for two S-orthonormal subspaces."""
    left, _ = orthonormalize_impurity_orbitals(left, ovlp)
    right, _ = orthonormalize_impurity_orbitals(right, ovlp)
    if left.shape[1] != right.shape[1]:
        return np.zeros(0)
    return np.linalg.svd(left.conj().T @ ovlp @ right,
                         compute_uv=False)


def same_subspace(left, right, ovlp, atol=1e-7):
    """Whether two AO coefficient matrices span the same S-metric space."""
    if left is None or right is None:
        return left is None and right is None
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        return False
    try:
        singular_values = subspace_singular_values(left, right, ovlp)
    except (ValueError, scipy.linalg.LinAlgError):
        return False
    return (singular_values.size == left.shape[1]
            and np.allclose(singular_values, 1.0, atol=atol, rtol=0))


def project_reference_orbitals(mol, aolabels, reference_basis,
                               lindep=1e-10):
    """Project selected reference AOs into a molecule's parent AO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule carrying the large parent AO basis used for the Hamiltonian.
    aolabels : str or list[str]
        PySCF AO labels selecting the desired functions in the reference
        molecule, e.g. ``'Co 3d'`` or ``['Ce 4f', 'Ce 5d']``.
    reference_basis : str, dict, or PySCF basis object
        Independent projection basis.  For an element-specific custom basis
        in a molecule, pass a dictionary such as
        ``{'default': 'minao', 'Co': custom_co_basis}``.
    lindep : float
        Relative linear-dependence threshold.

    Returns
    -------
    imp_orb : (nao, nimp) ndarray
        S-orthonormal impurity orbitals in the parent AO basis.
    info : dict
        Reference labels, projection completeness, and conditioning metrics.
    """
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = reference_basis
    # The reference molecule is used only for overlap integrals.  Do not let a
    # parent ECP/pseudopotential alter how an all-electron projection basis is
    # interpreted.
    pmol.ecp = {}
    pmol.pseudo = None
    pmol.build(False, False)

    ref_idx = np.asarray(pmol.search_ao_label(aolabels), dtype=int)
    if ref_idx.size == 0:
        raise ValueError(
            f'no reference AOs match {aolabels!r}; inspect pmol.ao_labels()')

    ovlp = mol.intor_symmetric('int1e_ovlp')
    ref_ovlp_full = pmol.intor_symmetric('int1e_ovlp')
    ref_ovlp = ref_ovlp_full[np.ix_(ref_idx, ref_idx)]
    cross = gto.intor_cross('int1e_ovlp', pmol, mol)[ref_idx, :]

    # Least-squares projection of |ref> into span(parent AO):
    # S_parent C = <parent|ref>.
    raw = scipy.linalg.solve(ovlp, cross.conj().T, assume_a='pos')
    projected_gram = raw.conj().T @ ovlp @ raw
    fidelity = scipy.linalg.eigvalsh(projected_gram, ref_ovlp)
    imp_orb, orth_info = orthonormalize_impurity_orbitals(
        raw, ovlp, lindep=lindep)

    ref_labels = pmol.ao_labels()
    info = {
        'aolabels': ([aolabels] if isinstance(aolabels, str)
                     else list(aolabels)),
        'reference_ao_indices': ref_idx.tolist(),
        'reference_ao_labels': [ref_labels[i].strip() for i in ref_idx],
        'projection_eigenvalues': np.real_if_close(fidelity).real.tolist(),
        'projection_min': float(np.min(fidelity)),
        'projection_max': float(np.max(fidelity)),
    }
    info.update(orth_info)
    return imp_orb, info
