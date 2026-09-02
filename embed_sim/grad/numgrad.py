"""Central-difference nuclear gradients, for validating the analytic ones.

``numerical_grad`` displaces each Cartesian coordinate, rebuilds the molecule
and calls a user-supplied energy function.  ``ssdmet_energy`` is the matching
HF-in-HF energy closure for :class:`embed_sim.ssdmet.SSDMET`.
"""

import numpy as np

from pyscf import gto, scf

BOHR = 0.52917721092


def numerical_grad(efunc, mol, step=1e-3, atmlst=None, verbose=True):
    """Central-difference gradient of ``efunc(mol)`` in Eh/Bohr.

    Args:
        efunc: callable taking a ``gto.Mole`` and returning the total energy.
        mol: reference molecule.
        step: displacement in Bohr.
        atmlst: atoms to differentiate (default: all).
    """
    if atmlst is None:
        atmlst = range(mol.natm)
    coords = mol.atom_coords()          # Bohr
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for x in range(3):
            ep, em = [], []
            for sign in (1, -1):
                c = coords.copy()
                c[ia, x] += sign * step
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                (ep if sign > 0 else em).append(efunc(m))
            de[k, x] = (ep[0] - em[0]) / (2 * step)
        if verbose:
            print(f'  numerical grad atom {ia}: {de[k]}')
    return de


def ssdmet_energy(mol, imp_idx, conv_tol=1e-12, verbose=0, **dmet_kwargs):
    """RHF + one-shot SSDMET (HF-in-HF) total energy of ``mol``."""
    from embed_sim import ssdmet

    mol = mol.copy()
    mol.verbose = verbose
    mol.build(False, False)

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.verbose = verbose
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('reference RHF did not converge')

    dmet_kwargs.setdefault('verbose', verbose)
    mydmet = ssdmet.SSDMET(mf, title='numgrad', imp_idx=imp_idx, **dmet_kwargs)
    mydmet.build(save_chk=False)
    return mydmet.es_mf.e_tot + mydmet.fo_ene
