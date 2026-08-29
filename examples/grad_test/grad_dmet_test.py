"""Validation of the one-shot SSDMET analytic nuclear gradient (stage 1).

Three independent checks:

1. HF-in-HF exactness (full bath): the DMET gradient must reproduce the plain
   RHF analytic gradient, since a full bath makes the embedding exact.
2. Truncated bath (H2O, STO-3G, single bath): analytic gradient vs central
   finite difference of the DMET energy.
3. Truncated bath (water dimer, STO-3G, one H2O fragment + 2 bath orbitals):
   a genuinely truncated case whose energy differs from RHF.

Run from the repository root:

    python examples/test_example/grad_dmet_test.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from pyscf import gto, scf

from embed_sim import ssdmet
from embed_sim.grad.numgrad import numerical_grad, ssdmet_energy

DIMER_ATOM = '''
   O  8.70835e-01  6.24543e+00  5.08445e+00
   H  8.51421e-01  6.33649e+00  6.05969e+00
   H  1.66206e+00  5.60635e+00  5.02479e+00
   O  2.38299e+00  8.36926e+00  4.10083e+00
   H  1.76679e+00  7.63665e+00  4.17552e+00
   H  2.40734e+00  8.80363e+00  4.99023e+00 '''


def test_full_bath_hfinhf():
    """Full bath -> DMET == RHF, gradient must match mf.nuc_grad_method()."""
    mol = gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    d = ssdmet.SSDMET(mf, title='full_bath', imp_idx=[0])
    d.build(save_chk=False)
    e_dmet = d.es_mf.e_tot + d.fo_ene
    assert abs(e_dmet - mf.e_tot) < 1e-10

    gd = d.nuc_grad_method().kernel()
    gr = mf.nuc_grad_method().kernel()
    err = np.abs(gd - gr).max()
    print(f'[1] full-bath HF-in-HF: E_dev={abs(e_dmet-mf.e_tot):.2e}  '
          f'grad_dev={err:.2e}')
    assert err < 1e-6


def test_truncated_sto3g():
    """Truncated bath vs central finite differences (single H2O)."""
    mol = gto.M(atom='O 0 0 0; H 0 0.96 0.26; H 0 -0.24 -0.96',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    d = ssdmet.SSDMET(mf, title='trunc_sto3g', imp_idx=[0], bath_norb=1)
    d.build(save_chk=False)
    gd = d.nuc_grad_method().kernel()

    num = numerical_grad(
        lambda m: ssdmet_energy(m, imp_idx=[0], bath_norb=1, verbose=0),
        mol, step=1e-3, verbose=False)
    err = np.abs(gd - num).max()
    print(f'[2] truncated STO-3G: grad vs FD = {err:.2e}')
    assert err < 1e-6


def test_truncated_dimer():
    """Truncated water dimer, genuinely non-trivial embedding."""
    mol = gto.M(atom=DIMER_ATOM, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()

    d = ssdmet.SSDMET(mf, title='trunc_dimer', imp_idx=[0, 1, 2], bath_norb=2)
    d.build(save_chk=False)
    e_dmet = d.es_mf.e_tot + d.fo_ene
    # make sure the test is actually exercising a truncated (non-exact) bath
    assert abs(e_dmet - mf.e_tot) > 1e-5

    gd = d.nuc_grad_method().kernel()
    num = numerical_grad(
        lambda m: ssdmet_energy(m, imp_idx=[0, 1, 2], bath_norb=2,
                                verbose=0, conv_tol=1e-14),
        mol, step=1e-3, verbose=False)
    err = np.abs(gd - num).max()
    assert abs(gd.sum(axis=0)).max() < 1e-12, 'gradient breaks translation'
    print(f'[3] truncated dimer: E_dev={abs(e_dmet-mf.e_tot):.3e}  '
          f'grad vs FD = {err:.2e}  sum(grad)={abs(gd.sum(axis=0)).max():.1e}')
    assert err < 1e-6


if __name__ == '__main__':
    test_full_bath_hfinhf()
    test_truncated_sto3g()
    test_truncated_dimer()
    print('all one-shot SSDMET gradient tests passed')
