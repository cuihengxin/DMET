"""Analytic nuclear gradients for embed_sim DMET (one-shot).

Usage::

    from pyscf import gto, scf
    from embed_sim import ssdmet

    mol = gto.M(atom='...', basis='sto-3g')
    mf = scf.RHF(mol).run()

    mydmet = ssdmet.SSDMET(mf, title='x', imp_idx=[0])
    mydmet.build(save_chk=False)

    de = mydmet.nuc_grad_method().kernel()
"""

from embed_sim.grad.ssdmet import Gradients, SSDMETGradients

__all__ = ['Gradients', 'SSDMETGradients']
