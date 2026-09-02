"""反解 ROHF P 响应里"占-占 oo" 的正确权重 w_ij。

数值 P 响应 - (zdens + zPFP + 0.5 vhf) = 残差。
假设残差 = Σ_{i,j∈occ} w_ij moFbar_ij (C^T dS/dR C)_ij  （占-占，w 对称）
多几何 + 多分量 lstsq 反解 w_ij，看块状结构（docc-docc / docc-socc / socc-socc）。

若 w 是干净块（如 2/1/1 或 3/2/2 等）-> 找到正确 oo；
若 w 无模式或残差含占-虚 -> 缺口不是简单占-占。

运行：cd 8dmet4reac/DMET && python examples/test_example/invert_rohf_oo.py
"""

import os
import sys

import numpy as np
from pyscf import gto, scf

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)

from embed_sim import ssdmet
from embed_sim.grad.linalg import matfun_grad


def tight(m):
    mf = scf.rohf.ROHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    return mf


def sym(a):
    return a + a.T


def presp_blocks(mol):
    """返回 (zdens+zPFP+0.5vhf 解析, 数值 P 响应, S-导数基 b_ij)。"""
    mf = tight(mol)
    d = ssdmet.SSDMET(mf, title='bl', imp_idx=[0], bath_norb=1, verbose=0)
    d.build(save_chk=False)
    g = d.nuc_grad_method()
    g.check_support(); g.decompose(); g.make_densities()
    g.orb_grad(); g.lo_grad(); g.solve_z()

    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    nmo = C.shape[1]
    occa, occb, var_a, var_b, uniq_ab = g._rohf_rotation_masks()
    ka = np.where(var_a, g.kappa, 0.0)
    kb = np.where(var_b, g.kappa, 0.0)
    dm = mf.make_rdm1(C, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa = getattr(fock, 'focka', fock)
    fb = getattr(fock, 'fockb', fock)
    occ = mo_occ > 0
    Zma, Zmb = C @ ka @ C.T, C @ kb @ C.T
    Pa, Pb = C[:, occa] @ C[:, occa].T, C[:, occb] @ C[:, occb].T

    sbar_lowdin = (matfun_grad(g.w_s, g.u_s, g.B_X,
                               lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5)
                   + matfun_grad(g.w_s, g.u_s, g.B_Y,
                                 lambda x: x ** 0.5, lambda x: 0.5 * x ** -0.5))
    zvec0, dPa0, dPb0 = g.zvec_ao.copy(), g.dPa.copy(), g.dPb.copy()
    vj, vk = mf.get_jk(mol, np.asarray((g.dPa, g.dPb)))
    vhf = sym(Pa @ (vj[0] + vj[1] - vk[0]) @ Pa
              + Pb @ (vj[0] + vj[1] - vk[1]) @ Pb)

    def grad(use_z, use_zeta, use_vhf):
        g.zvec_ao = use_z * zvec0
        g.dPa = use_z * dPa0
        g.dPb = use_z * dPb0
        sb = sbar_lowdin.copy()
        if use_zeta:
            sb = sb - sym(Zma @ fa @ Pa + Zmb @ fb @ Pb)
        if use_vhf:
            sb = sb - vhf
        g.sbar = sb
        return g.contract() + g.mfgrad.grad_nuc()

    base = grad(0, 0, 0)
    ana = grad(1, 1, 1) - base            # zdens + zPFP + vhf (系数 1/1/1)
    # 实际用 zPFP=1, vhf=0.5：重算
    ana2 = (grad(1, 1, 1) - grad(1, 1, 0)) * 0.5 + (grad(1, 1, 0) - base)
    # 简化：直接组合块
    c_zdens = grad(1, 0, 0) - base
    c_zPFP = grad(0, 1, 0) - base
    c_vhf = grad(0, 0, 1) - base
    ana = c_zdens + c_zPFP + 0.5 * c_vhf

    # numerical P-response
    coords = mol.atom_coords()
    G_P = g.G_P.copy()
    num = np.zeros((mol.natm, 3))
    h = 1e-3
    for ia in range(mol.natm):
        for x in range(3):
            q1 = q2 = 0.0
            for s in (1, -1):
                c = coords.copy()
                c[ia, x] += s * h
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                mf2 = tight(m)
                q = np.einsum('ij,ij->', G_P,
                              mf2.make_rdm1()[0] + mf2.make_rdm1()[1])
                if s > 0:
                    q1 = q
                else:
                    q2 = q
            num[ia, x] = (q1 - q2) / (2 * h)

    # S-derivative basis b_ij: (C^T dS/dR C)_ij for i,j in occ
    s1 = g.mfgrad.get_ovlp(mol)           # (3, natm, nao, nao) 或 (3,nao,nao)?
    if s1.ndim == 3:
        s1 = s1[:, None]                  # (3, 1, nao, nao)
    natm = mol.natm
    occ_idx = np.where(occ)[0]
    b = np.zeros((natm * 3, len(occ_idx), len(occ_idx)))
    moFbar_oo = g.moFbar[np.ix_(occ, occ)]
    for ia in range(natm):
        for x in range(3):
            Smo = C[:, occ_idx].T @ s1[x, ia] @ C[:, occ_idx]
            b[ia * 3 + x] = Smo * moFbar_oo      # 每分量基
    return ana, num, b, occ_idx, mo_occ


def main():
    def ch3(dx=0.0, dy=0.0, dz=0.0):
        return gto.M(atom=f'C 0 0 0; H {0.63+dx} 0.61 0.05; '
                          f'H -0.60 {0.65+dy} -0.03; H 0.02 -0.89 {0.07+dz}',
                     basis='sto-3g', spin=1, verbose=0)

    mols = [ch3(), ch3(0.08, -0.05), ch3(-0.06, 0.09, 0.04),
            ch3(0.12, 0.11, -0.05)]
    A_list, r_list, infos = [], [], []
    for m in mols:
        ana, num, b, occ_idx, mo_occ = presp_blocks(m)
        resid = (num - ana).ravel()
        # 未知 w（对称，占-占），b 是 (12, nocca, nocca) 每分量
        B = b.reshape(-1, b.shape[1] * b.shape[2])   # (12, nocca^2)
        # 对称化：w 对称，用上三角
        n_occ = b.shape[1]
        tri = np.array([(i, j) for i in range(n_occ) for j in range(i, n_occ)])
        Btri = np.zeros((B.shape[0], len(tri)))
        for k, (i, j) in enumerate(tri):
            Btri[:, k] = B[:, i * n_occ + j] + (B[:, j * n_occ + i]
                                                if i != j else 0)
        A_list.append(Btri)
        r_list.append(resid)
        infos.append((occ_idx, mo_occ))

    A = np.vstack(A_list)
    r = np.concatenate(r_list)
    w, *_ = np.linalg.lstsq(A, r, rcond=None)
    print('=== 反解 w_ij (占-占 oo 权重) ===')
    print('  fit 残差 max = %.3e' % np.abs(A @ w - r).max())
    print('  残差总量级 max = %.3e' % np.abs(r).max())
    occ_idx, mo_occ = infos[0]
    n_occ = len(occ_idx)
    print('  占据数:', mo_occ[occ_idx])
    wmat = np.zeros((n_occ, n_occ))
    k = 0
    for i in range(n_occ):
        for j in range(i, n_occ):
            wmat[i, j] = wmat[j, i] = w[k]
            k += 1
    print('  w 矩阵 (按占据轨道):')
    print(np.round(wmat, 3))
    print('\n若 w 分块干净（docc-docc / docc-socc / socc-socc 各自常数）且残差归零，')
    print('即得正确 oo；若残差大 -> 缺口不是简单占-占（含占-虚或别的项）。')


if __name__ == '__main__':
    main()
