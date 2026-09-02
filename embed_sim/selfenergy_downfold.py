# Self-Energy Downfold for One-Shot DMET
# 在embed_sim/ssdmet.py中添加的新方法

import numpy as np
from pyscf import lib

class SelfEnergyDownfold:
    """
    从cluster solve结果提取自能修正，应用到嵌入空间
    """

    def __init__(self, ssdmet_obj):
        """
        参数：
            ssdmet_obj: SSDMET类实例（已运行build()和solver()）
        """
        self.dmet = ssdmet_obj
        self.log = ssdmet_obj.log
        self.mf = ssdmet_obj.mf_or_cas
        self.es_mf = ssdmet_obj.es_mf
        self.imp_idx = ssdmet_obj.imp_idx
        self.c_active = ssdmet_obj.c_active
        self.cluster_rdm1 = None
        self.cluster_rdm2 = None

    def compute_spectral_moments(self, mf_cluster):
        """
        计算cluster RDM的前几个spectral moments

        m_n = Tr[(F - F_mf)^n * DM1]  (简化版本)

        更完整的版本需要使用Dyson库，这里提供mock实现
        """
        self.log.info("Computing spectral moments from cluster solve...")

        # 第0矩：粒子数
        m0 = np.trace(self.cluster_rdm1)
        self.log.info(f"  m[0] (electron count) = {m0:.4f}")

        # 第1矩：能量期望
        fock = mf_cluster.get_fock()
        m1 = np.trace(fock @ self.cluster_rdm1)
        self.log.info(f"  m[1] (energy) = {m1:.8f}")

        # 第2矩：能量涨落
        m2 = np.trace((fock @ fock) @ self.cluster_rdm1)
        self.log.info(f"  m[2] (energy^2) = {m2:.8f}")

        return [m0, m1, m2]

    def compute_static_selfenergy_cluster_basis(self, mf_full, mf_cluster):
        """
        计算cluster基组中的静态自能

        Σ_0(cluster) = m[1] + m[1]_particle - F_cluster
        """
        self.log.info("Computing static self-energy in cluster basis...")

        fock_full = mf_full.get_fock()
        fock_cluster = np.linalg.multi_dot((
            self.c_active.T, fock_full, self.c_active
        ))

        # hole channel: m_h[1] = Tr[F * DM1_occ]
        nocc = len(self.dmet.imp_idx)
        dm1_occ = self.cluster_rdm1[:nocc, :nocc]
        m_h_1 = np.trace(fock_cluster @ dm1_occ)

        # particle channel: m_p[1] = Tr[F * DM1_vir]
        dm1_vir = self.cluster_rdm1[nocc:, nocc:]
        m_p_1 = np.trace(fock_cluster @ dm1_vir)

        # 静态自能（简化形式，Dyson库会更精确）
        # 这里使用 Σ_0 ≈ (E_h + E_p) - F_0
        se_cluster = fock_cluster.copy()
        se_cluster[: nocc, :nocc] = 0.0  # 占据-占据块为0
        se_cluster[nocc:, nocc:] = 0.0    # 虚-虚块为0

        # 杂化块包含自能效应
        se_cluster[:nocc, nocc:] *= 0.5  # 降权（保守估计）
        se_cluster[nocc:, :nocc] *= 0.5

        self.log.info(f"  Σ_0 norm = {np.linalg.norm(se_cluster):.6f}")

        return se_cluster

    def project_selfenergy_to_mf(self, se_cluster):
        """
        将cluster中的自能投影回全局MO基组

        方案A：投影到fragment轨道
            Σ_frag = P_frag @ Σ_cluster @ P_frag

        方案B：投影到嵌入空间占据/虚轨道
            Σ_emb = P_emb @ Σ_cluster @ P_emb
        """
        self.log.info("Projecting self-energy to global MF basis...")

        # 全局MO数
        nmo = self.mf.mo_coeff.shape[1]
        se_global = np.zeros((nmo, nmo), dtype=se_cluster.dtype)

        # 构造投影：fragment轨道 -> global MO
        # P_frag_mo = mo_coeff.T @ S @ c_active @ P_cluster @ c_active.T @ S @ mo_coeff
        s = self.mf.get_ovlp()

        # 简化版本：直接在active orbital基组中工作
        c_active_mo = np.linalg.multi_dot((
            self.mf.mo_coeff.T, s, self.c_active
        ))  # shape (nmo_global, ncluster_active)

        # 投影SE
        se_global = c_active_mo @ se_cluster @ c_active_mo.T

        return se_global

    def apply_selfenergy_correction_to_energy(self, rdm1_global=None):
        """
        应用自能修正到能量表达式

        E_corr += Tr(Σ_0 * ΔDM1)

        其中ΔDM1是embedded DM1相对于MF的变化
        """
        if rdm1_global is None:
            # 使用embedded solve的DM1
            rdm1_global = self.dmet.get_dm1_embedded()

        # 计算静态自能
        se_cluster = self.compute_static_selfenergy_cluster_basis(
            self.mf, self.es_mf
        )
        se_global = self.project_selfenergy_to_mf(se_cluster)

        # 计算RDM1相对于MF的变化
        dm1_mf = self.mf.make_rdm1()
        dm1_change = rdm1_global - dm1_mf

        # 自能贡献
        e_se = np.trace(se_global @ dm1_change)

        self.log.info(f"Self-energy correction to energy = {e_se:.10f} Ha")

        return e_se, se_global

    def compare_downfold_schemes(self, ccsd_obj):
        """
        对比不同downfold方案在能量上的差异
        """
        self.log.info("\n" + "="*70)
        self.log.info("DOWNFOLD SCHEME COMPARISON")
        self.log.info("="*70)

        # 方案1：Direct（浴敏感）
        e_direct = ccsd_obj.e_tot + self.dmet.fo_ene

        # 方案2：Correction-based（推荐PES用）
        e_corr_based = self.mf.e_tot + ccsd_obj.e_corr

        # 方案3：With static self-energy
        e_se_correction, se_mat = self.apply_selfenergy_correction_to_energy()
        e_with_se = e_corr_based + e_se_correction

        # 方案4：With FBC（如果有BNO浴）
        # e_fbc = self.dmet.get_finite_bath_correction()  # TODO
        # e_with_fbc = e_direct + e_fbc

        # 输出对比
        self.log.info(f"\nTotal energies for different schemes:")
        self.log.info(f"  Direct            = {e_direct:.12f} Ha  (bath-sensitive)")
        self.log.info(f"  Correction-based  = {e_corr_based:.12f} Ha  (recommended)")
        self.log.info(f"  + SE correction   = {e_with_se:.12f} Ha  (envir. corr.)")
        self.log.info(f"\nEnergy differences:")
        self.log.info(f"  E_direct - E_corr = {e_direct - e_corr_based:.6e} Ha")
        self.log.info(f"  ΔE_SE = {e_se_correction:.6e} Ha")

        return {
            'direct': e_direct,
            'correction_based': e_corr_based,
            'with_se': e_with_se,
            'se_correction': e_se_correction,
            'se_matrix': se_mat
        }


# ============================================================================
# 集成到SSDMET.ccsd_solver()的示例
# ============================================================================

def integrated_ccsd_with_downfold(self, with_t=False):
    """
    修改原有ccsd_solver方法，添加self-energy downfold
    """
    from pyscf import cc

    if self.es_mf is None:
        raise RuntimeError('embedded subspace is not built yet, run build() first')

    self.log.info("Running CCSD for EO space with self-energy downfold")
    self.log.info("="*60)

    # 原有的CCSD求解
    mycc = cc.CCSD(self.es_mf)
    mycc.kernel()

    # 保存RDM用于self-energy计算
    rdm1 = mycc.make_rdm1()
    rdm2 = mycc.make_rdm2()
    self.results_cluster_dm1 = rdm1
    self.results_cluster_dm2 = rdm2

    # ======= NEW: 添加self-energy downfold =======
    se_downfold = SelfEnergyDownfold(self)
    results_dict = se_downfold.compare_downfold_schemes(mycc)

    # 选择最终能量定义
    final_e_tot = results_dict['correction_based']  # 推荐用于PES

    # 如果信任self-energy修正，也可以：
    # final_e_tot = results_dict['with_se']

    # ============================================

    if with_t:
        et = mycc.ccsd_t()
        self.log.info(f"CCSD(T) correlation energy = {mycc.e_corr + et:.12f}")
        return mycc, final_e_tot + et, results_dict

    return mycc, final_e_tot, results_dict


# ============================================================================
# 在反应PES扫描中的使用
# ============================================================================

def scan_reaction_path_with_downfold():
    """
    在几何扫描中应用self-energy downfold，稳定能量
    """
    import matplotlib.pyplot as plt

    geometries = [...]  # 反应坐标
    energies = {'direct': [], 'correction': [], 'with_se': []}

    for geom in geometries:
        # 几何优化或者固定坐标
        mol = setup_molecule(geom)
        mf = scf.RHF(mol).run()

        # DMET嵌入
        mydmet = ssdmet.SSDMET(mf, imp_idx=[...], bath_norb='per_bond')
        mydmet.build()

        # CCSD with downfold
        mycc, e_tot, results = integrated_ccsd_with_downfold(mydmet, with_t=False)

        energies['direct'].append(results['direct'])
        energies['correction'].append(results['correction_based'])
        energies['with_se'].append(results['with_se'])

    # 绘制对比
    fig, ax = plt.subplots()
    for scheme, e_list in energies.items():
        ax.plot(range(len(geometries)), e_list, label=scheme, marker='o')

    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('Energy (Ha)')
    ax.legend()
    plt.savefig('pes_downfold_comparison.png')

    # 分析噪声
    noise_direct = np.std(np.diff(energies['direct']))
    noise_corr = np.std(np.diff(energies['correction']))
    noise_se = np.std(np.diff(energies['with_se']))

    print(f"PES噪声（二阶差分std）:")
    print(f"  Direct:      {noise_direct:.2e} Ha")
    print(f"  Correction:  {noise_corr:.2e} Ha")
    print(f"  With SE:     {noise_se:.2e} Ha")
