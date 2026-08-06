import numpy as np
import copy
from embed_sim.ssdmet2 import SSDMET

class BathConsistencySweeper:
    def __init__(self, mf_list, imp_idx, threshold=1e-4):
        """
        处理势能面扫描的一致性工具。
        
        Args:
            mf_list (list): 已收敛的 Mean-Field 对象列表 [mf_0, mf_1, ..., mf_N]。
            imp_idx (list): 杂质原子索引。
            threshold (float): Schmidt 分解阈值。
        """
        self.mf_list = mf_list
        self.imp_idx = imp_idx
        self.threshold = threshold
        self.dmet_objects = [None] * len(mf_list)
        
        # 存储每一步产生的 Bath Coeff (AO基)，用于传递给下一步
        self.bath_history = [None] * len(mf_list)

    def run_forward_sweep(self):
        """
        前向扫描：Struct 0 -> Struct N
        """
        print("\n" + "="*80)
        print("STARTING FORWARD SWEEP (0 -> N)")
        print("="*80)
        
        ref_bath = None
        ref_mol = None
        
        for i, mf in enumerate(self.mf_list):
            print(f"\n---> Processing Point {i} / {len(self.mf_list)-1}")
            
            dmet = SSDMET(mf, imp_idx=self.imp_idx, threshold=self.threshold, verbose=4)
            
            if i == 0 or ref_bath is None:
                # 第一点：只能使用标准阈值构建
                print("  [Init] Building initial bath using standard threshold.")
                dmet.build() 
            else:
                # 后续点：取 [参考Top-M] U [当前阈值] 的并集
                print(f"  [Adaptive] Building consistent bath with reference from Point {i-1}.")
                # ref_bath 是 AO 基下的系数
                dmet.build_union_with_reference(ref_bath, ref_mol=ref_mol)
            
            # --- Convergence Logic: 保留更大的空间以防震荡，或者直接更新 ---
            # 为了更好的收敛性，如果历史记录的 Bath 空间显著更大，可以选择保留历史的
            # 但既然我们使用了 Union 策略，通常新的结果已经包含了历史信息
            if self.dmet_objects[i] is not None:
                old_nbath = self.dmet_objects[i].nes - len(self.imp_idx)
                new_nbath = dmet.nes - len(dmet.imp_idx)
                print(f"  [Info] History Bath: {old_nbath} | New Union Bath: {new_nbath}")
            
            # 存储结果
            self.dmet_objects[i] = dmet
            
            # 提取 Bath 轨道系数 (AO基)
            # es_orb 结构: [Imp | Bath]
            nimp = len(dmet.imp_idx)
            nbath = dmet.nes - nimp
            current_bath_ao = dmet.es_orb[:, nimp : nimp+nbath]
            
            self.bath_history[i] = current_bath_ao
            
            # 更新参考: 这里的参考包含了 Union 后的结果，所以信息会传递下去
            ref_bath = current_bath_ao
            ref_mol = dmet.mol

    def run_backward_sweep(self):
        """
        后向扫描：Struct N -> Struct 0
        """
        print("\n" + "="*80)
        print("STARTING BACKWARD SWEEP (N -> 0)")
        print("="*80)
        
        ref_bath = None
        ref_mol = None
        
        # 倒序遍历: N, N-1, ..., 0
        for i in range(len(self.mf_list) - 1, -1, -1):
            mf = self.mf_list[i]
            print(f"\n<--- Processing Point {i} / {len(self.mf_list)-1}")
            
            dmet = SSDMET(mf, imp_idx=self.imp_idx, threshold=self.threshold, verbose=4)
            
            # 构建 Bath
            if ref_bath is None:
                # 初始点（序列最后一点）
                # 为了连贯性，如果 forward sweep 已经跑过，可以用 forward 的结果做参考来初始化 backward
                # 这里简单起见，如果 forward 结果存在，就用它做参考
                if self.bath_history[i] is not None:
                     print("  [Init] Using Forward sweep result as reference for initial backward point.")
                     # 注意：如果是作为参考，我们要把 forward 的结果传给 build_union
                     # 但由于这是 backward 的起点，我们也可以选择直接用 forward 的结果作为当前点
                     # 或者基于它 rebuild。这里选择基于它 rebuild 以保持逻辑统一。
                     ref_from_fwd = self.bath_history[i]
                     dmet.build_union_with_reference(ref_from_fwd, ref_mol=dmet.mol)
                else:
                     print("  [Init] Building initial backward bath using standard threshold.")
                     dmet.build()
            else:
                # 使用后一点 (i+1) 作为参考
                print(f"  [Adaptive] Building consistent bath with reference from Point {i+1}.")
                dmet.build_union_with_reference(ref_bath, ref_mol=ref_mol)
            
            # --- Convergence Logic ---
            self.dmet_objects[i] = dmet
            
            # 更新参考
            nimp = len(dmet.imp_idx)
            nbath = dmet.nes - nimp
            current_bath_ao = dmet.es_orb[:, nimp : nimp+nbath]
            self.bath_history[i] = current_bath_ao
            
            ref_bath = current_bath_ao
            ref_mol = dmet.mol

    def run_cycle(self):
        """执行完整的一轮 Forward-Backward 循环"""
        self.run_forward_sweep()
        self.run_backward_sweep()
        return self.dmet_objects

    def run_converge(self, max_cycles=3):
        """
        运行 Forward-Backward 循环直到 Bath 空间大小收敛。
        
        Args:
            max_cycles (int): 最大循环次数
        """
        print("\n" + "#"*80)
        print(f"STARTING CONVERGENCE LOOP (Max {max_cycles} cycles)")
        print("#"*80)

        prev_sizes = []
        
        for cycle in range(max_cycles):
            print(f"\n>>> CYCLE {cycle + 1} START")
            self.run_forward_sweep()
            self.run_backward_sweep()
            
            # 收集当前所有点的 Bath 大小
            current_sizes = [dmet.nes - len(dmet.imp_idx) for dmet in self.dmet_objects]
            print(f">>> CYCLE {cycle + 1} END. Bath sizes: {current_sizes}")
            
            if prev_sizes == current_sizes:
                print(f"\n*** CONVERGED at Cycle {cycle + 1} ***")
                break
            
            prev_sizes = current_sizes
        else:
            print("\n*** WARNING: Max cycles reached without full convergence ***")
            
        return self.dmet_objects
    def run_2cycle(self):
        """执行完整的2轮 Forward-Backward 循环"""
        self.run_forward_sweep()
        self.run_backward_sweep()
        self.run_forward_sweep()
        self.run_backward_sweep()
        return self.dmet_objects

