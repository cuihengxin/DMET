# 复现说明:乙烷 C–C 拉伸,每键一个 bath(CH3 杂质 + 1 个 bath)

## 1. 复现步骤

```bash
# 依赖:pyscf、numpy、scipy、sympy(Python 3.9+;DMET_main 需要加入 PYTHONPATH)
cd /Users/cuihengxin/Desktop/2026phd/DMET_main
PYTHONPATH=$PWD python examples/test_example/one_bath_per_bond_ethane/ethane_one_bath_pes.py
# 或 SLURM:
cd examples/test_example/one_bath_per_bond_ethane && sbatch run.sh
```

输入参数都在脚本顶部,可直接修改:

- `BASIS = '6-31g'`,扫描 `R_GRID`(1.30–3.00 Å,11 点);
- 杂质 = CH3 基团(`IMP_LABELS = ['0 C.*','2 H.*','3 H.*','4 H.*']`,即 C0 及其 3 个 H);
- `BATH_NORB = 1`:每键一个 bath 轨道。

## 2. 输出文件

- `ethane_one_bath_pes_results.txt`:完整能量与能量差表(每个 R 一行):
  `R  E_full_HF  E_full_MP2  E_emb_HF  E_emb_MP2  e_corr_emb  E_Direct  E_Corr
  lam_bath  dE_full_MP2  dE_Direct  dE_Corr  err_Direct_mHa  err_Corr_mHa`
  - `E_Direct = E(嵌入MP2) + E(冻结占据)`(对 bath 大小敏感);
  - `E_Corr = E(全分子HF) + e_corr(嵌入MP2)`(修正型公式,推荐用于 PES);
  - `dE(X) = E(X;R) − E(X;R_eq)`,`err = dE(X) − dE(全电子MP2)`,单位 mHa。
- 终端同时打印 PES 误差统计(Direct / Corr 的 max 与 RMS)和 bath 选择报告。

## 3. 这一个 bath 轨道是怎么选的(评判标准)

1. 把全分子 RHF 密度变换到 Löwdin 正交基,取**环境块**(去掉杂质 CH3 后的
   15 个轨道)对角化,得到环境自然轨道及占据数 λ;
2. 评判标准是 **λ 越接近 1 越"纠缠"**,排序键取 `min(λ, 2−λ)` 从大到小
   (等价于 QC-DMET 的 `max(−λ, λ−2)` 升序);
3. 取排序第一的轨道作为唯一的 bath 轨道。

平衡点(R = 1.54 Å)的实测环境占据谱:

```
λ     : 0, 0, 0, 0, 0, 0, 1e-4, 0.0137, 0.0137, 0.0318, 1.0000,
        1.9682, 1.9863, 1.9863, 1.9999
min(λ,2−λ): 0, 0, 0, 0, 0, 0, 1e-4, 0.0137, 0.0137, 0.0318, 1.0000,
            0.0318, 0.0137, 0.0137, 1e-4
```

选中的是 **λ = 1.0000** 的环境自然轨道(C–C σ 键在环境 CH3 侧的伙伴):
占据数正好为 1 说明该键在两侧各占一半,纠缠最强;其余轨道要么近 0(环境
内空轨道)、要么近 2(环境内成对占据),纠缠可忽略。

## 4. 预期结果(摘要)

CH3/1 + Corr 公式:MP2 PES 误差 max 3.2 mHa、RMS 1.2 mHa(相对全电子 MP2);
Direct 公式:max 181 mHa、RMS 114 mHa。绝对能量差 `E_Corr − E_full_MP2`
约 99–102 mHa(嵌入空间只有 16 个轨道,环境相关未算),但沿键长几乎不变,
在能量差中抵消。
