# PNO 型 DMET bath 构造：理论、实现与工作流程

> 面向 reviewer/自查的设计文档。写于 2026-08-22，代码状态：`pno_bath.py` 与
> `ssdmet_pno.py` 已实现并通过结构完整性测试，收敛性/能量差对比实验正在进行。
> 本文把「为什么这么做」和「公式从哪来」写清楚，也如实记录了目前的**负面结果**。

---

## 0. 一句话概括

把 DLPNO（domain-based local pair natural orbital）里「**按占据对分辨** MP2 密度、逐对取自然轨道」的思路，移植到单发 DMET 的 bath 构造上，替代 `BNO_bath` 里「**对所有对求和**再对角化」的全局做法，使 bath 选择判据从「占据数 $\eta$」变成「**关联能（Hartree）**」。

参考文献：Bensberg & Neugebauer, *J. Chem. Phys.* **157**, 064102 (2022)
（"Orbital pair selection for relative energies in DLPNO-CC"）。

---

## 1. 背景与问题

### 1.1 DMET 的 frozen 空间与 bath

单发 DMET 把全体系轨道分成三块：

$$
C_\text{full} = [\underbrace{C_\text{imp}+C_\text{bath}}_{C_\text{es}:\ \text{嵌入空间}} \mid \underbrace{C_\text{fo}}_{冻结占据} \mid \underbrace{C_\text{fv}}_{冻结虚}]
$$

- **嵌入空间**（impurity + bath）里做高等级求解器（CCSD(T)、CASSCF+NEVPT2 等）。
- **冻结占据 / 冻结虚**被冻结，只贡献平均场能量 $E_\text{fo}$。

DMET 精确性关系保证：**当 bath 完备时**，
$$
E(\text{嵌入 mean field}) + E_\text{fo} = E(\text{全体系 mean field}) .
$$
这条关系是结构完整性的硬判据（下面用它做 stage 1 测试）。

### 1.2 问题的根源

bath 来自低等级 1-RDM，只把纠缠最强的环境占据轨道放进嵌入空间；环境虚轨道全被冻结。
于是**动态关联被截断**：全体系 MP2/CCSD 里那些「从嵌入占据 → 冻结虚」「从冻结占据 → 嵌入虚」
的激发被丢掉。这正是 IPDMET 论文里观察到的「NEVPT2 后误差显著增大」的来源，也是
你们调研报告里「CCSD(T) 需要 60%+ EO 才收敛」的原因。

### 1.3 现有 `BNO_bath` 的做法及其缺陷

`BNO_bath.get_RMP2_bath` 做的是：

1. 在冻结虚空间里建半变换振幅 $t_{ij}^{\tilde a\tilde b}$（对嵌入占据 $i,j$，冻结虚 $\tilde a,\tilde b$）；
2. 由 $t$ 造**全局** MP2 密度 $D^{\tilde a\tilde b}=\sum_{ij}D^{ij}$（对全部占据对求和）；
3. 对角化 $D$ 得自然轨道，取占据数 $n>\eta$ 者作为新 bath。

**缺陷**（本工作的动机）：
- $D$ 是对全部对**求和后再对角化**。一个对某单个对很关键、但全局平均下来不显眼的环境轨道会被平均掉。
- 阈值 $\eta$ 是**占据数**，无量纲、无能量含义，无法对应「我丢了多少关联能」。
- 无法区分「这个轨道对哪个对的关联能重要」。

---

## 2. 理论基础：对分辨 PNO

### 2.1 半正则 MP2 振幅（在 frozen 空间内）

固定某一侧（以虚侧为例：占据对 $i,j\in$ 嵌入占据，目标 $a,b\in$ 冻结虚）。先把冻结虚空间
在其内部半正则化（对角化该空间里的 Fock 块），得到一致的能量分母，于是

$$
t_{ij}^{ab} = \frac{(ia\,|\,jb)}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}
\qquad(\text{chemists' notation})
$$

记 $K_{ab}=(ia\,|\,jb)$，则 $t_{ij}^{ab}=K_{ab}/D_{ab}$，$D_{ab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b$。

> 注意：与 `BNO_bath` 不同，这里振幅**直接建在 frozen 空间内**，因此下面的对能量
> 天然就是「当前 cluster 描述不了的那部分」——这正是「该不该把这个环境轨道提为 bath」
> 应该看的量。

### 2.2 对能量（pair energy）

$$
\varepsilon_{ij}=\sum_{ab}t_{ij}^{ab}\big(2K_{ab}-K_{ba}\big)
$$

半正则 MP2 的总关联能按对求和：

$$
E_\text{corr}=\sum_{i\le j}w_{ij}\,\varepsilon_{ij},\qquad
w_{ij}=\begin{cases}1,&i=j\\2,&i<j\end{cases}
$$

这是 DLPNO 里 `T_CutPairs` 的筛对货币：$|w_{ij}\varepsilon_{ij}|$ 小的对不贡献 bath。

### 2.3 DLPNO 对密度与 PNO

Riplinger & Neese, *JCP* **138**, 034106 (2013), Eq. 8：

$$
D^{ij} = (1+\delta_{ij})^{-1}\big(\tilde T_{ij}^\dagger\tilde T_{ij}+\tilde T_{ij}\tilde T_{ij}^\dagger\big),
\qquad
\tilde T_{ij}=2T_{ij}-T_{ij}^{T}
$$

对角化 $D^{ij}=v\,n\,v^T$，得该对的自然轨道 $v$ 与占据数 $n$。占据数 $n$ 即 DLPNO 的
`T_CutPNO` 判据。

### 2.4 关键新增：对能量在轨道上的严格分解

这是本实现区别于「简单把对能量乘在占据数上」的地方。给定该对自己的 PNO 基
$\tilde T=v^T T v,\ \tilde K=v^T K v$，令

$$
M_{\tilde a\tilde b}=\tilde T_{\tilde a\tilde b}\big(2\tilde K_{\tilde a\tilde b}-\tilde K_{\tilde b\tilde a}\big),
\qquad
c_{\tilde n}=\frac12\Big(\textstyle\sum_{\tilde b}M_{\tilde n\tilde b}+\sum_{\tilde a}M_{\tilde a\tilde n}\Big)
$$

则 $c_{\tilde n}$ 是**该 PNO 携带的对能量份额**（单位 Hartree），且

$$
\sum_{\tilde n} c_{\tilde n} = \frac12\Big(\sum_{\tilde a\tilde b}M_{\tilde a\tilde b}+\sum_{\tilde a\tilde b}M_{\tilde a\tilde b}\Big)
= \sum_{\tilde a\tilde b}M_{\tilde a\tilde b}
= \varepsilon_{ij}
$$

**严格成立**（双边正交变换下缩并不变）。$c_{\tilde n}$ 跨对可比，可作为轨道级能量判据
`t_orb_energy`，也让下面的能量目标模式成为可能。

### 2.5 为什么**不需要** domain

DLPNO 里的 PAO spatial domain 是**线性标度的成本装置**（限制每对的积分变换范围），
对**选择精度零贡献**。本场景的 frozen 空间（几十个轨道）本来就整块建得出来，逐对振幅
`t_ij^ab` 全都可算，域截断毫无必要。**对分辨本身才是核心**。之前方案里把
coupling-strength domain 当重点，是错误判断，已放弃。

### 2.6 固定尺寸模式 = 加权自然轨道（受控实验的关键）

由 $D^{ij}=v\,n\,v^T$ 有

$$
\sum_{ij}D^{ij}=\underbrace{M\,\mathrm{diag}(n)\,M^T}_{M=\text{所有 PNO 向量拼成的矩阵}}
=\big(M\,\mathrm{diag}(\sqrt n)\big)\big(M\,\mathrm{diag}(\sqrt n)\big)^T
$$

因此「把候选列按 $\sqrt{\text{score}}$ 加权后取前 $k$ 个左奇异向量」**精确等价于**对角化
加权后的对密度之和。于是：

| `score` | 权重 | 等价对象 |
|---|---|---|
| `occ` | $\sqrt{n}$ | **全局 BNO 式自然轨道**（精确复现旧方法） |
| `energy` | $\sqrt{\,|c_{\tilde n}|\,}$ | 能量加权自然轨道（新） |
| `occ_x_pair` | $\sqrt{n\cdot|\varepsilon_{ij}|}$ | 对照：占据数 $\times$ 对能量 |

三种排序在**固定嵌入空间尺寸**下对比，唯一差别就是权重——这就是 stage 3/5 的受控实验设计。

---

## 3. 代码实现逻辑

三个**全新文件**，`BNO_bath.py` 与 `ssdmet.py` 未改动。

### 3.1 `embed_sim/pno_bath.py`（核心，无状态函数集）

| 函数 | 职责 |
|---|---|
| `_semicanonicalize(fock_ao, C)` | 在 $C$ 张成的空间内对角化 AO Fock，返回能量 + 旋转后系数 + 旋转 |
| `_mo_eri(mf, C1..C4)` | $(C_1 C_2\|C_3 C_4)$ 四指标积分（DF 或常规） |
| `_pno_density(T, diag)` | Eq. 2.3 的对密度 |
| `_pair_energy(T, K)` | Eq. 2.2 的对能量 |
| `_collect_pairs(eri, ...)` | **逐对**算 $K,T,\varepsilon_{ij}$、对角化 $D^{ij}$ 得 $n,v$，以及 $c_{\tilde n}$（Eq. 2.4） |
| `_select_pnos(...)` | 两级筛选：`t_pair` 筛对 + `t_pno`/`t_orb_energy` 筛轨道；阈值模式取并集，固定尺寸模式取加权 SVD |
| `_recovered_energy(data, B)` | 在子空间 $B$ 内**重解**半正则 MP2（非投影），返回实际可回收的相关能 |
| `_tune_to_energy(...)` | 二分 `t_orb_energy` 使丢弃相关能 ≈ 目标 |
| `_complement(B, n)` | 求 $B$ 在 $n$ 维空间的正交补（保持总 span 完备） |
| `build_pno_rotations(...)` | 主入口：虚侧 + 占据侧分别走上面流程，返回 `{core_bath, core_rest, vir_bath, vir_rest}` |
| `get_RPNO_bath(...)` | 与 `get_RMP2_bath` 同签名的替换品，返回 LO 基下的 `(bath, core, vir)` |

**两侧对称**：
- 虚侧：对 = 嵌入占据 $(i,j)$，目标 = 冻结虚 $(a,b)$，`sign=+1`，补「嵌入占据→冻结虚」激发。
- 占据侧：对 = 嵌入虚 $(a,b)$，目标 = 冻结占据 $(i,j)$，`sign=-1`，补「冻结占据→嵌入虚」激发。

两类激发互补，各自有独立的 $\varepsilon$、丢弃能量与 bath 旋转。

### 3.2 `embed_sim/ssdmet_pno.py`（接线）

| 函数/类 | 职责 |
|---|---|
| `expand_bath_pno(mydmet, **kw)` | 对已 build 的 DMET 对象做**后处理式**扩展：LO 基里拼 `[es | new_bath]`，重建 `es_int1e/2e`、`es_dm`、`es_mf`、`fo_ene`，并检查精确性关系 |
| `check_exactness(d)` | 正交性、完备性、$E_\text{es\_mf}+E_\text{fo}-E_\text{mf}$ 三项诊断 |
| `SSDMET_PNO` | 便利子类：`pno_option=dict(...)` 自动在 build 里做 stage2 扩展 |

**为什么是后处理式**：扩展在 LO 基内做正交旋转，`[bath|rest]` 是 frozen 空间的完整正交基，
所以**总 span 严格不变** → DMET 精确关系在扩展后仍成立（stage 1 验证这点）。

### 3.3 数据流

```
mf (全体系 RHF)
 └─ SSDMET.build()          → 1-RDM bath，得到 C_es, C_fo, C_fv, es_mf
      └─ expand_bath_pno()
           └─ build_pno_rotations(mf, es_mf, C_es, C_fo, C_fv, ...)
                ├─ 半正则化 C_fo / C_fv / 嵌入占据 / 嵌入虚
                ├─ 虚侧: _mo_eri(C_eo,C_fv,C_eo,C_fv) → _collect_pairs → _select_pnos/_tune_to_energy
                ├─ 占据侧: _mo_eri(C_ev,C_fo,C_ev,C_fo) → ...（镜像）
                └─ 返回 4 个正交旋转 [bath|rest]（在各自空间内）
           └─ 拼 [es|core_bath|vir_bath]，重建 es 各量，check_exactness
```

---

## 4. 工作流程（参数语义）

| 参数 | 含义 | 单位 | 类比 DLPNO |
|---|---|---|---|
| `t_pair` | 对能量阈值，$|w_{ij}\varepsilon_{ij}|$ 小于此的对不贡献 bath | Hartree | `T_CutPairs` |
| `t_pno` | PNO 占据数阈值 | 无量纲 | `T_CutPNO` |
| `t_orb_energy` | 轨道能量份额阈值 $c_{\tilde n}$ | Hartree | （本工作新增） |
| `nbath_occ` / `nbath_vir` | 固定该侧 bath 轨道数（等尺寸对比用） | 整数 | — |
| `e_target` | 丢弃相关能目标（二分 `t_orb_energy`） | Hartree | — |
| `score` | 固定尺寸模式的排序权重：`occ` / `energy` / `occ_x_pair` | — | — |

优先级：`e_target` > `t_orb_energy` > `t_pno`（对轨道筛选）；`nbath_*` 是另一条独立路径（固定尺寸）。

---

## 5. 当前实测结论（截至 2026-08-22，**含负面结果**）

### 5.1 已验证 ✓

- **结构完整性**（stage 1）：`orth ~1e-12`、DMET 精确关系偏差 `dev ~1e-13`、`missing=0`，
  在 ethane / water 两体系、多阈值下全通过。bath/frozen 正交旋转与 span 保持正确。

### 5.2 负面结果（重要，需诚实记录）

1. **总能量上，占据数排序本来就接近最优，能量加权没赢。**
   stage 3（固定尺寸、比 $E_\text{corr}$）里 `occ` / `energy` / `occ_x_pair` 三者误差
   量级 $10^{-5}\sim10^{-3}$ Ha，**符号随体系与尺寸随机翻转**，没有一致赢家。
   —— 这是理论上应该发生的：按占据数截断 MP2 自然轨道是振幅矩阵的 Eckart–Young 最优
   低秩逼近，任何加权都只会偏离最优。**论文关心的是 relative energy，不是总能量**，所以
   stage 3 测的是错的量。

2. **阈值模式的动态范围近乎为零。** ethane：`t_pno` 从 $10^{-3}$ 到 $10^{-4}$，`nes`
   从 25 直接跳到 58（全空间）。这就是「并集膨胀」风险，实测确认。**阈值模式不能当
   尺寸旋钮用**；可用的是固定尺寸和 `e_target` 两种模式。

3. **`e_discarded` 低估真实误差约 5 倍。** water `e_target=10^{-2}` 报 discarded = 9.0e-3，
   但嵌入 MP2 实际误差 5.2e-2。原因：(a) 半正则 MP2 估计忽略 occ–vir Fock 非对角块；
   (b) 往 cluster 加轨道会改变整个嵌入 SCF，不只是加激发通道。**它是单调可控的旋钮，
   不是误差棒**，不能照搬论文「误差不超过阈值」的宣称。

4. **`nfo=0` 使占据侧从未被执行。** 默认 `threshold=1e-12` 太紧，环境占据轨道全被判为
   纠缠而进 bath。需设 `bath_norb` 才能造出非平凡 frozen-occupied（已修）。

### 5.3 由此修正的设计

- 新增 Eq. 2.4 的严格能量份额 $c_{\tilde n}$，替代「占据数 × 对能量」的双重计数错误。
- `e_target` 二分改到 `t_orb_energy` 旋钮上，整条选择链统一为能量单位。
- 加了 stage 5（直接测 C–C 拉伸的 $\Delta E_\text{corr}$）——这才是对能量判据真正该比的量。

---

## 6. 尚未完成 / 待验证

1. **Stage 5 结果**（ethane C–C 拉伸的能量差）——判据见下节。
2. **Δ-bath**（对能量机制的唯一可能用武之地）——按用户要求，等 PNO 部分测通后再做。
3. 开壳层（ROHF/UHF）版本——当前 `pno_bath` 仅支持闭壳层（`build_pno_rotations` 有显式断言）。
4. DF 变体与 `df.DFSSDMET` 的接线。

### Stage 5 的判读标准（事先声明，避免事后找理由）

三种排序在 $\Delta E_\text{corr}$（kcal/mol）上的误差若：
- **`energy` 显著小于 `occ`（≳2×，且随尺寸稳定）** → 对能量判据在能量差上有效，Δ-bath 值得做、且应做为主线；
- **三者随机翻转、量级相当** → 能量加权在 DMET bath 里不成立，Δ-bath 不应照「能量加权」做，真正该动的是**跨结构轨道映射/一致性**（论文 DOS 部分 + 你们调研报告反复验证的「一致性 > 空间大小」）；
- **`occ_x_pair` 反而最好** → 我对「双重计数」的分析是错的，需重想。

> 注意：stage 5 当前两个几何**各自独立选 bath**，里面混着「选择判据」与「跨结构不一致」
> 两个误差源。若三者接近，说明主导误差是后者——这本身支持「Δ-bath 应做成跨结构一致
> 而非能量加权」的结论。

---

## 7. 文件清单

| 文件 | 状态 |
|---|---|
| `embed_sim/pno_bath.py` | 核心，已写 |
| `embed_sim/ssdmet_pno.py` | 接线，已写 |
| `examples/test_example/pno_bath_test.py` | 5 阶段测试，已写 |
| `BNO_bath.py` / `ssdmet.py` | **未改动** |
