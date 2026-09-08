#######################################################################
#
# 这是一LODMET的计算程序，计算之前可能需要准备：
# title + '_cahf.chk'
# 修改sys.path
#
#######################################################################

import sys, os
sys.path.insert(0,"/Users/cuihengxin/Desktop/2025-2030phd/8dmet4reac/iao_dmet/DMET")

from embed_sim import sacasscf_mixer, myavas, ssdmet_modify5, rdiis, cahf
import numpy as np
from pyscf import scf, gto, df
from scipy.linalg import svd
import tempfile
import time

#######################################################################
#
# 第一部分：全电子CAHF计算
# 注意修改：title、atom、spin、charge、basis、densityfitting
#
#######################################################################

start = time.time()  # 记录开始时间

omega = -0.1
title = 'fel2oh'

start = time.time()  # 记录开始时间

mol = gto.M(atom = '''Fe 0 0 0''',
basis='ccpvdz', symmetry=0, spin=5, charge=3, verbose=4)
mol.incore_anyway = True
mf_basis = cahf.CAHF(mol, ncas=5, nelecas=5, spin=1).x2c().density_fit()
mf_basis.with_df.auxbasis = df.make_auxbasis(mol)
mf_basis.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=mol.search_ao_label(['Fe.*d']), power=0.2)
mf_basis.chkfile = title + '_cahf.chk'
mf_basis.init_guess = 'chk'
mf_basis.level_shift = .2
mf_basis.diis_space = 16
mf_basis.max_memory = 4000

# 本地若无收敛的cahf.chk：先收敛常规CAHF(不加omega项)并保存
# 加omega=-0.1的r2项后SCF从atom初猜无法收敛(|g|~1.7震荡)，服务器流程假定chk已存在
if os.path.exists(mf_basis.chkfile):
    mf_basis.max_cycle = 0
else:
    mf_basis.max_cycle = 100
    mf_basis.kernel()
    mf_basis.max_cycle = 0

# omega项只进入单点能评估与后续SA-CASSCF的hcore（轨道取自chk）
hcore = (mf_basis.get_hcore() + 0.5 * omega * mol.intor('int1e_r2'))
mf_basis.get_hcore = lambda *args: hcore
mf_basis.get_ovlp = lambda *args: mol.intor_symmetric('int1e_ovlp')
mf_basis._eri = mf_basis.with_df.ao2mo(np.eye(mol.nao_nr()))

mf_basis.kernel()

ncas, nelec, mo = myavas.avas(mf_basis, ['Fe 3d'], minao='ccpvdz', openshell_option=2, threshold=0.5)

mycas = sacasscf_mixer.sacasscf_mixer(mf_basis, ncas, nelec, statelis=[0, 1, 0, 1, 0, 1])
mycas.max_cycle = 1000
mycas.max_memory = 4000
mycas.kernel(mo)
from pyscf import tools
tools.molden.from_mcscf(mycas, title + '_cahf.molden',cas_natorb =True)
print("""np.argmax(np.abs(mycas.mo_coeff), axis=0):""")
print(np.argmax(np.abs(mycas.mo_coeff), axis=0))
print("""mycas.mo_coeff[21:26,9:14]:""")
print(mycas.mo_coeff[21:26,9:14])

mol = gto.M(atom = '''FeL2OH.xyz''',
basis='cc-pVDZ', symmetry=0, spin=1, charge=0, verbose=4)

chk_fname = title + '_ls.chk'

auxbasis = df.autoaux(mol)
mf = scf.rohf.ROHF(mol).density_fit(auxbasis=auxbasis).x2c()
# mf = cahf.CAHF(mol, ncas=5, nelecas=5, spin=1).x2c().density_fit()

folder = './'
if os.path.exists(folder+title+'_df.h5'):
    mf.with_df._cderi = folder+title+'_df.h5'
    mf.with_df.auxmol = df.make_auxmol(mol, auxbasis=auxbasis)
else:
    mf.with_df._cderi_to_save = folder+title+'_df.h5'

mf.chkfile = chk_fname
mf.diis = rdiis.RDIIS(rdiis_prop='dS', imp_idx=mol.search_ao_label(['Fe.*d']), power=0.2)
mf.init_guess = 'atom'
mf.level_shift = .2
mf.diis_space = 16
mf.max_cycle = 10000
mf.max_memory = 4000
mf.kernel()

end = time.time()    # 记录结束时间
print(f"全电子CAHF耗时: {end - start:.3f} 秒")
assert(mf.converged)

#######################################################################
#
# 第二部分：LODMET的初始化（包括生成双电子积分）
# 注意修改：imp_idx
#
#######################################################################

orb_ao2mo = np.eye(mol.nao_nr())
orb_ao2mo[0:43, 21:26] = mycas.mo_coeff[:, 9:14]

print("Fe: ", gto.mole._aolabels2baslst(mol, ['Fe.*'], base=0))
print("Fe.3d: ", gto.mole._aolabels2baslst(mol, ['Fe.3d'], base=0))
print("Fe.*s: ", gto.mole._aolabels2baslst(mol, ['Fe.*s'], base=0))
print("Fe.*p: ", gto.mole._aolabels2baslst(mol, ['Fe.*p'], base=0))
print("Fe.*d: ", gto.mole._aolabels2baslst(mol, ['Fe.*d'], base=0))
print("Fe.*f: ", gto.mole._aolabels2baslst(mol, ['Fe.*f'], base=0))
sys.exit()

param_list = list(range(0,201,10))
nev4_2 = []
nev6_2 = []
for param in param_list:
    print(f"计算参数 {param} ")

    start = time.time()  # 记录开始时间

    if param > 0:
        mydmet = ssdmet_modify5.SSDMET(mf, title=title, imp_idx=['Fe.3d'], es_natorb=False, bath_option={'ROMP2':param})
    else:
        mydmet = ssdmet_modify5.SSDMET(mf, title=title, imp_idx=['Fe.3d'])
    mydmet.build(basis_rot=orb_ao2mo, save_chk=False, restore_imp = True)

    end = time.time()    # 记录结束时间
    print(f"LODMET耗时: {end - start:.3f} 秒")
    print("mycas.fo_ene:", mydmet.fo_ene)

    #######################################################################
    #
    # 第三部分：计算DMET后SA-CASSCF
    # 注意修改：ncas、nelec、statelis、weights（两部分）
    # 根据全电子CASSCF使用sort_mo
    #
    #######################################################################

    start = time.time()  # 记录开始时间

    ncas, nelec, mo = mydmet.avas(['Fe 3d'], minao='ccpvdz', openshell_option=2, threshold=0.5)

    es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec, statelis=[0, 1, 0, 1, 0, 1])
    es_cas.max_cycle = 1000
    es_cas.max_memory = 4000
    es_cas.chkfile = title + "_casscf.chk"
    es_cas.kernel(mo)
    assert(es_cas.converged)

    end = time.time()    # 记录结束时间
    print(f"DMET中CASSCF的耗时: {end - start:.3f} 秒")

    # sacasscf_mixer.analysis(mydmet.total_cas(es_cas))

    #######################################################################
    #
    # 第四部分：计算DMET后NEVPT2
    #
    #######################################################################

    start = time.time()  # 记录开始时间

    e_corr = sacasscf_mixer.sacasscf_nevpt2(es_cas)

    end = time.time()    # 记录结束时间

    print(f"DMET中NEVPT2耗时： {end - start:.3f} 秒")
    print("casscf(4-2,6-2):", es_cas.fcisolver.e_states[1] - es_cas.fcisolver.e_states[0], es_cas.fcisolver.e_states[2] - es_cas.fcisolver.e_states[0])
    print("nevpt2_corr(4-2,6-2):", e_corr[1] - e_corr[0], e_corr[2] - e_corr[0])
    print("nevpt2(4-2,6-2):", es_cas.fcisolver.e_states[1] - es_cas.fcisolver.e_states[0] + e_corr[1] - e_corr[0], es_cas.fcisolver.e_states[2] - es_cas.fcisolver.e_states[0] + e_corr[2] - e_corr[0])
    print("nevpt2(4-2,6-2):", (es_cas.fcisolver.e_states[1] - es_cas.fcisolver.e_states[0] + e_corr[1] - e_corr[0])*27.2113863, (es_cas.fcisolver.e_states[2] - es_cas.fcisolver.e_states[0] + e_corr[2] - e_corr[0])*27.2113863)

    print(f"已生成 结果_{param}.txt")

    nev4_2.append((es_cas.fcisolver.e_states[1] - es_cas.fcisolver.e_states[0] + e_corr[1] - e_corr[0])*27.2113863)
    nev6_2.append((es_cas.fcisolver.e_states[2] - es_cas.fcisolver.e_states[0] + e_corr[2] - e_corr[0])*27.2113863)

print("param_list:", param_list)
print("nev4_2:", nev4_2)
print("nev6_2:", nev6_2)