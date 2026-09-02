import sys
import os
sys.path.append('/data/home/cuihx/pyscf_learn/LAMP_emb/')
import numpy as np
from pyscf import scf,gto,tools, mcscf
from embed_sim import myavas, sacasscf_mixer, siso, ssdmet, rdiis
import time
### this work need to expand the bath by LODMET and amplify the space of impurity.
start_time = time.time()    
# load the structure of CoSPh4 and give correct spin and charge

mol = gto.M(atom = " H 0 0 0; F 0 0 1.10", basis = {'Co':' def2tzvp', 'S':'def2tzvp','C':'6-31G*', 'H':'ccpvdz', 'F':'ccpvdz'}, symmetry=0, spin = 0, charge = 0, verbose = 4)

current_file_name = os.path.basename(sys.argv[0])

print(f"当前文件名: {current_file_name}")
title = current_file_name.split('.py')[0]

mf = scf.RHF(mol)
chk_fname = title + '_rhf.chk'

mf.chkfile = chk_fname
mf.init_guess = 'atom'
mf.level_shift = .1
#mf.diis = rdiis.RDIIS(rdiis_prop = 'dS', imp_idx = mol.search_ao_label(['Co 3d']), power = 0.2)
mf.max_cycle = 1000
mf.max_memory = 100000

# 使用下方的可以正常读取chk文件不用进行那么多圈的scf迭代
#dm = mf.from_chk('CoSPh4_DMET_CAS_PT2_Coimp_rohf.chk')
mf.kernel()



mydmet = ssdmet.SSDMET(mf, title = title, imp_idx = ('H.*','F.*'),threshold = 1e-12)
mydmet.build()

# run ccsdt
e_tot = mydmet.ccsdt_solver()

mydmet2 = ssdmet.SSDMET(mf, title = title+'_2', imp_idx = ('H.*'),threshold = 1e-12)
mydmet2.build()
e_tot2 = mydmet2.ccsdt_solver()
print('CCSD(T) calculation for the full system:')
origin_cc = mf.CCSD().run()
et = origin_cc.ccsd_t()
print('CCSD(T) correlation energy', origin_cc.e_corr + et)
print('CCSD(T) total energy', mf.e_tot + origin_cc.e_corr + et)
print("="*50)
print('DMET-CCSDT energy 1:', e_tot)
print('DMET-CCSDT energy 2:', e_tot2)
#ncas, nelec, es_mo = mydmet.avas('Co 3d', minao='def2tzvp',  threshold=0.5)
#es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec)
#es_cas = sacasscf_mixer.sacasscf_mixer(mydmet.es_mf, ncas, nelec, statelis=[0, 40, 0, 10])
'''
es_cas.kernel(es_mo)
## NEVPT2
es_corr = sacasscf_mixer.sacasscf_nevpt2(es_cas)
es_cas.fcisolver.e_states = es_cas.fcisolver.e_states + es_corr
total_cas = mydmet.total_cas(es_cas)

#from pyscf import molden
molden_filename = f"{title}_dmet_cas_natural.molden"
tools.molden.from_mcscf(total_cas, molden_filename, cas_natorb=True)
total_cas.analyze()


mysiso = siso.SISO(title, total_cas)
mysiso.kernel()
'''

# 记录结束时间并计算总时长
end_time = time.time()
total_time = end_time - start_time

# 格式化输出时间
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"\n{'='*50}")
print(f"程序总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
print(f"总秒数: {total_time:.2f}秒")
print(f"{'='*50}")
