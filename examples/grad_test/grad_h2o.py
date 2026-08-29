import os 
import numpy as np 
from pyscf import gto, scf
mol = gto.M(atom='''
    O   0.0000000000  0.0000000000  0.0000000000
    H   1.0000000000  0.0000000000  1.0000000000
    H   -1.0000000000  0.0000000000  1.0000000
''', basis='ccpvdz', verbose=4)          # �坐标 Bohr
mf = scf.RHF(mol)
from pyscf.geomopt.geometric_solver import optimize
mol_eq = optimize(mf, maxsteps=100, tol=1e-6)
print(mol_eq.tostring())
coords = mol_eq.atom_coords(unit='Bohr')  # shape (3,3): O, H1, H2
O, H1, H2 = coords

# 键长 (Bohr -> Angstrom)
bond_length_bohr = np.linalg.norm(H1 - O)
bond_length_ang = bond_length_bohr * 0.52917721067

# 键角
v1 = H1 - O
v2 = H2 - O
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

print(f"O-H 键长: {bond_length_ang:.8f} Å")
print(f"H-O-H 键角: {angle_deg:.2f}°")