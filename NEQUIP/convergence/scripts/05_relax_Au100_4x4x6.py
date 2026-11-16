
import os
import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from nequip.integrations.ase import NequIPCalculator

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

atoms = read(
    "convergence/outputs/structures/"
    "clean_Au100_4x4x6_initial.extxyz"
)

# Fix the bottom layer (16 atoms)
z = atoms.positions[:, 2]
bottom = np.where(
    np.isclose(z, z.min(), atol=0.01)
)[0]

atoms.set_constraint(FixAtoms(indices=bottom))

atoms.calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device="cuda",
    chemical_species_to_atom_type_map=True
)

optimizer = BFGS(
    atoms,
    logfile="convergence/outputs/logs/"
    "clean_Au100_4x4x6.log",
    trajectory="convergence/outputs/structures/"
    "clean_Au100_4x4x6.traj"
)

converged = optimizer.run(fmax=0.005, steps=500)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
max_force = np.linalg.norm(forces, axis=1).max()

write(
    "convergence/outputs/structures/"
    "clean_Au100_4x4x6_NequIP.extxyz",
    atoms
)

print("NequIP 4x4x6 clean Au(100)")
print("Atoms:", len(atoms))
print("Fixed atoms:", len(bottom))
print("Energy:", energy, "eV")
print("Maximum force:", max_force, "eV/Angstrom")
print("Converged:", converged)

