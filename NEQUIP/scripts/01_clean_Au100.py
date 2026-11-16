import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from nequip.integrations.ase import NequIPCalculator

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

DEVICE = "cuda"
FMAX = 0.005
MAX_STEPS = 500

atoms = read("outputs/structures/clean_Au100_initial.extxyz")

atoms.set_constraint(
    FixAtoms(indices=np.arange(25))
)

atoms.calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device=DEVICE,
    chemical_species_to_atom_type_map=True
)

optimizer = BFGS(
    atoms,
    logfile="outputs/logs/clean_Au100.log",
    trajectory="outputs/structures/clean_Au100.traj"
)

converged = optimizer.run(
    fmax=FMAX,
    steps=MAX_STEPS
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
max_force = np.linalg.norm(forces, axis=1).max()

write("outputs/structures/clean_Au100_NequIP.extxyz", atoms)

print("NequIP clean Au(100)")
print("Atoms:", len(atoms))
print("Energy:", energy, "eV")
print("Maximum force:", max_force, "eV/Angstrom")
print("Converged:", converged)
