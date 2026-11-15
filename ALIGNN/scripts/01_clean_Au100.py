
import os
import numpy as np
from ase.build import fcc100
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

# --------------------------------------------------
# Parameters
# --------------------------------------------------
LATTICE_CONSTANT = 4.08
SLAB_SIZE = (5, 5, 6)
VACUUM = 12.0
FMAX = 0.005
MAX_STEPS = 500
DEVICE = "cuda"

# --------------------------------------------------
# Directories
# --------------------------------------------------
os.makedirs("outputs/structures", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

# --------------------------------------------------
# Construct Au(100) slab
# --------------------------------------------------
slab = fcc100(
    "Au",
    size=SLAB_SIZE,
    a=LATTICE_CONSTANT,
    vacuum=VACUUM
)

# Fix bottom 25 atoms, matching the MACE calculation
z = slab.positions[:, 2]
fixed_indices = np.argsort(z)[:25]

slab.set_constraint(
    FixAtoms(indices=fixed_indices)
)

print("Number of Au atoms:", len(slab))
print("Fixed atoms:", len(fixed_indices))
print("Cell dimensions:")
print(slab.cell)

# --------------------------------------------------
# ALIGNN-FF calculator
# --------------------------------------------------
calc = AlignnAtomwiseCalculator(
    path=default_path(),
    device=DEVICE
)

slab.calc = calc

# --------------------------------------------------
# Initial energy
# --------------------------------------------------
initial_energy = slab.get_potential_energy()

print("\nInitial energy:", initial_energy, "eV")

# --------------------------------------------------
# Geometry optimization
# --------------------------------------------------
optimizer = BFGS(
    slab,
    logfile="outputs/logs/clean_Au100_relax.log",
    trajectory="outputs/clean_Au100_relax.traj"
)

optimizer.run(
    fmax=FMAX,
    steps=MAX_STEPS
)

# --------------------------------------------------
# Final results
# --------------------------------------------------
final_energy = slab.get_potential_energy()

forces = slab.get_forces()
max_force = np.max(
    np.linalg.norm(forces, axis=1)
)

print("\nRelaxation completed")
print("Final energy:", final_energy, "eV")
print("Energy per atom:", final_energy / len(slab), "eV/atom")
print("Maximum force:", max_force, "eV/Angstrom")
print("Converged:", optimizer.converged())

# --------------------------------------------------
# Save relaxed structure
# --------------------------------------------------
slab.write(
    "outputs/structures/clean_Au100_ALIGNN.extxyz"
)

slab.write(
    "outputs/structures/clean_Au100_ALIGNN.vasp",
    direct=True,
    vasp5=True
)

print("\nStructures saved successfully.")

