import os
import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from nequip.integrations.ase import NequIPCalculator

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

INPUT = (
    "convergence/consistent_geometry/outputs/structures/"
    "Au100_5x5x6_vac16_initial.extxyz"
)

OUTPUT = (
    "convergence/consistent_geometry/outputs/structures/"
    "clean_Au100_5x5x6_vac16_NequIP.extxyz"
)

LOG = (
    "convergence/consistent_geometry/outputs/logs/"
    "clean_Au100_5x5x6_vac16.log"
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

slab = read(INPUT)

z = slab.positions[:, 2]
bottom_layer = np.isclose(z, z.min(), atol=0.1)
fixed_indices = np.where(bottom_layer)[0]

slab.set_constraint(FixAtoms(indices=fixed_indices))

slab.calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device="cuda",
    chemical_species_to_atom_type_map=True
)

optimizer = BFGS(
    slab,
    trajectory=LOG.replace(".log", ".traj"),
    logfile=LOG
)

converged = optimizer.run(fmax=0.005, steps=500)

energy = slab.get_potential_energy()
forces = slab.get_forces()
movable = np.ones(len(slab), dtype=bool)
movable[fixed_indices] = False

max_force = np.linalg.norm(forces[movable], axis=1).max()

write(OUTPUT, slab)

print("NequIP 5x5x6 clean Au(100), 16 Å vacuum")
print("Atoms:", len(slab))
print("Fixed atoms:", len(fixed_indices))
print("Energy:", energy, "eV")
print("Maximum movable force:", max_force, "eV/Å")
print("Converged:", converged)
print("Saved:", OUTPUT)
