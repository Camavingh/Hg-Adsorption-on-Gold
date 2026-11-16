
import os
import numpy as np
from ase.build import fcc100
from ase.io import write

# ==========================================
# PARAMETERS
# ==========================================

LATTICE_PARAMETER = 4.08
SIZE = (4, 4, 6)
VACUUM = 12.0

OUTPUT = (
    "convergence/outputs/structures/"
    "clean_Au100_4x4x6_initial.extxyz"
)

# ==========================================
# BUILD Au(100) SLAB
# ==========================================

os.makedirs(
    "convergence/outputs/structures",
    exist_ok=True
)

slab = fcc100(
    "Au",
    size=SIZE,
    a=LATTICE_PARAMETER,
    vacuum=VACUUM,
    orthogonal=True
)

slab.pbc = (True, True, False)

# ==========================================
# IDENTIFY BOTTOM LAYER
# ==========================================

z = slab.positions[:, 2]

bottom_indices = np.where(
    np.isclose(z, z.min(), atol=0.01)
)[0]

# ==========================================
# VERIFY GEOMETRY
# ==========================================

unique_layers = np.unique(np.round(z, 3))

print("Au(100) 4x4x6 slab")
print("---------------------------")
print("Total atoms:", len(slab))
print("Layers:", len(unique_layers))
print("Atoms per layer:", len(bottom_indices))
print("Surface cell:")
print(slab.cell)
print("PBC:", slab.pbc)
print("Layer z coordinates:", unique_layers)
print("Vacuum:", VACUUM, "Angstrom")

# ==========================================
# SAVE STRUCTURE
# ==========================================

write(OUTPUT, slab)

print("\nSaved:", OUTPUT)

