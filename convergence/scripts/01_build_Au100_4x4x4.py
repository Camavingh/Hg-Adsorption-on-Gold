
import os
import numpy as np
from ase.build import fcc100
from ase.io import write

os.makedirs("convergence/outputs/structures", exist_ok=True)

# Same lattice parameter as the baseline Au(100) slab
a = 4.08

slab = fcc100(
    "Au",
    size=(4, 4, 4),
    a=a,
    vacuum=12.0,
    orthogonal=True
)

slab.pbc = (True, True, False)

# Fix the bottom layer: 16 atoms
z = slab.positions[:, 2]
bottom_layer = np.where(
    np.isclose(z, z.min(), atol=0.01)
)[0]

print("Atoms:", len(slab))
print("Cell:", slab.cell)
print("PBC:", slab.pbc)
print("Bottom-layer atoms:", len(bottom_layer))

write(
    "convergence/outputs/structures/clean_Au100_4x4x4_initial.extxyz",
    slab
)

