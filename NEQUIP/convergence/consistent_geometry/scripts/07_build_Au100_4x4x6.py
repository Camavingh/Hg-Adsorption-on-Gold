
import os
from ase.build import fcc100
from ase.io import write

a = 4.08
size = (4, 4, 6)
vacuum = 12.0

output = (
    "convergence/consistent_geometry/outputs/structures/"
    "Au100_4x4x6_initial.extxyz"
)

slab = fcc100(
    "Au",
    size=size,
    a=a,
    vacuum=vacuum,
    orthogonal=True
)

slab.pbc = (True, True, False)

os.makedirs(os.path.dirname(output), exist_ok=True)
write(output, slab)

print("Atoms:", len(slab))
print("Layers:", len(set(slab.positions[:, 2].round(3))))
print("Cell:", slab.cell)
print("PBC:", slab.pbc)
print("Saved:", output)

