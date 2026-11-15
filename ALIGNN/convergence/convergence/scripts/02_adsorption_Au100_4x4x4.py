
import os
import csv
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

# --------------------------------------------------
# Parameters
# --------------------------------------------------
DEVICE = "cuda"
HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500

# --------------------------------------------------
# Output directories
# --------------------------------------------------
os.makedirs("convergence/outputs/structures", exist_ok=True)
os.makedirs("convergence/outputs/logs", exist_ok=True)

# --------------------------------------------------
# Load relaxed clean Au(100) slab
# --------------------------------------------------
clean_slab = read(
    "convergence/outputs/structures/clean_Au100_4x4x4.extxyz"
)

# Fix bottom 16 Au atoms
fixed_indices = np.arange(16)

clean_slab.set_constraint(
    FixAtoms(indices=fixed_indices)
)

print("Au atoms:", len(clean_slab))
print("Fixed atoms:", len(fixed_indices))

# --------------------------------------------------
# ALIGNN-FF calculator
# --------------------------------------------------
calc = AlignnAtomwiseCalculator(
    path=default_path(),
    device=DEVICE
)

clean_slab.calc = calc

E_clean = clean_slab.get_potential_energy()

print("Clean slab energy:", E_clean, "eV")

# --------------------------------------------------
# Isolated Hg reference
# --------------------------------------------------
hg_atom = Atoms(
    "Hg",
    positions=[[10.0, 10.0, 10.0]],
    cell=[20.0, 20.0, 20.0],
    pbc=True
)

hg_atom.calc = calc
E_hg = hg_atom.get_potential_energy()

print("Isolated Hg energy:", E_hg, "eV")

# --------------------------------------------------
# Determine adsorption sites
# --------------------------------------------------
positions = clean_slab.positions
z = positions[:, 2]

top_indices = np.where(z > z.max() - 0.5)[0]
top_xy = positions[top_indices, :2]

center = clean_slab.cell[:2, :2].sum(axis=0) / 2

distances = np.linalg.norm(top_xy - center, axis=1)
ref = top_xy[np.argmin(distances)]

# Find nearest-neighbour vectors using periodic boundaries
cell_xy = clean_slab.cell[:2, :2]

vectors = []
for point in top_xy:
    delta = point - ref
    delta -= np.round(
        delta @ np.linalg.inv(cell_xy)
    ) @ cell_xy

    length = np.linalg.norm(delta)

    if length > 0.1:
        vectors.append((length, delta))

vectors.sort(key=lambda item: item[0])

a1 = vectors[0][1]
a2 = None

for length, vector in vectors[1:]:
    if abs(np.dot(a1, vector)) < 0.1:
        a2 = vector
        break

if a2 is None:
    raise RuntimeError("Could not identify perpendicular surface vectors")

sites = {
    "ontop": ref,
    "bridge": ref + 0.5 * a1,
    "hollow": ref + 0.5 * (a1 + a2)
}

print("\nAdsorption sites:")
for name, xy in sites.items():
    print(name, xy)

# --------------------------------------------------
# Adsorption calculations
# --------------------------------------------------
results = []

for name, xy in sites.items():

    print("\nCalculating site:", name)

    ads = clean_slab.copy()

    add_adsorbate(
        ads,
        "Hg",
        height=HG_HEIGHT,
        position=tuple(xy)
    )

    ads.set_constraint(
        FixAtoms(indices=fixed_indices)
    )

    ads.calc = calc

    logfile = (
        f"convergence/outputs/logs/Hg_Au100_4x4x4_{name}.log"
    )

    trajectory = (
        f"convergence/outputs/Hg_Au100_4x4x4_{name}.traj"
    )

    optimizer = BFGS(
        ads,
        logfile=logfile,
        trajectory=trajectory
    )

    converged = optimizer.run(
        fmax=FMAX,
        steps=MAX_STEPS
    )

    E_total = ads.get_potential_energy()
    E_ads = E_total - E_clean - E_hg

    forces = ads.get_forces()
    max_force = np.max(
        np.linalg.norm(forces, axis=1)
    )

    hg_position = ads.positions[-1]

    print("Total energy:", E_total, "eV")
    print("Adsorption energy:", E_ads, "eV")
    print("Maximum force:", max_force, "eV/Å")
    print("Converged:", converged)
    print("Hg position:", hg_position)

    output = (
        f"convergence/outputs/structures/"
        f"Hg_Au100_4x4x4_{name}.extxyz"
    )

    write(output, ads)

    results.append({
        "slab": "4x4x4",
        "vacuum_A": 12.0,
        "site": name,
        "E_clean_eV": E_clean,
        "E_Hg_eV": E_hg,
        "E_total_eV": E_total,
        "E_ads_eV": E_ads,
        "max_force_eV_A": max_force,
        "converged": converged,
        "Hg_x_A": hg_position[0],
        "Hg_y_A": hg_position[1],
        "Hg_z_A": hg_position[2]
    })

# --------------------------------------------------
# Save results
# --------------------------------------------------
csv_file = (
    "convergence/outputs/adsorption_4x4x4_results.csv"
)

with open(csv_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

best = min(results, key=lambda x: x["E_ads_eV"])

print("\n===================================")
print("ALIGNN-FF Hg@Au(100) 4x4x4 RESULTS")
print("===================================")

for result in results:
    print(
        result["site"],
        f"{result['E_ads_eV']:.6f} eV"
    )

print("\nMost stable site:", best["site"])
print("Adsorption energy:", best["E_ads_eV"], "eV")

