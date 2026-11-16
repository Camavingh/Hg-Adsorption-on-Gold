
import os
import csv
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate

from nequip.integrations.ase import NequIPCalculator

# ==================================================
# PARAMETERS
# ==================================================

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

DEVICE = "cuda"
HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500

HG_REFERENCE_ENERGY = -0.1235  # eV, same NequIP model

# ==================================================
# DIRECTORIES
# ==================================================

STRUCTURE_DIR = "convergence/outputs/structures"
LOG_DIR = "convergence/outputs/logs"

os.makedirs(STRUCTURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==================================================
# NEQUIP CALCULATOR
# ==================================================

calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device=DEVICE,
    chemical_species_to_atom_type_map=True
)

# ==================================================
# LOAD RELAXED 4x4x4 CLEAN SLAB
# ==================================================

clean_slab = read(
    f"{STRUCTURE_DIR}/clean_Au100_4x4x4_NequIP.extxyz"
)

# Identify and fix bottom layer (16 atoms)
z = clean_slab.positions[:, 2]

bottom_indices = np.where(
    np.isclose(z, z.min(), atol=0.01)
)[0]

clean_slab.set_constraint(
    FixAtoms(indices=bottom_indices)
)

clean_slab.calc = calc

E_clean = clean_slab.get_potential_energy()

print("Clean Au(100) energy:", E_clean, "eV")
print("Fixed atoms:", len(bottom_indices))
print("Isolated Hg energy:", HG_REFERENCE_ENERGY, "eV")

# ==================================================
# DETERMINE ADSORPTION SITES
# ==================================================

positions = clean_slab.get_positions()
z = positions[:, 2]

top_mask = z > (z.max() - 0.5)
top_xy = positions[top_mask][:, :2]

cell_xy = clean_slab.cell[:2, :2]
center = cell_xy.sum(axis=0) / 2.0

dists = np.linalg.norm(top_xy - center, axis=1)
ref = top_xy[np.argmin(dists)]

vecs = top_xy - ref
lengths = np.linalg.norm(vecs, axis=1)

vecs = vecs[lengths > 0.1]
lengths = lengths[lengths > 0.1]

vecs = vecs[np.argsort(lengths)]

if len(vecs) < 2:
    raise RuntimeError("Not enough surface neighbours to identify sites")

a1 = vecs[0]
a2 = None

for v in vecs[1:]:
    cross = abs(a1[0] * v[1] - a1[1] * v[0])
    if cross > 0.1:
        a2 = v
        break

if a2 is None:
    raise RuntimeError("Could not identify two independent surface vectors")

sites = {
    "ontop": ref,
    "bridge": ref + 0.5 * a1,
    "hollow": ref + 0.5 * (a1 + a2)
}

# ==================================================
# ADSORPTION CALCULATIONS
# ==================================================

results = []

for name, xy in sites.items():

    print("\n===================================")
    print("Calculating site:", name)
    print("===================================")

    ads = clean_slab.copy()
    ads.calc = None

    add_adsorbate(
        ads,
        "Hg",
        height=HG_HEIGHT,
        position=tuple(xy)
    )

    ads.calc = calc

    logfile = f"{LOG_DIR}/Hg_Au100_4x4x4_{name}.log"

    trajectory = (
        f"{STRUCTURE_DIR}/Hg_Au100_4x4x4_{name}.traj"
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

    E_ads = (
        E_total
        - E_clean
        - HG_REFERENCE_ENERGY
    )

    forces = ads.get_forces()

    movable_indices = np.ones(len(ads), dtype=bool)
    movable_indices[bottom_indices] = False

    max_force = np.linalg.norm(
        forces[movable_indices],
        axis=1
    ).max()

    hg_position = ads.positions[-1]

    print("Total energy:", E_total, "eV")
    print("Adsorption energy:", E_ads, "eV")
    print("Maximum movable force:", max_force, "eV/Angstrom")
    print("Converged:", converged)
    print("Hg final position:", hg_position)

    write(
        f"{STRUCTURE_DIR}/Hg_Au100_4x4x4_{name}.extxyz",
        ads
    )

    results.append({
        "system": "4x4x4",
        "site": name,
        "E_clean_eV": E_clean,
        "E_Hg_eV": HG_REFERENCE_ENERGY,
        "E_total_eV": E_total,
        "E_ads_eV": E_ads,
        "max_force_eV_A": max_force,
        "converged": converged,
        "Hg_x_A": hg_position[0],
        "Hg_y_A": hg_position[1],
        "Hg_z_A": hg_position[2]
    })

# ==================================================
# SAVE RESULTS
# ==================================================

csv_file = "convergence/outputs/adsorption_4x4x4_results.csv"

with open(csv_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

# ==================================================
# SUMMARY
# ==================================================

print("\n===================================")
print("NequIP 4x4x4 Hg@Au(100) RESULTS")
print("===================================")

for result in results:
    print(
        f"{result['site']:8s}: "
        f"{result['E_ads_eV']:.6f} eV | "
        f"Converged: {result['converged']}"
    )

best = min(results, key=lambda x: x["E_ads_eV"])

print("\nMost stable site:", best["site"])
print("Adsorption energy:", best["E_ads_eV"], "eV")
print("\nSaved:", csv_file)

