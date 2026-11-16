
import os
import csv
import numpy as np

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate

from nequip.integrations.ase import NequIPCalculator

# ==========================================
# PARAMETERS
# ==========================================

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

DEVICE = "cuda"
HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500
E_HG = -0.1235

BASE = "convergence/consistent_geometry/outputs"
STRUCTURES = f"{BASE}/structures"
LOGS = f"{BASE}/logs"

os.makedirs(STRUCTURES, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# ==========================================
# CALCULATOR
# ==========================================

calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device=DEVICE,
    chemical_species_to_atom_type_map=True
)

# ==========================================
# LOAD CLEAN SLAB
# ==========================================

clean = read(
    f"{STRUCTURES}/clean_Au100_4x4x6_NequIP.extxyz"
)

z = clean.positions[:, 2]

bottom = np.where(
    np.isclose(z, z.min(), atol=0.01)
)[0]

clean.set_constraint(FixAtoms(indices=bottom))
clean.calc = calc

E_clean = clean.get_potential_energy()

print("Clean slab energy:", E_clean, "eV")
print("Isolated Hg energy:", E_HG, "eV")
print("Fixed atoms:", len(bottom))

# ==========================================
# ADSORPTION SITES
# ==========================================

positions = clean.positions
z = positions[:, 2]

top_xy = positions[z > z.max() - 0.5, :2]

cell_xy = clean.cell[:2, :2]
center = cell_xy.sum(axis=0) / 2.0

distances = np.linalg.norm(top_xy - center, axis=1)
ref = top_xy[np.argmin(distances)]

vectors = top_xy - ref
lengths = np.linalg.norm(vectors, axis=1)

valid = lengths > 0.1
vectors = vectors[valid]
lengths = lengths[valid]

vectors = vectors[np.argsort(lengths)]

if len(vectors) < 2:
    raise RuntimeError("Not enough surface vectors")

a1 = vectors[0]
a2 = None

for v in vectors[1:]:
    if abs(a1[0]*v[1] - a1[1]*v[0]) > 0.1:
        a2 = v
        break

if a2 is None:
    raise RuntimeError("Could not identify independent surface vectors")

sites = {
    "ontop": ref,
    "bridge": ref + 0.5*a1,
    "hollow": ref + 0.5*(a1+a2)
}

# ==========================================
# ADSORPTION CALCULATIONS
# ==========================================

results = []

for name, xy in sites.items():

    print("\nCalculating:", name)

    ads = clean.copy()
    ads.calc = None

    add_adsorbate(
        ads,
        "Hg",
        height=HG_HEIGHT,
        position=tuple(xy)
    )

    ads.calc = calc

    optimizer = BFGS(
        ads,
        logfile=f"{LOGS}/Hg_Au100_4x4x6_{name}.log",
        trajectory=f"{STRUCTURES}/Hg_Au100_4x4x6_{name}.traj"
    )

    converged = optimizer.run(
        fmax=FMAX,
        steps=MAX_STEPS
    )

    E_total = ads.get_potential_energy()

    E_ads = E_total - E_clean - E_HG

    forces = ads.get_forces()

    movable = np.ones(len(ads), dtype=bool)
    movable[bottom] = False

    max_force = np.linalg.norm(
        forces[movable], axis=1
    ).max()

    hg_position = ads.positions[-1]

    write(
        f"{STRUCTURES}/Hg_Au100_4x4x6_{name}.extxyz",
        ads
    )

    print("Total energy:", E_total, "eV")
    print("Adsorption energy:", E_ads, "eV")
    print("Maximum movable force:", max_force, "eV/Å")
    print("Converged:", converged)
    print("Hg position:", hg_position)

    results.append({
        "system": "4x4x6",
        "site": name,
        "E_clean_eV": E_clean,
        "E_Hg_eV": E_HG,
        "E_total_eV": E_total,
        "E_ads_eV": E_ads,
        "max_force_eV_A": max_force,
        "converged": converged,
        "Hg_x_A": hg_position[0],
        "Hg_y_A": hg_position[1],
        "Hg_z_A": hg_position[2]
    })

# ==========================================
# SAVE CSV
# ==========================================

output = f"{BASE}/adsorption_4x4x6_results.csv"

with open(output, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

# ==========================================
# SUMMARY
# ==========================================

print("\nNequIP consistent 4x4x6 results")

for r in results:
    print(
        f"{r['site']:8s}: "
        f"{r['E_ads_eV']:.6f} eV | "
        f"Converged: {r['converged']}"
    )

best = min(results, key=lambda x: x["E_ads_eV"])

print("\nMost stable site:", best["site"])
print("Adsorption energy:", best["E_ads_eV"], "eV")
print("Saved:", output)

