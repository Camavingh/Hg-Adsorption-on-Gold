
import os
import csv
import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate

from nequip.integrations.ase import NequIPCalculator

# --------------------------------------------------
# Parameters
# --------------------------------------------------
MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

DEVICE = "cuda"
HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500

# --------------------------------------------------
# Directories
# --------------------------------------------------
os.makedirs("outputs/structures", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

# --------------------------------------------------
# NequIP calculator
# --------------------------------------------------
calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device=DEVICE,
    chemical_species_to_atom_type_map=True
)

# --------------------------------------------------
# Load relaxed clean Au(100) slab
# --------------------------------------------------
clean_slab = read(
    "outputs/structures/clean_Au100_NequIP.extxyz"
)

clean_slab.set_constraint(
    FixAtoms(indices=np.arange(25))
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

a1 = vecs[0]
a2 = None

for v in vecs[1:]:
    if abs(a1[0]*v[1] - a1[1]*v[0]) > 0.1:
        a2 = v
        break

if a2 is None:
    raise RuntimeError("Could not identify surface neighbour vectors")

sites = {
    "ontop": ref,
    "bridge": ref + 0.5*a1,
    "hollow": ref + 0.5*(a1+a2)
}

# --------------------------------------------------
# Adsorption calculations
# --------------------------------------------------
results = []

for name, xy in sites.items():

    print("\nCalculating site:", name)

    ads = clean_slab.copy()
    ads.calc = None

    add_adsorbate(
        ads,
        "Hg",
        height=HG_HEIGHT,
        position=tuple(xy)
    )

    ads.calc = calc

    logfile = f"outputs/logs/Hg_Au100_{name}.log"
    trajectory = f"outputs/structures/Hg_Au100_{name}.traj"

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
    movable = np.ones(len(ads), dtype=bool)
    movable[:25] = False

    max_force = np.linalg.norm(
        forces[movable], axis=1
    ).max()

    hg_position = ads.positions[-1]

    print("Total energy:", E_total, "eV")
    print("Adsorption energy:", E_ads, "eV")
    print("Maximum movable force:", max_force, "eV/Å")
    print("Converged:", converged)
    print("Hg final position:", hg_position)

    write(
        f"outputs/structures/Hg_Au100_{name}.extxyz",
        ads
    )

    results.append({
        "site": name,
        "E_clean_eV": E_clean,
        "E_Hg_eV": E_hg,
        "E_total_eV": E_total,
        "E_ads_eV": E_ads,
        "max_force_eV_A": max_force,
        "converged": converged
    })

# --------------------------------------------------
# Save CSV results
# --------------------------------------------------
csv_file = "outputs/adsorption_energy_results.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

best = min(results, key=lambda x: x["E_ads_eV"])

print("\n===================================")
print("NequIP-OAM-M Hg@Au(100) RESULTS")
print("===================================")

for result in results:
    print(
        result["site"],
        f"{result['E_ads_eV']:.6f} eV"
    )

print("\nMost stable site:", best["site"])
print("Adsorption energy:", best["E_ads_eV"], "eV")

