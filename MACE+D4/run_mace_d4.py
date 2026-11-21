
import os
import csv
import numpy as np

from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.calculators.mixing import SumCalculator

from mace.calculators import mace_mp
from dftd4.ase import DFTD4


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTDIR, exist_ok=True)

FMAX = 0.005
SITES = ("ontop", "bridge", "hollow")


# ============================================================
# CALCULATORS
# ============================================================

print("Initializing MACE-MP medium on CUDA...")

mace_calc = mace_mp(
    model="medium",
    device="cuda",
    default_dtype="float64",
    dispersion=False
)

print("Initializing DFT-D4 with PBE parameters...")

d4_calc = DFTD4(method="PBE")

combined_calc = SumCalculator([mace_calc, d4_calc])


# ============================================================
# OPTIMIZATION FUNCTION
# ============================================================

def optimize(atoms, label):

    atoms.calc = combined_calc

    initial_positions = atoms.get_positions().copy()

    traj_path = os.path.join(OUTDIR, f"{label}.traj")
    log_path = os.path.join(OUTDIR, f"{label}.log")
    vasp_path = os.path.join(OUTDIR, f"{label}.vasp")

    with Trajectory(traj_path, "w", atoms) as traj:

        dyn = BFGS(
            atoms,
            trajectory=traj,
            logfile=log_path
        )

        dyn.run(fmax=FMAX)

    displacement = np.linalg.norm(
        atoms.get_positions() - initial_positions,
        axis=1
    ).max()

    total_energy = atoms.get_potential_energy()

    mace_energy = mace_calc.get_potential_energy(atoms)
    d4_energy = d4_calc.get_potential_energy(atoms)

    max_force = np.linalg.norm(
        atoms.get_forces(),
        axis=1
    ).max()

    write(vasp_path, atoms, format="vasp", direct=True)

    print(f"\n{label}")
    print(f"Optimization steps: {dyn.nsteps}")
    print(f"Combined energy:    {total_energy:.8f} eV")
    print(f"MACE energy:        {mace_energy:.8f} eV")
    print(f"D4 energy:          {d4_energy:.8f} eV")
    print(f"Maximum force:      {max_force:.8f} eV/A")
    print(f"Maximum movement:   {displacement:.6f} A")

    return {
        "energy": total_energy,
        "mace_energy": mace_energy,
        "d4_energy": d4_energy,
        "steps": dyn.nsteps,
        "max_force": max_force,
        "max_displacement": displacement
    }


# ============================================================
# BUILD Au(100) SLAB
# ============================================================

print("\nBuilding Au(100) slab...")

slab = fcc100(
    "Au",
    size=(5, 5, 6),
    vacuum=12.0
)

zmin = min(atom.position[2] for atom in slab)

fixed_mask = [
    atom.position[2] < zmin + 0.1
    for atom in slab
]

slab.set_constraint(FixAtoms(mask=fixed_mask))

print("Au atoms:", len(slab))
print("Fixed Au atoms:", sum(fixed_mask))
print("Cell:", slab.cell.lengths())
print("PBC:", slab.pbc)


# ============================================================
# CLEAN SLAB REFERENCE
# ============================================================

clean_result = optimize(slab, "clean_Au100")


# ============================================================
# ISOLATED Hg REFERENCE
# ============================================================

print("\nBuilding isolated Hg reference...")

hg = Atoms("Hg", pbc=True)
hg.set_cell([15.0, 15.0, 15.0])
hg.center()

hg_result = optimize(hg, "isolated_Hg")


# ============================================================
# ADSORPTION CALCULATIONS
# ============================================================

results = {}

for site in SITES:

    print("\n" + "=" * 55)
    print("Starting adsorption calculation:", site)
    print("=" * 55)

    atoms = slab.copy()

    add_adsorbate(
        atoms,
        "Hg",
        height=2.5,
        position=site
    )

    result = optimize(
        atoms,
        f"Hg_Au100_{site}"
    )

    adsorption_energy = (
        result["energy"]
        - clean_result["energy"]
        - hg_result["energy"]
    )

    result["adsorption_energy"] = adsorption_energy

    results[site] = result


# ============================================================
# SAVE RESULTS
# ============================================================

csv_path = os.path.join(
    OUTDIR,
    "adsorption_energy_results.csv"
)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Site",
        "Combined energy (eV)",
        "MACE energy (eV)",
        "D4 energy (eV)",
        "Adsorption energy (eV)",
        "Optimization steps",
        "Maximum force (eV/A)",
        "Maximum displacement (A)"
    ])

    for site, result in results.items():

        writer.writerow([
            site,
            result["energy"],
            result["mace_energy"],
            result["d4_energy"],
            result["adsorption_energy"],
            result["steps"],
            result["max_force"],
            result["max_displacement"]
        ])


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MACE-MP + D4 ADSORPTION RESULTS")
print("=" * 65)

print(f"Clean slab energy: {clean_result['energy']:.8f} eV")
print(f"Isolated Hg energy: {hg_result['energy']:.8f} eV")

for site, result in results.items():

    print(
        f"{site:10s} "
        f"E_ads = {result['adsorption_energy']:+.8f} eV"
    )

best_site = min(
    results,
    key=lambda site: results[site]["adsorption_energy"]
)

print("\nMost stable adsorption site:", best_site)

print("\nResults saved in:", OUTDIR)
print("CSV file:", csv_path)
print("=" * 65)

