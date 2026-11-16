
import os
import csv
import numpy as np

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.build import add_adsorbate

from nequip.integrations.ase import NequIPCalculator


# ==================================================
# Parameters
# ==================================================

MODEL = os.path.expanduser(
    "~/work/NequIP-Hg-Au100/NequIP-OAM-M.nequip.pt2"
)

DEVICE = "cuda"

HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500

N_FIXED = 25

COMMON_SLAB = (
    "data/common_geometry/"
    "Au100_5x5x6_vac12_initial.extxyz"
)

OUTPUT_DIR = "data/optimization_convergence/NequIP"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# Load common initial Au(100) geometry
# ==================================================

slab = read(COMMON_SLAB)

if len(slab) != 150:
    raise ValueError(
        f"Expected 150 Au atoms, found {len(slab)}"
    )

top_z = slab.positions[:, 2].max()

top_indices = np.where(
    np.isclose(slab.positions[:, 2], top_z, atol=0.01)
)[0]

top_atoms = slab.positions[top_indices]

# Reference top-layer atom at minimum x + y
ref = top_atoms[
    np.argmin(top_atoms[:, 0] + top_atoms[:, 1])
]

# Surface lattice spacing
a_surface = np.linalg.norm(slab.cell[0]) / 5.0

sites = {
    "ontop": (ref[0], ref[1]),
    "bridge": (ref[0] + a_surface / 2, ref[1]),
    "hollow": (
        ref[0] + a_surface / 2,
        ref[1] + a_surface / 2
    )
}

print("Common slab:", COMMON_SLAB)
print("Number of Au atoms:", len(slab))
print("Reference atom:", ref)
print("Surface spacing:", a_surface)
print("Adsorption sites:", sites)


# ==================================================
# Load NequIP calculator
# ==================================================

print("\nLoading NequIP-OAM-M model...")

calc = NequIPCalculator.from_compiled_model(
    compile_path=MODEL,
    device=DEVICE,
    chemical_species_to_atom_type_map=True
)

print("NequIP calculator loaded successfully.")


# ==================================================
# Optimization for each adsorption site
# ==================================================

summary = []

for site_name, xy in sites.items():

    print("\n" + "=" * 55)
    print("Optimizing site:", site_name)
    print("=" * 55)

    ads = slab.copy()

    add_adsorbate(
        ads,
        "Hg",
        height=HG_HEIGHT,
        position=xy
    )

    ads.calc = calc

    ads.set_constraint(
        FixAtoms(indices=np.arange(N_FIXED))
    )

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"NequIP_{site_name}_convergence.csv"
    )

    traj_file = os.path.join(
        OUTPUT_DIR,
        f"NequIP_{site_name}.traj"
    )

    log_file = os.path.join(
        OUTPUT_DIR,
        f"NequIP_{site_name}.log"
    )

    final_file = os.path.join(
        OUTPUT_DIR,
        f"NequIP_{site_name}_final.extxyz"
    )

    # Movable atoms: all atoms except fixed bottom layer
    movable = np.ones(len(ads), dtype=bool)
    movable[:N_FIXED] = False

    # Initial energy
    initial_energy = ads.get_potential_energy()

    records = []

    def record_step(optimizer):
        step = optimizer.nsteps

        # Avoid duplicate initial-step entries
        if records and step <= records[-1]["step"]:
            return

        energy = ads.get_potential_energy()
        forces = ads.get_forces()

        max_force = np.linalg.norm(
            forces[movable], axis=1
        ).max()

        records.append({
            "step": step,
            "energy_eV": energy,
            "relative_energy_eV": energy - initial_energy,
            "max_force_eV_A": max_force
        })

    # Record initial structure exactly once
    forces = ads.get_forces()

    records.append({
        "step": 0,
        "energy_eV": initial_energy,
        "relative_energy_eV": 0.0,
        "max_force_eV_A": np.linalg.norm(
            forces[movable], axis=1
        ).max()
    })

    optimizer = BFGS(
        ads,
        trajectory=traj_file,
        logfile=log_file
    )

    optimizer.attach(
        record_step,
        interval=1,
        optimizer=optimizer
    )

    converged = optimizer.run(
        fmax=FMAX,
        steps=MAX_STEPS
    )

    # Final values
    final_energy = ads.get_potential_energy()
    final_forces = ads.get_forces()

    final_max_force = np.linalg.norm(
        final_forces[movable], axis=1
    ).max()

    hg_position = ads.positions[-1]

    # Save convergence data
    with open(csv_file, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "energy_eV",
                "relative_energy_eV",
                "max_force_eV_A"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    # Save final optimized structure
    write(final_file, ads)

    result = {
        "site": site_name,
        "steps": optimizer.nsteps,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "energy_change_eV": final_energy - initial_energy,
        "final_max_force_eV_A": final_max_force,
        "converged": converged,
        "Hg_x_A": hg_position[0],
        "Hg_y_A": hg_position[1],
        "Hg_z_A": hg_position[2]
    }

    summary.append(result)

    print("Steps:", optimizer.nsteps)
    print("Initial energy:", initial_energy, "eV")
    print("Final energy:", final_energy, "eV")
    print("Energy change:", final_energy - initial_energy, "eV")
    print("Final maximum force:", final_max_force, "eV/Å")
    print("Converged:", converged)
    print("Final Hg position:", hg_position)


# ==================================================
# Save summary
# ==================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "NequIP_optimization_summary.csv"
)

with open(summary_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


# ==================================================
# Final report
# ==================================================

print("\n" + "=" * 60)
print("NequIP-OAM-M OPTIMIZATION CONVERGENCE SUMMARY")
print("=" * 60)

for result in summary:

    print(
        f"{result['site']:8s} | "
        f"Steps: {result['steps']:3d} | "
        f"Energy: {result['final_energy_eV']:.8f} eV | "
        f"Fmax: {result['final_max_force_eV_A']:.6f} eV/Å | "
        f"Converged: {result['converged']}"
    )

print("\nSummary saved to:", summary_file)
print("All NequIP optimization runs completed.")

