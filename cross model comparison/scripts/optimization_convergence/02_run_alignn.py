
import os
import csv
import numpy as np

from ase.io import read, write
from ase.build import add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INITIAL_SLAB = os.path.join(
    BASE_DIR,
    "data/common_geometry/Au100_5x5x6_vac12_initial.extxyz"
)

OUTDIR = os.path.join(
    BASE_DIR,
    "data/optimization_convergence/ALIGNN"
)

os.makedirs(OUTDIR, exist_ok=True)

DEVICE = "cuda"

HG_HEIGHT = 2.5
FMAX = 0.005
MAX_STEPS = 500

SITES = ["ontop", "bridge", "hollow"]


# ============================================================
# ALIGNN CALCULATOR
# ============================================================

calc = AlignnAtomwiseCalculator(
    path=default_path(),
    device=DEVICE
)


# ============================================================
# CONVERGENCE LOGGER
# ============================================================

class ConvergenceLogger:

    def __init__(self, atoms, site, movable_indices, csv_path):

        self.atoms = atoms
        self.site = site
        self.movable_indices = movable_indices
        self.csv_path = csv_path

        self.step = 0
        self.initial_energy = None

        self.file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "Model",
            "Site",
            "Step",
            "Energy_eV",
            "Delta_E_eV",
            "Max_Force_eV_per_A"
        ])

    def __call__(self):

        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()

        movable_forces = forces[self.movable_indices]

        max_force = np.linalg.norm(
            movable_forces,
            axis=1
        ).max()

        if self.initial_energy is None:
            self.initial_energy = energy

        delta_energy = energy - self.initial_energy

        self.writer.writerow([
            "ALIGNN-FF",
            self.site,
            self.step,
            f"{energy:.10f}",
            f"{delta_energy:.10f}",
            f"{max_force:.10f}"
        ])

        self.file.flush()

        print(
            f"{self.site:8s} | "
            f"Step {self.step:4d} | "
            f"E = {energy: .8f} eV | "
            f"dE = {delta_energy: .8f} eV | "
            f"Fmax = {max_force:.6f} eV/A"
        )

        self.step += 1

    def close(self):
        self.file.close()


# ============================================================
# ADSORPTION SITE COORDINATES
# ============================================================

def get_adsorption_position(slab, site):

    top_z = slab.positions[:, 2].max()

    top_indices = np.where(
        np.isclose(
            slab.positions[:, 2],
            top_z,
            atol=0.01
        )
    )[0]

    top_atoms = slab.positions[top_indices]

    ref = top_atoms[
        np.argmin(top_atoms[:, 0] + top_atoms[:, 1])
    ]

    a = np.linalg.norm(slab.cell[0])

    if site == "ontop":
        position = (ref[0], ref[1])

    elif site == "bridge":
        position = (ref[0] + a / 2, ref[1])

    elif site == "hollow":
        position = (ref[0] + a / 2, ref[1] + a / 2)

    else:
        raise ValueError(f"Unknown site: {site}")

    return position


# ============================================================
# OPTIMIZATION FUNCTION
# ============================================================

def optimize_site(site):

    print("\n" + "=" * 65)
    print(f"Starting ALIGNN optimization: {site.upper()}")
    print("=" * 65)

    # Load common initial slab
    slab = read(INITIAL_SLAB)

    # Freeze bottom Au layer
    zmin = slab.positions[:, 2].min()

    fixed_indices = np.where(
        slab.positions[:, 2] < zmin + 0.1
    )[0]

    slab.set_constraint(
        FixAtoms(indices=fixed_indices)
    )

    print("Au atoms:", len(slab))
    print("Fixed Au atoms:", len(fixed_indices))

    # Determine adsorption coordinates
    adsorption_position = get_adsorption_position(
        slab,
        site
    )

    print("Adsorption position:", adsorption_position)

    # Add Hg
    add_adsorbate(
        slab,
        "Hg",
        height=HG_HEIGHT,
        position=adsorption_position
    )

    # Assign ALIGNN calculator
    slab.calc = calc

    # Movable atom indices
    fixed_set = set(fixed_indices)

    movable_indices = np.array([
        i for i in range(len(slab))
        if i not in fixed_set
    ], dtype=int)

    # Output paths
    prefix = os.path.join(
        OUTDIR,
        f"ALIGNN_{site}"
    )

    trajectory_path = prefix + ".traj"
    logfile_path = prefix + ".log"
    csv_path = prefix + "_convergence.csv"
    final_structure_path = prefix + "_final.extxyz"

    logger = ConvergenceLogger(
        slab,
        site,
        movable_indices,
        csv_path
    )

    try:

        with Trajectory(
            trajectory_path,
            "w",
            slab
        ) as trajectory:

            optimizer = BFGS(
                slab,
                trajectory=trajectory,
                logfile=logfile_path
            )

            optimizer.attach(
                logger,
                interval=1
            )

            # Initial configuration, step 0
            logger()

            optimizer.run(
                fmax=FMAX,
                steps=MAX_STEPS
            )

    finally:
        logger.close()

    # Save final geometry
    write(
        final_structure_path,
        slab
    )

    # Final energy and force
    final_energy = slab.get_potential_energy()

    final_forces = slab.get_forces()[movable_indices]

    final_fmax = np.linalg.norm(
        final_forces,
        axis=1
    ).max()

    converged = final_fmax <= FMAX

    print("\nOptimization completed:", site)
    print("BFGS steps:", optimizer.nsteps)
    print("Final energy:", final_energy, "eV")
    print("Final Fmax:", final_fmax, "eV/A")
    print("Converged:", converged)

    print("CSV:", csv_path)
    print("Trajectory:", trajectory_path)
    print("Final structure:", final_structure_path)

    return {
        "site": site,
        "steps": optimizer.nsteps,
        "energy": final_energy,
        "fmax": final_fmax,
        "converged": converged
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(INITIAL_SLAB):
        raise FileNotFoundError(
            f"Initial slab not found: {INITIAL_SLAB}"
        )

    results = []

    for site in SITES:

        result = optimize_site(site)
        results.append(result)

    print("\n" + "=" * 75)
    print("ALIGNN OPTIMIZATION SUMMARY")
    print("=" * 75)

    for result in results:

        print(
            f"{result['site']:8s} | "
            f"Steps: {result['steps']:4d} | "
            f"Energy: {result['energy']: .8f} eV | "
            f"Fmax: {result['fmax']:.6f} eV/A | "
            f"Converged: {result['converged']}"
        )

    print("\nAll ALIGNN optimizations finished.")

