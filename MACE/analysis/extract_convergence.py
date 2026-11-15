from ase.io import read
import numpy as np
import csv

files = {
    "Ontop": "HgAu100_ontop.traj",
    "Bridge": "HgAu100_bridge.traj",
    "Hollow": "HgAu100_hollow.traj"
}

with open("optimization_convergence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Site", "Step", "Max_Force_eV_per_A", "Energy_eV"])

    for site, filename in files.items():
        frames = read(filename, index=":")

        for step, atoms in enumerate(frames):
            forces = atoms.get_forces()
            max_force = np.linalg.norm(forces, axis=1).max()
            energy = atoms.get_potential_energy()

            writer.writerow([
                site, step, max_force, energy
            ])

print("Export completed: optimization_convergence.csv")
