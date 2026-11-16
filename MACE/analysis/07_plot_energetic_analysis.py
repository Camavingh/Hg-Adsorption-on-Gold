
#!/usr/bin/env python3

"""
Relative adsorption energy analysis for Hg/Au(100) using MACE-MP.

Plots the relative energy of ontop, bridge and hollow sites
for all four slab geometries.

Reference for each geometry:
    Its own lowest-energy adsorption site.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = Path("analysis/tables/MACE_master_dataset.csv")
FIGURE_DIR = Path("analysis/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRIES = [
    "4x4x4",
    "4x4x6",
    "5x5x6",
    "5x5x6_vacuum16",
]

GEOMETRY_LABELS = [
    "4×4×4",
    "4×4×6",
    "5×5×6\n(12 Å)",
    "5×5×6\n(16 Å)",
]

SITES = ["Ontop", "Bridge", "Hollow"]

COLORS = {
    "Ontop": "tab:blue",
    "Bridge": "tab:orange",
    "Hollow": "tab:green",
}

MARKERS = {
    "Ontop": "o",
    "Bridge": "s",
    "Hollow": "^",
}


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    data = {}

    for row in rows:
        geometry = row["Calculation"]
        site = row["Site"]

        data.setdefault(geometry, {})
        data[geometry][site] = float(
            row["Adsorption_Energy_eV"]
        )

    for geometry in GEOMETRIES:
        if geometry not in data:
            raise ValueError(f"Missing geometry: {geometry}")

        for site in SITES:
            if site not in data[geometry]:
                raise ValueError(
                    f"Missing {site} result for {geometry}"
                )

    return data


def main():

    data = load_data()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(GEOMETRIES))

    for site in SITES:

        relative_energies = []

        for geometry in GEOMETRIES:

            energies = data[geometry]
            minimum = min(energies.values())

            relative_energies.append(
                energies[site] - minimum
            )

        ax.plot(
            x,
            relative_energies,
            marker=MARKERS[site],
            markersize=8,
            linewidth=2,
            color=COLORS[site],
            label=site
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_LABELS)
    ax.set_xlabel("Au(100) slab geometry")
    ax.set_ylabel("Relative adsorption energy (eV)")
    ax.set_title(
        "MACE-MP: Relative adsorption energies of Hg on Au(100)"
    )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax.set_ylim(bottom=0)

    fig.tight_layout()

    for extension in ["png", "pdf"]:
        output = FIGURE_DIR / (
            f"MACE_relative_adsorption_energy.{extension}"
        )

        if extension == "png":
            fig.savefig(
                output,
                dpi=300,
                bbox_inches="tight"
            )
        else:
            fig.savefig(
                output,
                bbox_inches="tight"
            )

        print(f"Saved: {output}")

    plt.close(fig)

    print("\nRelative adsorption energy plot completed.")


if __name__ == "__main__":
    main()

