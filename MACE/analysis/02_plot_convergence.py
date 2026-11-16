
#!/usr/bin/env python3

"""
MACE-MP convergence analysis for Hg adsorption on Au(100).

Generates:
1. Slab-size convergence
2. Vacuum-thickness convergence
3. BFGS optimization convergence

Outputs:
    analysis/figures/
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------------------

MASTER_FILE = Path("analysis/tables/MACE_master_dataset.csv")
OPTIMIZATION_FILE = Path(
    "convergence/outputs/optimization_convergence.csv"
)

FIGURE_DIR = Path("analysis/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["Ontop", "Bridge", "Hollow"]

SITE_MARKERS = {
    "Ontop": "o",
    "Bridge": "s",
    "Hollow": "^",
}

SITE_COLORS = {
    "Ontop": "tab:blue",
    "Bridge": "tab:orange",
    "Hollow": "tab:green",
}


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_csv(filename):

    if not filename.exists():
        raise FileNotFoundError(f"Missing input file: {filename}")

    with open(filename, "r", newline="") as file:
        return list(csv.DictReader(file))


def save_figure(fig, filename):

    png = FIGURE_DIR / f"{filename}.png"
    pdf = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    plt.close(fig)


# ------------------------------------------------------------
# 1. SLAB-SIZE CONVERGENCE
# ------------------------------------------------------------

def plot_slab_convergence(data):

    selected = ["4x4x4", "4x4x6", "5x5x6"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(selected))

    for site in SITES:

        energies = []

        for geometry in selected:

            matches = [
                row for row in data
                if row["Calculation"] == geometry
                and row["Site"] == site
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one result for {geometry}, {site}"
                )

            energies.append(
                float(matches[0]["Adsorption_Energy_eV"])
            )

        ax.plot(
            x,
            energies,
            marker=SITE_MARKERS[site],
            linewidth=2,
            markersize=7,
            label=site,
            color=SITE_COLORS[site],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(selected)
    ax.set_xlabel("Au(100) slab geometry")
    ax.set_ylabel("Adsorption energy (eV)")
    ax.set_title("MACE-MP: Slab-size convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "MACE_slab_size_convergence")


# ------------------------------------------------------------
# 2. VACUUM-THICKNESS CONVERGENCE
# ------------------------------------------------------------

def plot_vacuum_convergence(data):

    geometries = ["5x5x6", "5x5x6_vacuum16"]
    vacuum_values = [12, 16]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.array(vacuum_values)

    for site in SITES:

        energies = []

        for geometry in geometries:

            matches = [
                row for row in data
                if row["Calculation"] == geometry
                and row["Site"] == site
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected one result for {geometry}, {site}"
                )

            energies.append(
                float(matches[0]["Adsorption_Energy_eV"])
            )

        ax.plot(
            x,
            energies,
            marker=SITE_MARKERS[site],
            linewidth=2,
            markersize=8,
            label=site,
            color=SITE_COLORS[site],
        )

    ax.set_xticks(vacuum_values)
    ax.set_xlabel("Vacuum thickness (Å)")
    ax.set_ylabel("Adsorption energy (eV)")
    ax.set_title("MACE-MP: Vacuum-thickness convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "MACE_vacuum_convergence")


# ------------------------------------------------------------
# 3. BFGS OPTIMIZATION CONVERGENCE
# ------------------------------------------------------------

def plot_optimization_convergence():

    data = load_csv(OPTIMIZATION_FILE)

    grouped = {site: [] for site in SITES}

    for row in data:

        site = row["Site"]

        if site not in grouped:
            continue

        grouped[site].append({
            "step": int(row["Step"]),
            "energy": float(row["Energy_eV"]),
            "force": float(row["Max_Force_eV_per_A"]),
        })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    for site in SITES:

        rows = sorted(grouped[site], key=lambda item: item["step"])

        if not rows:
            raise ValueError(
                f"No optimization data found for {site}"
            )

        steps = [row["step"] for row in rows]
        energies = [row["energy"] for row in rows]
        forces = [row["force"] for row in rows]

        # Energy change relative to the initial configuration.
        energy_change = [
            energy - energies[0] for energy in energies
        ]

        axes[0].plot(
            steps,
            energy_change,
            marker=SITE_MARKERS[site],
            markersize=4,
            linewidth=1.8,
            label=site,
            color=SITE_COLORS[site],
        )

        axes[1].semilogy(
            steps,
            forces,
            marker=SITE_MARKERS[site],
            markersize=4,
            linewidth=1.8,
            label=site,
            color=SITE_COLORS[site],
        )

    axes[0].set_xlabel("BFGS optimization step")
    axes[0].set_ylabel("Energy change from initial step (eV)")
    axes[0].set_title("Energy minimization")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].axhline(
        0.005,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Convergence threshold (0.005 eV/Å)",
    )

    axes[1].set_xlabel("BFGS optimization step")
    axes[1].set_ylabel("Maximum force (eV/Å)")
    axes[1].set_title("Force convergence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.suptitle(
        "MACE-MP: BFGS optimization convergence",
        fontsize=14
    )

    fig.tight_layout()
    save_figure(fig, "MACE_optimization_convergence")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("MACE-MP CONVERGENCE ANALYSIS")
    print("=" * 60)

    data = load_csv(MASTER_FILE)

    print(f"Master dataset records: {len(data)}")

    plot_slab_convergence(data)
    plot_vacuum_convergence(data)
    plot_optimization_convergence()

    print("\nAll convergence figures generated successfully.")


if __name__ == "__main__":
    main()

