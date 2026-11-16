
#!/usr/bin/env python3

"""
Layer-resolved Au displacement profiles for Hg adsorption on Au(100)
using MACE-MP.

Input:
    analysis/tables/MACE_layer_displacements.csv

Outputs:
    MACE_layer_RMS_displacement.png/.pdf
    MACE_layer_max_displacement.png/.pdf
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("analysis/tables/MACE_layer_displacements.csv")
FIGURE_DIR = Path("analysis/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["Ontop", "Bridge", "Hollow"]

SITE_COLORS = {
    "Ontop": "tab:blue",
    "Bridge": "tab:orange",
    "Hollow": "tab:green",
}

SITE_MARKERS = {
    "Ontop": "o",
    "Bridge": "s",
    "Hollow": "^",
}


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Layer displacement CSV is empty.")

    return rows


def save_figure(fig, filename):

    fig.savefig(
        FIGURE_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURE_DIR / f"{filename}.pdf",
        bbox_inches="tight"
    )

    print(f"Saved {filename}.png and {filename}.pdf")
    plt.close(fig)


def plot_layer_profile(data, column, ylabel, title, filename):

    fig, ax = plt.subplots(figsize=(8, 5))

    for site in SITE_ORDER:

        site_rows = [
            row for row in data
            if row["Site"] == site
        ]

        site_rows.sort(key=lambda row: int(row["Layer"]))

        layers = [
            int(row["Layer"]) for row in site_rows
        ]

        values = [
            float(row[column]) for row in site_rows
        ]

        ax.plot(
            layers,
            values,
            marker=SITE_MARKERS[site],
            markersize=7,
            linewidth=2,
            color=SITE_COLORS[site],
            label=site
        )

    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Au atomic layer (bottom to top)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax.set_xlim(0.8, 6.2)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_figure(fig, filename)


def main():

    print("=" * 60)
    print("MACE-MP LAYER-RESOLVED DISPLACEMENT ANALYSIS")
    print("=" * 60)

    data = load_data()

    print(f"Layer records loaded: {len(data)}")

    plot_layer_profile(
        data,
        column="RMS_Displacement_A",
        ylabel="RMS Au displacement (Å)",
        title="MACE-MP: Layer-resolved RMS displacement",
        filename="MACE_layer_RMS_displacement"
    )

    plot_layer_profile(
        data,
        column="Maximum_Displacement_A",
        ylabel="Maximum Au displacement (Å)",
        title="MACE-MP: Layer-resolved maximum displacement",
        filename="MACE_layer_max_displacement"
    )

    print("\nLayer displacement figures generated successfully.")


if __name__ == "__main__":
    main()

