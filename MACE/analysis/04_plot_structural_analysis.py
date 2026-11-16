
#!/usr/bin/env python3

"""
Plot structural properties of Hg adsorption on Au(100)
using MACE-MP structural analysis results.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = Path("analysis/tables/MACE_structural_analysis.csv")
FIGURE_DIR = Path("analysis/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["Ontop", "Bridge", "Hollow"]

SITE_COLORS = {
    "Ontop": "tab:blue",
    "Bridge": "tab:orange",
    "Hollow": "tab:green",
}


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    data = {row["Site"]: row for row in rows}

    for site in SITE_ORDER:
        if site not in data:
            raise ValueError(f"Missing structural data for {site}")

    return data


def save_figure(fig, name):

    fig.savefig(
        FIGURE_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight"
    )

    print(f"Saved {name}.png and {name}.pdf")
    plt.close(fig)


def plot_adsorption_height(data):

    fig, ax = plt.subplots(figsize=(7, 5))

    values = [
        float(data[site]["Adsorption_Height_A"])
        for site in SITE_ORDER
    ]

    bars = ax.bar(
        SITE_ORDER,
        values,
        color=[SITE_COLORS[s] for s in SITE_ORDER],
        width=0.6
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    ax.set_ylabel("Hg adsorption height (Å)")
    ax.set_xlabel("Adsorption site")
    ax.set_title("MACE-MP: Hg adsorption height on Au(100)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "MACE_adsorption_height")


def plot_distance_coordination(data):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(SITE_ORDER))

    distances = [
        float(data[s]["Nearest_Hg_Au_Distance_A"])
        for s in SITE_ORDER
    ]

    coordination = [
        int(data[s]["Coordination_Number_3.5A"])
        for s in SITE_ORDER
    ]

    bars = ax1.bar(
        x - 0.18,
        distances,
        width=0.36,
        color=[SITE_COLORS[s] for s in SITE_ORDER],
        label="Nearest Hg–Au distance"
    )

    ax1.set_ylabel("Nearest Hg–Au distance (Å)")
    ax1.set_xlabel("Adsorption site")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SITE_ORDER)
    ax1.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, distances):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

    ax2 = ax1.twinx()

    ax2.plot(
        x + 0.18,
        coordination,
        marker="D",
        linestyle="--",
        linewidth=2,
        markersize=7,
        color="black",
        label="Coordination number"
    )

    ax2.set_ylabel("Coordination number (3.5 Å cutoff)")
    ax2.set_ylim(0, max(coordination) + 2)
    ax2.set_yticks(range(0, max(coordination) + 2))

    for xi, value in zip(x, coordination):
        ax2.annotate(
            str(value),
            (xi + 0.18, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        frameon=False
    )

    ax1.set_title("MACE-MP: Hg–Au distance and coordination")

    fig.tight_layout()
    save_figure(fig, "MACE_Hg_Au_distance_coordination")


def plot_substrate_displacement(data):

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(SITE_ORDER))
    width = 0.34

    rms = [
        float(data[s]["Au_RMS_Displacement_A"])
        for s in SITE_ORDER
    ]

    maximum = [
        float(data[s]["Au_Max_Displacement_A"])
        for s in SITE_ORDER
    ]

    bars1 = ax.bar(
        x - width / 2,
        rms,
        width,
        label="Au RMS displacement"
    )

    bars2 = ax.bar(
        x + width / 2,
        maximum,
        width,
        label="Au maximum displacement"
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(SITE_ORDER)
    ax.set_xlabel("Adsorption site")
    ax.set_ylabel("Au displacement (Å)")
    ax.set_title("MACE-MP: Au substrate displacement")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "MACE_Au_substrate_displacement")


def main():

    print("=" * 60)
    print("MACE-MP STRUCTURAL PLOTTING")
    print("=" * 60)

    data = load_data()

    plot_adsorption_height(data)
    plot_distance_coordination(data)
    plot_substrate_displacement(data)

    print("\nAll structural figures generated successfully.")


if __name__ == "__main__":
    main()

