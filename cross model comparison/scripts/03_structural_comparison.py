#!/usr/bin/env python3

"""
Structural comparison of Hg adsorption on Au(100).

Models:
    MACE-MP
    ALIGNN-FF
    NequIP-OAM-M

Uses matching 5x5x6 slab geometries and 12 A vacuum
where applicable.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
HG_ROOT = PROJECT_DIR.parent

MACE_FILE = (
    HG_ROOT / "ML-IP_mace" / "analysis" /
    "tables" / "MACE_structural_analysis.csv"
)

ALIGNN_FILE = (
    Path.home() / "work/projects/Hg-Au100-ALIGNN" /
    "analysis" / "tables" / "ALIGNN_structural_analysis.csv"
)

NEQUIP_FILE = (
    HG_ROOT / "ML-IP_nequip" / "convergence" /
    "consistent_geometry" / "analysis" / "tables" /
    "nequip_structural_summary.csv"
)

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading structural datasets...")

mace = pd.read_csv(MACE_FILE)
alignn = pd.read_csv(ALIGNN_FILE)
nequip = pd.read_csv(NEQUIP_FILE)


# ==================================================
# NORMALIZE MACE
# ==================================================

mace["Site"] = mace["Site"].str.lower()

mace_data = pd.DataFrame({
    "Model": "MACE-MP",
    "Configuration": "5x5x6_vac12",
    "Site": mace["Site"],
    "Adsorption_Height_A": mace["Adsorption_Height_A"],
    "Nearest_Hg_Au_Distance_A":
        mace["Nearest_Hg_Au_Distance_A"],
    "Coordination_Number":
        mace["Coordination_Number_3.5A"],
    "Au_RMS_Displacement_A":
        mace["Au_RMS_Displacement_A"],
    "Au_Max_Displacement_A":
        mace["Au_Max_Displacement_A"],
})


# ==================================================
# NORMALIZE ALIGNN
# ==================================================

alignn = alignn[
    alignn["configuration"] == "5x5x6_vac12"
].copy()

alignn_data = pd.DataFrame({
    "Model": "ALIGNN-FF",
    "Configuration": alignn["configuration"],
    "Site": alignn["site"].str.lower(),
    "Adsorption_Height_A": alignn["adsorption_height_A"],
    "Nearest_Hg_Au_Distance_A": alignn["nearest_Hg_Au_A"],
    "Coordination_Number": alignn["coordination_number"],
    "Au_RMS_Displacement_A": alignn["rms_Au_displacement_A"],
    "Au_Max_Displacement_A": alignn["max_Au_displacement_A"],
})


# ==================================================
# NORMALIZE NEQUIP
# ==================================================

nequip_data = pd.DataFrame({
    "Model": "NequIP-OAM-M",
    "Configuration": "5x5x6_vac12",
    "Site": nequip["site"].str.lower(),
    "Adsorption_Height_A": nequip["adsorption_height_A"],
    "Nearest_Hg_Au_Distance_A":
        nequip["nearest_Hg_Au_distance_A"],
    "Coordination_Number":
        nequip["coordination_number_3.5A"],
    "Au_RMS_Displacement_A":
        nequip["Au_RMS_displacement_A"],
    "Au_Max_Displacement_A":
        nequip["Au_maximum_displacement_A"],
})


# ==================================================
# COMBINE DATA
# ==================================================

df = pd.concat(
    [mace_data, alignn_data, nequip_data],
    ignore_index=True
)

models = ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]
sites = ["ontop", "bridge", "hollow"]

if len(df) != 9:
    raise ValueError(f"Expected 9 records, found {len(df)}")

if df.duplicated(["Model", "Site"]).any():
    raise ValueError("Duplicate model/site records found.")

if set(df["Site"]) != set(sites):
    raise ValueError("Unexpected adsorption-site names.")

if set(df["Model"]) != set(models):
    raise ValueError("Unexpected model names.")

if df.isnull().any().any():
    raise ValueError("Missing values detected in structural dataset.")

# Save combined structural dataset
csv_file = DATA_DIR / "structural_model_comparison.csv"
df.to_csv(csv_file, index=False)

print(f"\nCombined structural dataset saved:\n{csv_file}")


# ==================================================
# PLOTTING FUNCTION
# ==================================================

plot_parameters = [
    (
        "Adsorption_Height_A",
        "Adsorption height (A)",
        "Hg adsorption height on Au(100)",
        "adsorption_height_comparison"
    ),
    (
        "Nearest_Hg_Au_Distance_A",
        "Nearest Hg-Au distance (A)",
        "Nearest Hg-Au distance comparison",
        "nearest_Hg_Au_distance_comparison"
    ),
    (
        "Coordination_Number",
        "Coordination number",
        "Hg-Au coordination number comparison",
        "coordination_number_comparison"
    ),
    (
        "Au_RMS_Displacement_A",
        "Au RMS displacement (A)",
        "Au substrate RMS displacement",
        "Au_RMS_displacement_comparison"
    ),
    (
        "Au_Max_Displacement_A",
        "Maximum Au displacement (A)",
        "Maximum Au substrate displacement",
        "Au_max_displacement_comparison"
    ),
]


# ==================================================
# GENERATE FIVE PLOTS
# ==================================================

x = np.arange(len(sites))
width = 0.25

for column, ylabel, title, filename in plot_parameters:

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):

        subset = (
            df[df["Model"] == model]
            .set_index("Site")
            .reindex(sites)
        )

        offset = (i - 1) * width

        bars = ax.bar(
            x + offset,
            subset[column],
            width,
            label=model
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8
        )

    ax.set_xlabel("Adsorption site", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(sites)

    ax.legend(title="Interatomic potential")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    png_file = FIGURE_DIR / f"{filename}.png"
    pdf_file = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png_file}")
    print(f"Saved: {pdf_file}")


# ==================================================
# NUMERICAL COMPARISON
# ==================================================

print("\nSTRUCTURAL COMPARISON SUMMARY")
print("=" * 100)

columns = [
    "Adsorption_Height_A",
    "Nearest_Hg_Au_Distance_A",
    "Coordination_Number",
    "Au_RMS_Displacement_A",
    "Au_Max_Displacement_A",
]

for site in sites:

    print(f"\nSITE: {site.upper()}")
    print("-" * 100)

    subset = df[df["Site"] == site].set_index("Model")

    print(
        subset[columns].round(5).to_string()
    )

print("\n" + "=" * 100)
print("STRUCTURAL COMPARISON COMPLETED SUCCESSFULLY.")
print(f"Records: {len(df)}")
print("Plots generated: 5 PNG + 5 PDF")

