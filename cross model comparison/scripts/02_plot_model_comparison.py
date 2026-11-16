#!/usr/bin/env python3

"""
Cross-model comparison of Hg adsorption energies on Au(100).

Input:
    data/model_comparison.csv

Outputs:
    figures/absolute_adsorption_energies.png
    figures/relative_adsorption_energies.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "data" / "model_comparison.csv"
FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required_columns = {
    "model",
    "site",
    "E_ads_eV",
    "relative_energy_eV",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

models = ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]
sites = ["ontop", "bridge", "hollow"]

for model in models:
    model_data = df[df["model"] == model]

    if set(model_data["site"]) != set(sites):
        raise ValueError(
            f"{model} does not contain exactly ontop, bridge and hollow."
        )


# --------------------------------------------------
# Plot 1: Absolute adsorption energies
# --------------------------------------------------

x = np.arange(len(sites))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):

    model_data = (
        df[df["model"] == model]
        .set_index("site")
        .reindex(sites)
    )

    offset = (i - 1) * width

    bars = ax.bar(
        x + offset,
        model_data["E_ads_eV"],
        width,
        label=model,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

ax.set_xlabel("Adsorption site", fontsize=12)
ax.set_ylabel("Adsorption energy (eV)", fontsize=12)

ax.set_title(
    "Hg Adsorption Energies on Au(100): Model Comparison",
    fontsize=14,
)

ax.set_xticks(x)
ax.set_xticklabels(sites)

ax.axhline(0, linewidth=0.8)
ax.legend(title="Interatomic potential")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_file = FIGURE_DIR / "absolute_adsorption_energies.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_file}")


# --------------------------------------------------
# Plot 2: Relative adsorption energies
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):

    model_data = (
        df[df["model"] == model]
        .set_index("site")
        .reindex(sites)
    )

    offset = (i - 1) * width

    bars = ax.bar(
        x + offset,
        model_data["relative_energy_eV"],
        width,
        label=model,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

ax.set_xlabel("Adsorption site", fontsize=12)
ax.set_ylabel("Relative adsorption energy (eV)", fontsize=12)

ax.set_title(
    "Relative Hg Adsorption Energies on Au(100): Model Comparison",
    fontsize=14,
)

ax.set_xticks(x)
ax.set_xticklabels(sites)

ax.set_ylim(bottom=0)
ax.legend(title="Interatomic potential")
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_file = FIGURE_DIR / "relative_adsorption_energies.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_file}")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPreferred adsorption site predicted by each model:")

for model in models:

    model_data = df[df["model"] == model]

    preferred = model_data.loc[
        model_data["relative_energy_eV"].idxmin()
    ]

    print(
        f"{model:12s} : {preferred['site']:8s} "
        f"({preferred['E_ads_eV']:.6f} eV)"
    )

print("\nPlot generation completed successfully.")

