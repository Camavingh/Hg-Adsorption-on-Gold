#!/usr/bin/env python3

"""Export cross-model comparison figures as PDF."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "model_comparison.csv"
FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = {
    "model",
    "site",
    "E_ads_eV",
    "relative_energy_eV",
}

if not required.issubset(df.columns):
    raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

models = ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]
sites = ["ontop", "bridge", "hollow"]

# Absolute adsorption energies
fig, ax = plt.subplots(figsize=(10, 6))

for model in models:
    subset = df[df["model"] == model].set_index("site").reindex(sites)
    ax.plot(
        sites,
        subset["E_ads_eV"],
        marker="o",
        linewidth=2,
        label=model,
    )

ax.set_xlabel("Adsorption site")
ax.set_ylabel("Adsorption energy (eV)")
ax.set_title("Hg Adsorption Energies on Au(100)")
ax.legend(title="Interatomic potential")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()

path = FIGURE_DIR / "absolute_adsorption_energies.pdf"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# Relative adsorption energies
fig, ax = plt.subplots(figsize=(10, 6))

for model in models:
    subset = df[df["model"] == model].set_index("site").reindex(sites)
    ax.plot(
        sites,
        subset["relative_energy_eV"],
        marker="o",
        linewidth=2,
        label=model,
    )

ax.set_xlabel("Adsorption site")
ax.set_ylabel("Relative adsorption energy (eV)")
ax.set_title("Relative Hg Adsorption Energies on Au(100)")
ax.legend(title="Interatomic potential")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_ylim(bottom=0)
fig.tight_layout()

path = FIGURE_DIR / "relative_adsorption_energies.pdf"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

print("\nPDF export completed successfully.")

