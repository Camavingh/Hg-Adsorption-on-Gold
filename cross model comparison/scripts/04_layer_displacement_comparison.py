
#!/usr/bin/env python3

"""Layer-wise Au displacement comparison for Hg adsorption on Au(100)."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
HG_ROOT = PROJECT_DIR.parent

MACE_FILE = (
    HG_ROOT / "ML-IP_mace/analysis/tables/"
    "MACE_layer_displacements.csv"
)

ALIGNN_FILE = (
    Path.home() / "work/projects/Hg-Au100-ALIGNN/"
    "analysis/tables/ALIGNN_layer_displacements.csv"
)

NEQUIP_FILE = (
    HG_ROOT / "ML-IP_nequip/convergence/consistent_geometry/"
    "analysis/tables/nequip_layer_displacements.csv"
)

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("Loading layer displacement datasets...")

mace = pd.read_csv(MACE_FILE)
alignn = pd.read_csv(ALIGNN_FILE)
nequip = pd.read_csv(NEQUIP_FILE)


# ==================================================
# MACE
# ==================================================

mace_data = pd.DataFrame({
    "Model": "MACE-MP",
    "Site": mace["Site"].str.lower(),
    "Layer": mace["Layer"],
    "RMS_Displacement_A": mace["RMS_Displacement_A"],
    "Maximum_Displacement_A": mace["Maximum_Displacement_A"],
})


# ==================================================
# ALIGNN: SELECT MATCHING CONFIGURATION
# ==================================================

alignn = alignn[
    alignn["configuration"] == "5x5x6_vac12"
].copy()

alignn_data = pd.DataFrame({
    "Model": "ALIGNN-FF",
    "Site": alignn["site"].str.lower(),
    "Layer": alignn["layer"],
    "RMS_Displacement_A": alignn["rms_displacement_A"],
    "Maximum_Displacement_A": alignn["max_displacement_A"],
})


# ==================================================
# NEQUIP
# ==================================================

nequip_data = pd.DataFrame({
    "Model": "NequIP-OAM-M",
    "Site": nequip["site"].str.lower(),
    "Layer": nequip["layer"],
    "RMS_Displacement_A": nequip["RMS_displacement_A"],
    "Maximum_Displacement_A": nequip["maximum_displacement_A"],
})


# ==================================================
# COMBINE AND VALIDATE
# ==================================================

df = pd.concat(
    [mace_data, alignn_data, nequip_data],
    ignore_index=True
)

models = ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]
sites = ["ontop", "bridge", "hollow"]
layers = [1, 2, 3, 4, 5, 6]

df["Layer"] = pd.to_numeric(df["Layer"], errors="raise")

if len(df) != 54:
    raise ValueError(f"Expected 54 records, found {len(df)}")

if df.duplicated(["Model", "Site", "Layer"]).any():
    raise ValueError("Duplicate model/site/layer records detected.")

expected = {
    (model, site, layer)
    for model in models
    for site in sites
    for layer in layers
}

actual = set(
    zip(df["Model"], df["Site"], df["Layer"])
)

if actual != expected:
    raise ValueError(
        f"Missing or unexpected records: {expected.symmetric_difference(actual)}"
    )

if df[["RMS_Displacement_A", "Maximum_Displacement_A"]].isnull().any().any():
    raise ValueError("Missing displacement values detected.")

df = df.sort_values(
    ["Site", "Layer", "Model"]
).reset_index(drop=True)


# ==================================================
# SAVE COMBINED DATASET
# ==================================================

csv_file = DATA_DIR / "layer_displacement_comparison.csv"

df.to_csv(csv_file, index=False)

print(f"\nCombined dataset saved:\n{csv_file}")
print(f"Total records: {len(df)}")


# ==================================================
# GENERATE LAYER-WISE PLOTS
# ==================================================

metrics = [
    (
        "RMS_Displacement_A",
        "RMS displacement (A)",
        "RMS"
    ),
    (
        "Maximum_Displacement_A",
        "Maximum displacement (A)",
        "Maximum"
    ),
]

for site in sites:

    for column, ylabel, metric_name in metrics:

        fig, ax = plt.subplots(figsize=(9, 6))

        for model in models:

            subset = df[
                (df["Model"] == model) &
                (df["Site"] == site)
            ].sort_values("Layer")

            ax.plot(
                subset["Layer"],
                subset[column],
                marker="o",
                linewidth=2,
                markersize=7,
                label=model,
            )

        ax.set_xlabel("Au layer number", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        ax.set_title(
            f"{metric_name} Layer Displacement: {site.capitalize()} Hg Adsorption",
            fontsize=13,
        )

        ax.set_xticks(layers)
        ax.legend(title="Interatomic potential")
        ax.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()

        filename = (
            f"layer_{metric_name.lower()}_displacement_{site}"
        )

        png_file = FIGURE_DIR / f"{filename}.png"
        pdf_file = FIGURE_DIR / f"{filename}.pdf"

        fig.savefig(png_file, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_file, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved: {png_file}")
        print(f"Saved: {pdf_file}")


# ==================================================
# SUMMARY
# ==================================================

print("\nLAYER DISPLACEMENT SUMMARY")
print("=" * 75)

for site in sites:

    print(f"\n{site.upper()}")

    subset = df[df["Site"] == site]

    summary = subset.groupby("Model").agg(
        Maximum_RMS_A=("RMS_Displacement_A", "max"),
        Maximum_Au_Displacement_A=("Maximum_Displacement_A", "max"),
    )

    print(summary.round(6).to_string())

print("\n" + "=" * 75)
print("LAYER DISPLACEMENT COMPARISON COMPLETED.")
print("Records: 54")
print("Plots: 6 PNG + 6 PDF")

