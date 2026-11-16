
#!/usr/bin/env python3

"""Cross-model slab-size and vacuum convergence comparison."""

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
    "MACE_energetic_convergence_summary.csv"
)

ALIGNN_FILE = (
    Path.home() / "work/projects/Hg-Au100-ALIGNN/"
    "analysis/tables/ALIGNN_energetic_site_analysis.csv"
)

NEQUIP_FILE = (
    HG_ROOT / "ML-IP_nequip/convergence/consistent_geometry/"
    "analysis/tables/nequip_convergence_combined.csv"
)

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# CONFIGURATION MAPPING
# ==================================================

CONFIG_MAP = {
    "4x4x4": "4x4x4_12A",
    "4x4x6": "4x4x6_12A",
    "5x5x6": "5x5x6_12A",
    "5x5x6_vacuum16": "5x5x6_16A",
    "4x4x4_vac12": "4x4x4_12A",
    "4x4x6_vac12": "4x4x6_12A",
    "5x5x6_vac12": "5x5x6_12A",
    "5x5x6_vac16": "5x5x6_16A",
    "4x4x4_12A": "4x4x4_12A",
    "4x4x6_12A": "4x4x6_12A",
    "5x5x6_12A": "5x5x6_12A",
    "5x5x6_16A": "5x5x6_16A",
}

CONFIG_ORDER = [
    "4x4x4_12A",
    "4x4x6_12A",
    "5x5x6_12A",
    "5x5x6_16A",
]

SLAB_CONFIGS = [
    "4x4x4_12A",
    "4x4x6_12A",
    "5x5x6_12A",
]

SITES = ["ontop", "bridge", "hollow"]

MODELS = ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]


# ==================================================
# LOAD DATA
# ==================================================

print("Loading convergence datasets...")

mace = pd.read_csv(MACE_FILE)
alignn = pd.read_csv(ALIGNN_FILE)
nequip = pd.read_csv(NEQUIP_FILE)


# ==================================================
# NORMALIZE MACE
# ==================================================

mace["Site"] = mace["Site"].str.lower()
mace["Configuration"] = mace["Calculation"].map(CONFIG_MAP)

if mace["Configuration"].isnull().any():
    raise ValueError("Unknown MACE configuration found.")

mace_data = pd.DataFrame({
    "Model": "MACE-MP",
    "Configuration": mace["Configuration"],
    "Site": mace["Site"],
    "E_ads_eV": mace["Adsorption_Energy_eV"],
})


# ==================================================
# NORMALIZE ALIGNN
# ==================================================

alignn["Configuration"] = alignn["configuration"].map(CONFIG_MAP)

if alignn["Configuration"].isnull().any():
    raise ValueError("Unknown ALIGNN configuration found.")

alignn_data = pd.DataFrame({
    "Model": "ALIGNN-FF",
    "Configuration": alignn["Configuration"],
    "Site": alignn["site"].str.lower(),
    "E_ads_eV": alignn["E_ads_eV"],
})


# ==================================================
# NORMALIZE NEQUIP
# ==================================================

nequip["Configuration"] = nequip["system"].map(CONFIG_MAP)

if nequip["Configuration"].isnull().any():
    raise ValueError("Unknown NequIP configuration found.")

nequip_data = pd.DataFrame({
    "Model": "NequIP-OAM-M",
    "Configuration": nequip["Configuration"],
    "Site": nequip["site"].str.lower(),
    "E_ads_eV": nequip["E_ads_eV"],
})


# ==================================================
# COMBINE DATA
# ==================================================

df = pd.concat(
    [mace_data, alignn_data, nequip_data],
    ignore_index=True
)

if len(df) != 36:
    raise ValueError(
        f"Expected 36 records (3 models x 4 configurations x 3 sites), "
        f"found {len(df)}"
    )

if df.duplicated(["Model", "Configuration", "Site"]).any():
    raise ValueError("Duplicate model/configuration/site records found.")

expected = {
    (model, config, site)
    for model in MODELS
    for config in CONFIG_ORDER
    for site in SITES
}

actual = set(
    zip(df["Model"], df["Configuration"], df["Site"])
)

if actual != expected:
    raise ValueError(
        "Missing or unexpected records:\n"
        f"{expected.symmetric_difference(actual)}"
    )

df["Configuration"] = pd.Categorical(
    df["Configuration"],
    categories=CONFIG_ORDER,
    ordered=True
)

df = df.sort_values(
    ["Model", "Site", "Configuration"]
).reset_index(drop=True)


# ==================================================
# CALCULATE WITHIN-MODEL RELATIVE ENERGIES
# ==================================================

# Relative site energy within each model/configuration.
df["Relative_E_ads_eV"] = (
    df["E_ads_eV"]
    - df.groupby(
        ["Model", "Configuration"],
        observed=True
    )["E_ads_eV"].transform("min")
)

# Reference is 5x5x6 with 12 A vacuum.
reference = df[
    df["Configuration"] == "5x5x6_12A"
][["Model", "Site", "E_ads_eV"]].rename(
    columns={"E_ads_eV": "Reference_E_ads_eV"}
)

df = df.merge(
    reference,
    on=["Model", "Site"],
    how="left",
    validate="many_to_one"
)

df["Difference_from_reference_meV"] = (
    df["E_ads_eV"] - df["Reference_E_ads_eV"]
) * 1000

df["Absolute_difference_from_reference_meV"] = (
    df["Difference_from_reference_meV"].abs()
)


# ==================================================
# SAVE UNIFIED DATASET
# ==================================================

csv_file = DATA_DIR / "convergence_model_comparison.csv"

df.to_csv(csv_file, index=False)

print(f"\nUnified convergence dataset saved:\n{csv_file}")
print(f"Total records: {len(df)}")


# ==================================================
# PLOTTING FUNCTION
# ==================================================

def save_figure(fig, filename):

    png_file = FIGURE_DIR / f"{filename}.png"
    pdf_file = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png_file}")
    print(f"Saved: {pdf_file}")


# ==================================================
# PLOT 1: SLAB-SIZE CONVERGENCE
# ==================================================

for site in SITES:

    fig, ax = plt.subplots(figsize=(9, 6))

    for model in MODELS:

        subset = df[
            (df["Model"] == model)
            & (df["Site"] == site)
            & (df["Configuration"].isin(SLAB_CONFIGS))
        ].copy()

        subset["Configuration"] = subset["Configuration"].astype(str)

        subset = subset.set_index("Configuration").reindex(SLAB_CONFIGS)

        ax.plot(
            ["4x4x4", "4x4x6", "5x5x6"],
            subset["E_ads_eV"],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Slab configuration")
    ax.set_ylabel("Adsorption energy (eV)")
    ax.set_title(f"Slab-Size Convergence: {site.capitalize()} Hg Adsorption")
    ax.legend(title="Interatomic potential")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    save_figure(fig, f"slab_size_convergence_{site}")


# ==================================================
# PLOT 2: VACUUM CONVERGENCE
# ==================================================

for site in SITES:

    fig, ax = plt.subplots(figsize=(9, 6))

    for model in MODELS:

        subset = df[
            (df["Model"] == model)
            & (df["Site"] == site)
            & (df["Configuration"].isin(
                ["5x5x6_12A", "5x5x6_16A"]
            ))
        ].copy()

        subset["Configuration"] = subset["Configuration"].astype(str)

        subset = subset.set_index("Configuration").reindex(
            ["5x5x6_12A", "5x5x6_16A"]
        )

        ax.plot(
            ["12 A", "16 A"],
            subset["Difference_from_reference_meV"],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Vacuum thickness")
    ax.set_ylabel("Energy difference from 12 A reference (meV)")
    ax.set_title(f"Vacuum Convergence: {site.capitalize()} Hg Adsorption")
    ax.legend(title="Interatomic potential")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    save_figure(fig, f"vacuum_convergence_{site}")


# ==================================================
# NUMERICAL SUMMARY
# ==================================================

print("\nSLAB-SIZE CONVERGENCE")
print("=" * 85)

for model in MODELS:

    print(f"\n{model}")

    subset = df[
        (df["Model"] == model)
        & (df["Configuration"].isin(SLAB_CONFIGS))
    ]

    summary = subset.pivot(
        index="Configuration",
        columns="Site",
        values="E_ads_eV"
    ).reindex(SLAB_CONFIGS)

    print(summary.round(6).to_string())


print("\nVACUUM CONVERGENCE")
print("=" * 85)

vacuum = df[
    df["Configuration"].isin(["5x5x6_12A", "5x5x6_16A"])
]

summary = vacuum.pivot_table(
    index=["Model", "Site"],
    columns="Configuration",
    values="E_ads_eV"
)

summary["Difference_meV"] = (
    summary["5x5x6_16A"] - summary["5x5x6_12A"]
) * 1000

print(summary.round(6).to_string())

print("\n" + "=" * 85)
print("CONVERGENCE COMPARISON COMPLETED SUCCESSFULLY.")
print("Records: 36")
print("Figures: 6 PNG + 6 PDF")

