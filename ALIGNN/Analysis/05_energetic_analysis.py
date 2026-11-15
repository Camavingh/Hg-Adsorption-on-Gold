
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "analysis" / "tables" / "ALIGNN_master_dataset.csv"
OUTPUT_DIR = BASE_DIR / "analysis" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "configuration",
    "site",
    "E_ads_eV",
    "converged",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["converged"] = (
    df["converged"].astype(str).str.strip().str.lower() == "true"
)

if not df["converged"].all():
    raise ValueError("At least one calculation is not converged.")

expected_sites = {"ontop", "bridge", "hollow"}

for config, group in df.groupby("configuration"):
    if set(group["site"]) != expected_sites or len(group) != 3:
        raise ValueError(
            f"{config}: expected exactly one result for each of "
            f"{sorted(expected_sites)}"
        )

# --------------------------------------------------
# 1. Site-wise energetic analysis
# --------------------------------------------------

site_rows = []

for config, group in df.groupby("configuration"):

    minimum_energy = group["E_ads_eV"].min()

    for _, row in group.iterrows():

        relative_energy = row["E_ads_eV"] - minimum_energy

        site_rows.append({
            "configuration": config,
            "site": row["site"],
            "E_ads_eV": row["E_ads_eV"],
            "E_ads_meV": row["E_ads_eV"] * 1000,
            "relative_E_ads_eV": relative_energy,
            "relative_E_ads_meV": relative_energy * 1000,
            "rank": (
                group["E_ads_eV"]
                .rank(method="min", ascending=True)
                .loc[row.name]
            ),
        })

site_df = pd.DataFrame(site_rows)

site_df["rank"] = site_df["rank"].astype(int)

site_df = site_df.sort_values(
    ["configuration", "rank"]
)

site_file = OUTPUT_DIR / "ALIGNN_energetic_site_analysis.csv"
site_df.to_csv(site_file, index=False)

# --------------------------------------------------
# 2. Energy separation between adsorption sites
# --------------------------------------------------

gap_rows = []

for config, group in df.groupby("configuration"):

    energies = group.set_index("site")["E_ads_eV"]

    ontop = energies["ontop"]
    bridge = energies["bridge"]
    hollow = energies["hollow"]

    gap_rows.append({
        "configuration": config,
        "bridge_minus_hollow_meV": (bridge - hollow) * 1000,
        "ontop_minus_bridge_meV": (ontop - bridge) * 1000,
        "ontop_minus_hollow_meV": (ontop - hollow) * 1000,
        "most_stable_site": energies.idxmin(),
    })

gap_df = pd.DataFrame(gap_rows)

gap_file = OUTPUT_DIR / "ALIGNN_site_energy_gaps.csv"
gap_df.to_csv(gap_file, index=False)

# --------------------------------------------------
# 3. Convergence variation across configurations
# --------------------------------------------------

convergence_rows = []

for site, group in df.groupby("site"):

    energies = group["E_ads_eV"]

    convergence_rows.append({
        "site": site,
        "number_of_configurations": len(group),
        "mean_E_ads_eV": energies.mean(),
        "minimum_E_ads_eV": energies.min(),
        "maximum_E_ads_eV": energies.max(),
        "energy_range_meV": (energies.max() - energies.min()) * 1000,
        "standard_deviation_meV": energies.std(ddof=1) * 1000,
    })

convergence_df = pd.DataFrame(convergence_rows)

convergence_file = (
    OUTPUT_DIR / "ALIGNN_energetic_convergence_summary.csv"
)
convergence_df.to_csv(convergence_file, index=False)

# --------------------------------------------------
# 4. Print results
# --------------------------------------------------

print("\nALIGNN SITE ENERGETIC ANALYSIS")
print("=" * 75)

print("\nAdsorption energies and relative energies:")
print(
    site_df[
        [
            "configuration",
            "site",
            "E_ads_eV",
            "relative_E_ads_meV",
            "rank",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print("\nEnergy separation between sites (meV):")
print(gap_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nVariation across configurations:")
print(
    convergence_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved files:")
print(site_file)
print(gap_file)
print(convergence_file)

print("\nEnergetic analysis completed successfully.")

