
import csv
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MACE_FILE = (
    PROJECT_ROOT / "ML-IP_mace" / "adsorption_energy_results.csv"
)

ALIGNN_FILE = (
    Path.home() / "work/projects/Hg-Au100-ALIGNN"
    / "outputs/adsorption_energy_results.csv"
)

NEQUIP_FILE = (
    PROJECT_ROOT / "ML-IP_nequip"
    / "outputs/adsorption_energy_results.csv"
)

OUTPUT = (
    PROJECT_ROOT / "cross_model_comparison"
    / "data" / "model_comparison.csv"
)

SITES = ["ontop", "bridge", "hollow"]

# --------------------------------------------------
# Read MACE results: select 5x5x6 configuration
# --------------------------------------------------

results = []

with open(MACE_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

selected = [
    row for row in rows
    if row["Calculation"] == "5x5x6"
]

if len(selected) != 1:
    raise ValueError("Expected exactly one MACE 5x5x6 record.")

row = selected[0]

for site, column in [
    ("ontop", "Ontop_eV"),
    ("bridge", "Bridge_eV"),
    ("hollow", "Hollow_eV")
]:
    results.append({
        "model": "MACE-MP",
        "site": site,
        "E_ads_eV": float(row[column])
    })

# --------------------------------------------------
# Read ALIGNN results
# --------------------------------------------------

def read_site_results(path, model):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 3:
        raise ValueError(
            f"{model}: expected 3 adsorption-site records."
        )

    model_results = []

    for row in rows:
        if row["converged"].strip().lower() != "true":
            raise ValueError(
                f"{model}: {row['site']} is not converged."
            )

        model_results.append({
            "model": model,
            "site": row["site"].strip().lower(),
            "E_ads_eV": float(row["E_ads_eV"])
        })

    if {r["site"] for r in model_results} != set(SITES):
        raise ValueError(f"{model}: unexpected adsorption sites.")

    return model_results


results.extend(read_site_results(ALIGNN_FILE, "ALIGNN-FF"))

# --------------------------------------------------
# Read NequIP results
# --------------------------------------------------

results.extend(read_site_results(NEQUIP_FILE, "NequIP-OAM-M"))

# --------------------------------------------------
# Calculate relative energies within each model
# --------------------------------------------------

for model in ["MACE-MP", "ALIGNN-FF", "NequIP-OAM-M"]:
    model_rows = [
        r for r in results if r["model"] == model
    ]

    minimum = min(r["E_ads_eV"] for r in model_rows)

    for row in model_rows:
        row["relative_energy_eV"] = (
            row["E_ads_eV"] - minimum
        )

# --------------------------------------------------
# Validate and export
# --------------------------------------------------

if len(results) != 9:
    raise ValueError("Expected exactly 9 comparison records.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "site",
            "E_ads_eV",
            "relative_energy_eV"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("Combined model comparison dataset saved:")
print(OUTPUT)

print("\n{:<14} {:<10} {:>14} {:>18}".format(
    "Model", "Site", "E_ads (eV)", "Relative (eV)"
))

for row in results:
    print(
        "{:<14} {:<10} {:>14.6f} {:>18.6f}".format(
            row["model"],
            row["site"],
            row["E_ads_eV"],
            row["relative_energy_eV"]
        )
    )

print("\nTotal records:", len(results))

