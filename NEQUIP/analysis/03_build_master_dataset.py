
import csv
import os

OUTPUT = "analysis/tables/NequIP_master_dataset.csv"

SOURCES = [
    ("4x4x4_12A", "convergence/consistent_geometry/outputs/adsorption_4x4x4_results.csv"),
    ("4x4x6_12A", "convergence/consistent_geometry/outputs/adsorption_4x4x6_results.csv"),
    ("5x5x6_12A", "convergence/consistent_geometry/outputs/adsorption_5x5x6_results.csv"),
    ("5x5x6_16A", "convergence/consistent_geometry/outputs/adsorption_5x5x6_vac16_results.csv"),
]

FIELDS = [
    "model",
    "configuration",
    "slab",
    "vacuum_A",
    "site",
    "E_clean_eV",
    "E_Hg_eV",
    "E_total_eV",
    "E_ads_eV",
    "relative_E_ads_eV",
    "max_force_eV_A",
    "converged",
    "Hg_x_A",
    "Hg_y_A",
    "Hg_z_A",
]

rows = []

for configuration, filepath in SOURCES:
    with open(filepath, newline="") as f:
        source_rows = list(csv.DictReader(f))

    energies = [float(row["E_ads_eV"]) for row in source_rows]
    minimum_energy = min(energies)

    slab, vacuum = configuration.split("_")
    vacuum = float(vacuum.replace("A", ""))

    for row in source_rows:
        record = {
            "model": "NequIP-OAM-M",
            "configuration": configuration,
            "slab": slab,
            "vacuum_A": vacuum,
            "site": row["site"],
            "E_clean_eV": row["E_clean_eV"],
            "E_Hg_eV": row["E_Hg_eV"],
            "E_total_eV": row["E_total_eV"],
            "E_ads_eV": row["E_ads_eV"],
            "relative_E_ads_eV": (
                float(row["E_ads_eV"]) - minimum_energy
            ),
            "max_force_eV_A": row["max_force_eV_A"],
            "converged": row["converged"],
            "Hg_x_A": row.get("Hg_x_A", ""),
            "Hg_y_A": row.get("Hg_y_A", ""),
            "Hg_z_A": row.get("Hg_z_A", ""),
        }
        rows.append(record)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Master dataset saved: {OUTPUT}")
print(f"Total records: {len(rows)}")
print(f"Configurations: {len(SOURCES)}")
print("Expected records per configuration: 3")

