
import csv
import os

# --------------------------------------------------
# Input files
# --------------------------------------------------
input_files = [
    "convergence/outputs/adsorption_4x4x4_results.csv",
    "convergence/outputs/adsorption_4x4x6_results.csv",
    "outputs/adsorption_energy_results.csv",
    "convergence/outputs/adsorption_5x5x6_vac16_results.csv",
]

output_file = "analysis/tables/ALIGNN_master_dataset.csv"

# --------------------------------------------------
# Read and combine datasets
# --------------------------------------------------
all_results = []

for filename in input_files:

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing input file: {filename}"
        )

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    if len(rows) != 3:
        raise ValueError(
            f"{filename}: expected 3 adsorption sites, "
            f"found {len(rows)}"
        )

    for row in rows:

        if row["converged"].lower() != "true":
            raise RuntimeError(
                f"Non-converged calculation found: "
                f"{filename}, {row['site']}"
            )

        row["model"] = "ALIGNN"

        if "4x4x4" in filename:
            row["configuration"] = "4x4x4_vac12"

        elif "4x4x6" in filename:
            row["configuration"] = "4x4x6_vac12"

        elif "vac16" in filename:
            row["configuration"] = "5x5x6_vac16"

        else:
            row["configuration"] = "5x5x6_vac12"

        if not row.get("vacuum_A"):
            row["vacuum_A"] = "12.0"

        all_results.append(row)

# --------------------------------------------------
# Calculate relative adsorption energies
# --------------------------------------------------
configurations = {}

for row in all_results:

    configuration = row["configuration"]

    configurations.setdefault(
        configuration, []
    ).append(row)

for configuration, rows in configurations.items():

    minimum_energy = min(
        float(row["E_ads_eV"])
        for row in rows
    )

    for row in rows:

        row["relative_E_ads_eV"] = (
            float(row["E_ads_eV"]) - minimum_energy
        )

# --------------------------------------------------
# Sort results
# --------------------------------------------------
site_order = {
    "ontop": 1,
    "bridge": 2,
    "hollow": 3
}

configuration_order = {
    "4x4x4_vac12": 1,
    "4x4x6_vac12": 2,
    "5x5x6_vac12": 3,
    "5x5x6_vac16": 4
}

all_results.sort(
    key=lambda row: (
        configuration_order[row["configuration"]],
        site_order[row["site"]]
    )
)

# --------------------------------------------------
# Output columns
# --------------------------------------------------
fieldnames = [
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
    "Hg_z_A"
]

# --------------------------------------------------
# Write master CSV
# --------------------------------------------------
with open(output_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in all_results:
        writer.writerow(row)

# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("\n===================================")
print("ALIGNN MASTER DATASET")
print("===================================")

print("Total calculations:", len(all_results))
print("Configurations:", len(configurations))

for configuration, rows in configurations.items():

    best = min(
        rows,
        key=lambda row: float(row["E_ads_eV"])
    )

    print(
        configuration,
        "| Best site:", best["site"],
        "| E_ads:", best["E_ads_eV"], "eV"
    )

print("\nAll calculations converged:",
      all(row["converged"].lower() == "true"
          for row in all_results))

print("\nSaved:", output_file)

