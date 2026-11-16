
#!/usr/bin/env python3

"""
Build the master adsorption-energy dataset for Hg adsorption on Au(100)
using the MACE-MP interatomic potential.

Input:
    outputs/adsorption_energy_results.csv

Output:
    analysis/tables/MACE_master_dataset.csv
"""

import csv
import os

INPUT_FILE = "outputs/adsorption_energy_results.csv"
OUTPUT_DIR = "analysis/tables"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "MACE_master_dataset.csv")

# Geometry metadata corresponding to the calculations in the input CSV.
GEOMETRY_INFO = {
    "4x4x4": {
        "Nx": 4,
        "Ny": 4,
        "Layers": 4,
        "Vacuum_A": 12.0,
        "Atoms_Au": 64,
    },
    "4x4x6": {
        "Nx": 4,
        "Ny": 4,
        "Layers": 6,
        "Vacuum_A": 12.0,
        "Atoms_Au": 96,
    },
    "5x5x6": {
        "Nx": 5,
        "Ny": 5,
        "Layers": 6,
        "Vacuum_A": 12.0,
        "Atoms_Au": 150,
    },
    "5x5x6_vacuum16": {
        "Nx": 5,
        "Ny": 5,
        "Layers": 6,
        "Vacuum_A": 16.0,
        "Atoms_Au": 150,
    },
}

SITE_COLUMNS = {
    "Ontop": "Ontop_eV",
    "Bridge": "Bridge_eV",
    "Hollow": "Hollow_eV",
}


def main():

    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_rows = []

    with open(INPUT_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"Calculation", *SITE_COLUMNS.values()}
        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")

        for record in reader:

            calculation = record["Calculation"]

            if calculation not in GEOMETRY_INFO:
                raise ValueError(
                    f"Unknown calculation geometry: {calculation}"
                )

            geometry = GEOMETRY_INFO[calculation]

            for site, column in SITE_COLUMNS.items():

                master_rows.append({
                    "Model": "MACE-MP",
                    "Surface": "Au(100)",
                    "Adsorbate": "Hg",
                    "Calculation": calculation,
                    "Nx": geometry["Nx"],
                    "Ny": geometry["Ny"],
                    "Layers": geometry["Layers"],
                    "Vacuum_A": geometry["Vacuum_A"],
                    "Atoms_Au": geometry["Atoms_Au"],
                    "Site": site,
                    "Adsorption_Energy_eV": float(record[column]),
                })

    fieldnames = list(master_rows[0].keys())

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_rows)

    print("=" * 60)
    print("MACE MASTER DATASET GENERATED")
    print("=" * 60)
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(master_rows)}")
    print(f"Geometries: {len(GEOMETRY_INFO)}")
    print(f"Adsorption sites: {len(SITE_COLUMNS)}")

    print("\nAdsorption energies:")
    for row in master_rows:
        print(
            f"{row['Calculation']:20s} "
            f"{row['Site']:8s} "
            f"{row['Adsorption_Energy_eV']: .6f} eV"
        )

    print("\nPreferred site for each geometry:")
    for calculation in GEOMETRY_INFO:
        subset = [
            row for row in master_rows
            if row["Calculation"] == calculation
        ]
        preferred = min(
            subset,
            key=lambda row: row["Adsorption_Energy_eV"]
        )
        print(
            f"{calculation:20s}: "
            f"{preferred['Site']} "
            f"({preferred['Adsorption_Energy_eV']:.6f} eV)"
        )

    print("\nDataset generation completed successfully.")


if __name__ == "__main__":
    main()

