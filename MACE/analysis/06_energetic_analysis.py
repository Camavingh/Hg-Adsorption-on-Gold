
#!/usr/bin/env python3

"""
Energetic analysis of Hg adsorption on Au(100) using MACE-MP.

Generates:
1. Energetic site analysis
2. Relative site-energy gaps
3. Slab-size and vacuum convergence summary
"""

import csv
from pathlib import Path


INPUT_FILE = Path("analysis/tables/MACE_master_dataset.csv")
OUTPUT_DIR = Path("analysis/tables")

SITE_ORDER = ["Ontop", "Bridge", "Hollow"]

GEOMETRY_ORDER = [
    "4x4x4",
    "4x4x6",
    "5x5x6",
    "5x5x6_vacuum16",
]

REFERENCE_GEOMETRY = "5x5x6"


def load_master_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    data = {}

    for row in rows:

        geometry = row["Calculation"]
        site = row["Site"]

        if geometry not in GEOMETRY_ORDER:
            raise ValueError(f"Unknown geometry: {geometry}")

        if site not in SITE_ORDER:
            raise ValueError(f"Unknown adsorption site: {site}")

        data.setdefault(geometry, {})

        if site in data[geometry]:
            raise ValueError(
                f"Duplicate result: {geometry}, {site}"
            )

        data[geometry][site] = float(
            row["Adsorption_Energy_eV"]
        )

    for geometry in GEOMETRY_ORDER:

        if geometry not in data:
            raise ValueError(f"Missing geometry: {geometry}")

        for site in SITE_ORDER:
            if site not in data[geometry]:
                raise ValueError(
                    f"Missing result: {geometry}, {site}"
                )

    return data


def write_csv(filename, rows):

    if not rows:
        raise ValueError(f"No data to write: {filename}")

    path = OUTPUT_DIR / filename

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


# ------------------------------------------------------------
# 1. ENERGETIC SITE ANALYSIS
# ------------------------------------------------------------

def build_site_analysis(data):

    rows = []

    for geometry in GEOMETRY_ORDER:

        energies = data[geometry]

        preferred_site = min(
            energies,
            key=energies.get
        )

        rows.append({
            "Calculation": geometry,
            "Ontop_Energy_eV": energies["Ontop"],
            "Bridge_Energy_eV": energies["Bridge"],
            "Hollow_Energy_eV": energies["Hollow"],
            "Preferred_Site": preferred_site,
            "Minimum_Adsorption_Energy_eV": energies[preferred_site],
            "Ontop_minus_Hollow_eV":
                energies["Ontop"] - energies["Hollow"],
            "Bridge_minus_Hollow_eV":
                energies["Bridge"] - energies["Hollow"],
            "Ontop_minus_Bridge_eV":
                energies["Ontop"] - energies["Bridge"],
        })

    return rows


# ------------------------------------------------------------
# 2. RELATIVE SITE-ENERGY GAPS
# ------------------------------------------------------------

def build_site_energy_gaps(data):

    rows = []

    for geometry in GEOMETRY_ORDER:

        energies = data[geometry]
        minimum_energy = min(energies.values())

        for site in SITE_ORDER:

            rows.append({
                "Calculation": geometry,
                "Site": site,
                "Adsorption_Energy_eV": energies[site],
                "Minimum_Energy_eV": minimum_energy,
                "Relative_Energy_Gap_eV":
                    energies[site] - minimum_energy,
            })

    return rows


# ------------------------------------------------------------
# 3. ENERGETIC CONVERGENCE SUMMARY
# ------------------------------------------------------------

def build_convergence_summary(data):

    reference = data[REFERENCE_GEOMETRY]
    rows = []

    for geometry in GEOMETRY_ORDER:

        for site in SITE_ORDER:

            energy = data[geometry][site]
            reference_energy = reference[site]

            rows.append({
                "Calculation": geometry,
                "Site": site,
                "Adsorption_Energy_eV": energy,
                "Reference_Geometry": REFERENCE_GEOMETRY,
                "Reference_Energy_eV": reference_energy,
                "Difference_from_5x5x6_eV":
                    energy - reference_energy,
                "Absolute_Difference_eV":
                    abs(energy - reference_energy),
            })

    return rows


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MACE-MP ENERGETIC ANALYSIS")
    print("=" * 65)

    data = load_master_dataset()

    site_rows = build_site_analysis(data)
    gap_rows = build_site_energy_gaps(data)
    convergence_rows = build_convergence_summary(data)

    write_csv(
        "MACE_energetic_site_analysis.csv",
        site_rows
    )

    write_csv(
        "MACE_site_energy_gaps.csv",
        gap_rows
    )

    write_csv(
        "MACE_energetic_convergence_summary.csv",
        convergence_rows
    )

    print("\nPreferred adsorption sites:")

    for row in site_rows:
        print(
            f"{row['Calculation']:20s} "
            f"{row['Preferred_Site']:8s} "
            f"{row['Minimum_Adsorption_Energy_eV']:.6f} eV"
        )

    print("\nEnergy differences relative to 5x5x6:")

    for row in convergence_rows:
        print(
            f"{row['Calculation']:20s} "
            f"{row['Site']:8s} "
            f"{row['Difference_from_5x5x6_eV']:+.6f} eV"
        )

    print("\nEnergetic analysis completed successfully.")


if __name__ == "__main__":
    main()

