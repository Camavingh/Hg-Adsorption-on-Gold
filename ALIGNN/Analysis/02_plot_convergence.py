
import csv
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Files and settings
# --------------------------------------------------
INPUT = "analysis/tables/ALIGNN_master_dataset.csv"
OUTDIR = "analysis/figures"

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# Read master dataset
# --------------------------------------------------
with open(INPUT, "r", newline="") as f:
    data = list(csv.DictReader(f))

for row in data:
    for key in [
        "E_ads_eV",
        "relative_E_ads_eV",
        "vacuum_A"
    ]:
        row[key] = float(row[key])

configurations = [
    "4x4x4_vac12",
    "4x4x6_vac12",
    "5x5x6_vac12",
    "5x5x6_vac16"
]

sites = ["ontop", "bridge", "hollow"]

labels = [
    "4x4x4\n12 Å",
    "4x4x6\n12 Å",
    "5x5x6\n12 Å",
    "5x5x6\n16 Å"
]

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def get_value(configuration, site, column):

    matches = [
        row for row in data
        if row["configuration"] == configuration
        and row["site"] == site
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one result for {configuration}, {site}"
        )

    return matches[0][column]


# --------------------------------------------------
# Figure 1: Absolute adsorption energies
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for site in sites:

    values = [
        get_value(config, site, "E_ads_eV")
        for config in configurations
    ]

    plt.plot(
        range(len(configurations)),
        values,
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(range(4), labels)
plt.ylabel("Adsorption energy (eV)")
plt.xlabel("Slab configuration")
plt.title("ALIGNN: Hg adsorption energy convergence")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTDIR}/ALIGNN_adsorption_energy_convergence.png",
    dpi=300
)

plt.savefig(
    f"{OUTDIR}/ALIGNN_adsorption_energy_convergence.pdf"
)

plt.close()


# --------------------------------------------------
# Figure 2: Relative adsorption energies
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for site in sites:

    values = [
        get_value(config, site, "relative_E_ads_eV")
        for config in configurations
    ]

    plt.plot(
        range(len(configurations)),
        values,
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(range(4), labels)
plt.ylabel("Relative adsorption energy (eV)")
plt.xlabel("Slab configuration")
plt.title("ALIGNN: Relative stability of adsorption sites")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTDIR}/ALIGNN_relative_adsorption_energy.png",
    dpi=300
)

plt.savefig(
    f"{OUTDIR}/ALIGNN_relative_adsorption_energy.pdf"
)

plt.close()


# --------------------------------------------------
# Figure 3: Vacuum convergence
# --------------------------------------------------
vacuum_sites = ["ontop", "bridge", "hollow"]
vacuum_values = [12.0, 16.0]

plt.figure(figsize=(8, 6))

for site in vacuum_sites:

    values = []

    for vacuum in vacuum_values:

        configuration = (
            "5x5x6_vac12" if vacuum == 12.0
            else "5x5x6_vac16"
        )

        matches = [
            row for row in data
            if row["configuration"] == configuration
            and row["site"] == site
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Missing or duplicate vacuum result: "
                f"{site}, {vacuum} Å"
            )

        values.append(matches[0]["E_ads_eV"])

    plt.plot(
        vacuum_values,
        values,
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(vacuum_values)
plt.xlabel("Vacuum thickness (Å)")
plt.ylabel("Adsorption energy (eV)")
plt.title("ALIGNN: Vacuum thickness convergence")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTDIR}/ALIGNN_vacuum_convergence.png",
    dpi=300
)

plt.savefig(
    f"{OUTDIR}/ALIGNN_vacuum_convergence.pdf"
)

plt.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\nALIGNN convergence plots generated.")

for filename in sorted(os.listdir(OUTDIR)):
    if filename.startswith("ALIGNN_"):
        print(filename)

