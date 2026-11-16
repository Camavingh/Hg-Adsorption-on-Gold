
import os
import csv
import matplotlib.pyplot as plt

BASE = "convergence/consistent_geometry"
OUTPUT = f"{BASE}/analysis"

os.makedirs(f"{OUTPUT}/tables", exist_ok=True)
os.makedirs(f"{OUTPUT}/figures", exist_ok=True)

FILES = {
    "4x4x4_12A": f"{BASE}/outputs/adsorption_4x4x4_results.csv",
    "4x4x6_12A": f"{BASE}/outputs/adsorption_4x4x6_results.csv",
    "5x5x6_12A": f"{BASE}/outputs/adsorption_5x5x6_results.csv",
    "5x5x6_16A": f"{BASE}/outputs/adsorption_5x5x6_vac16_results.csv",
}

SITES = ["ontop", "bridge", "hollow"]

# Read all existing results
data = {}

for system, filename in FILES.items():
    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    data[system] = {}

    for row in rows:
        site = row["site"].lower()

        data[system][site] = {
            "E_ads": float(row["E_ads_eV"]),
            "converged": row["converged"]
        }

# Calculate relative energies within each system
relative = {}

for system, values in data.items():
    minimum = min(values[site]["E_ads"] for site in SITES)

    relative[system] = {
        site: values[site]["E_ads"] - minimum
        for site in SITES
    }

# Write consolidated CSV
combined_file = f"{OUTPUT}/tables/nequip_convergence_combined.csv"

with open(combined_file, "w", newline="") as f:
    fields = [
        "system",
        "site",
        "E_ads_eV",
        "relative_E_ads_eV",
        "converged"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for system in FILES:
        for site in SITES:
            writer.writerow({
                "system": system,
                "site": site,
                "E_ads_eV": data[system][site]["E_ads"],
                "relative_E_ads_eV": relative[system][site],
                "converged": data[system][site]["converged"]
            })

# Print summary
print("\nNEQUIP CONVERGENCE SUMMARY")
print("-" * 75)

for system in FILES:
    print(f"\n{system}")

    for site in SITES:
        print(
            f"{site:8s}: "
            f"{data[system][site]['E_ads']: .6f} eV | "
            f"Relative: {relative[system][site]:.6f} eV"
        )

# Figure 1: Absolute adsorption energy vs slab system
systems = list(FILES.keys())
x = list(range(len(systems)))

plt.figure(figsize=(9, 5.5))

for site in SITES:
    values = [data[s][site]["E_ads"] for s in systems]
    plt.plot(x, values, marker="o", linewidth=2, label=site.capitalize())

plt.xticks(x, systems, rotation=15)
plt.ylabel("Adsorption energy (eV)")
plt.xlabel("Slab system")
plt.title("NequIP: Absolute adsorption energy convergence")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{OUTPUT}/figures/01_absolute_energy_convergence.png",
    dpi=300
)
plt.savefig(f"{OUTPUT}/figures/01_absolute_energy_convergence.pdf")
plt.close()

# Figure 2: Relative adsorption energies
plt.figure(figsize=(9, 5.5))

for site in SITES:
    values = [relative[s][site] for s in systems]
    plt.plot(x, values, marker="o", linewidth=2, label=site.capitalize())

plt.xticks(x, systems, rotation=15)
plt.ylabel("Relative adsorption energy (eV)")
plt.xlabel("Slab system")
plt.title("NequIP: Relative adsorption energy convergence")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{OUTPUT}/figures/02_relative_energy_convergence.png",
    dpi=300
)
plt.savefig(f"{OUTPUT}/figures/02_relative_energy_convergence.pdf")
plt.close()

# Figure 3: Vacuum convergence at 5x5x6
vacuum_systems = ["5x5x6_12A", "5x5x6_16A"]
xv = [12, 16]

plt.figure(figsize=(7, 5))

for site in SITES:
    values = [
        data[s][site]["E_ads"]
        for s in vacuum_systems
    ]
    plt.plot(xv, values, marker="o", linewidth=2, label=site.capitalize())

plt.xticks(xv)
plt.xlabel("Vacuum thickness (Å)")
plt.ylabel("Adsorption energy (eV)")
plt.title("NequIP: Vacuum convergence (5×5×6 slab)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{OUTPUT}/figures/03_vacuum_convergence.png",
    dpi=300
)
plt.savefig(f"{OUTPUT}/figures/03_vacuum_convergence.pdf")
plt.close()

# Figure 4: Energy differences between 12 and 16 Å
plt.figure(figsize=(7, 5))

differences = [
    (
        data["5x5x6_16A"][site]["E_ads"]
        - data["5x5x6_12A"][site]["E_ads"]
    ) * 1000
    for site in SITES
]

plt.bar(SITES, differences)
plt.axhline(0, linewidth=0.8)
plt.ylabel("E(16 Å) − E(12 Å) (meV)")
plt.xlabel("Adsorption site")
plt.title("NequIP: Vacuum-induced energy differences")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT}/figures/04_vacuum_energy_difference.png",
    dpi=300
)
plt.savefig(f"{OUTPUT}/figures/04_vacuum_energy_difference.pdf")
plt.close()

print("\nSaved combined CSV:", combined_file)
print("Saved figures in:", f"{OUTPUT}/figures")
print("\nAnalysis completed.")

