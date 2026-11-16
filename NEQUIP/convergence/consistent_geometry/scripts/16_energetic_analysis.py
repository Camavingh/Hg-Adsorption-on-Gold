import csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
TABLES = BASE / "analysis" / "tables"

source = TABLES / "nequip_convergence_combined.csv"

with open(source, newline="") as f:
    records = list(csv.DictReader(f))

groups = defaultdict(dict)

for row in records:
    system = row["system"]
    site = row["site"]

    groups[system][site] = {
        "E_ads": float(row["E_ads_eV"]),
        "relative": float(row["relative_E_ads_eV"]),
        "converged": row["converged"].strip().lower() == "true",
    }

expected_sites = {"ontop", "bridge", "hollow"}

for system, sites in groups.items():
    if set(sites) != expected_sites:
        raise ValueError(f"Incomplete site data for {system}: {set(sites)}")

# ---------------------------------------------------------
# 1. Energetic convergence summary: one row per geometry
# ---------------------------------------------------------

summary = []

for system, sites in groups.items():
    preferred = min(sites, key=lambda s: sites[s]["E_ads"])

    summary.append({
        "system": system,
        "E_ontop_eV": sites["ontop"]["E_ads"],
        "E_bridge_eV": sites["bridge"]["E_ads"],
        "E_hollow_eV": sites["hollow"]["E_ads"],
        "preferred_site": preferred,
        "preferred_E_ads_eV": sites[preferred]["E_ads"],
        "ontop_gap_eV": sites["ontop"]["E_ads"] - sites[preferred]["E_ads"],
        "bridge_gap_eV": sites["bridge"]["E_ads"] - sites[preferred]["E_ads"],
        "hollow_gap_eV": sites["hollow"]["E_ads"] - sites[preferred]["E_ads"],
        "all_converged": all(x["converged"] for x in sites.values()),
    })

summary_fields = list(summary[0].keys())

with open(TABLES / "nequip_energetic_convergence_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary)

# ---------------------------------------------------------
# 2. Energetic site analysis: one row per geometry and site
# ---------------------------------------------------------

site_analysis = []

for system, sites in groups.items():
    ranked = sorted(sites, key=lambda s: sites[s]["E_ads"])
    ranks = {site: i + 1 for i, site in enumerate(ranked)}
    preferred = ranked[0]

    for site in ["ontop", "bridge", "hollow"]:
        site_analysis.append({
            "system": system,
            "site": site,
            "E_ads_eV": sites[site]["E_ads"],
            "relative_E_ads_eV": sites[site]["relative"],
            "energy_rank": ranks[site],
            "is_preferred_site": site == preferred,
            "converged": sites[site]["converged"],
        })

site_fields = list(site_analysis[0].keys())

with open(TABLES / "nequip_energetic_site_analysis.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=site_fields)
    writer.writeheader()
    writer.writerows(site_analysis)

# ---------------------------------------------------------
# 3. Site energy gaps: signed pairwise energy differences
# ---------------------------------------------------------

site_gaps = []

for system, sites in groups.items():
    e_on = sites["ontop"]["E_ads"]
    e_br = sites["bridge"]["E_ads"]
    e_hol = sites["hollow"]["E_ads"]

    site_gaps.append({
        "system": system,
        "ontop_minus_bridge_eV": e_on - e_br,
        "ontop_minus_hollow_eV": e_on - e_hol,
        "bridge_minus_hollow_eV": e_br - e_hol,
        "preferred_site": min(sites, key=lambda s: sites[s]["E_ads"]),
    })

gap_fields = list(site_gaps[0].keys())

with open(TABLES / "nequip_site_energy_gaps.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=gap_fields)
    writer.writeheader()
    writer.writerows(site_gaps)

print("Successfully generated:")
print(TABLES / "nequip_energetic_convergence_summary.csv")
print(TABLES / "nequip_energetic_site_analysis.csv")
print(TABLES / "nequip_site_energy_gaps.csv")
print("\nGeometry groups:", len(groups))
print("Site-analysis records:", len(site_analysis))
print("All calculations converged:",
      all(x["converged"] for sites in groups.values() for x in sites.values()))
