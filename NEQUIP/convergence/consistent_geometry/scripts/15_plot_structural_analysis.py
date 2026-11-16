
import os
import csv
import matplotlib.pyplot as plt

BASE = "convergence/consistent_geometry/analysis"
TABLES = f"{BASE}/tables"
FIGURES = f"{BASE}/figures"

os.makedirs(FIGURES, exist_ok=True)

# Read structural summary
with open(f"{TABLES}/nequip_structural_summary.csv", newline="") as f:
    summary = list(csv.DictReader(f))

# Read layer-resolved displacements
with open(f"{TABLES}/nequip_layer_displacements.csv", newline="") as f:
    layers = list(csv.DictReader(f))

sites = ["ontop", "bridge", "hollow"]
site_labels = ["Ontop", "Bridge", "Hollow"]

summary_by_site = {
    row["site"]: row for row in summary
}

# Helper for bar plots
def make_bar_plot(key, ylabel, title, filename, scale=1.0):
    values = [
        float(summary_by_site[site][key]) * scale
        for site in sites
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(site_labels, values)
    plt.ylabel(ylabel)
    plt.xlabel("Adsorption site")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output = FIGURES + "/" + filename
    plt.savefig(output, dpi=300)
    plt.savefig(output.replace(".png", ".pdf"))
    plt.close()

# 1. Adsorption height
make_bar_plot(
    "adsorption_height_A",
    "Adsorption height (Å)",
    "NequIP: Hg adsorption height",
    "05_adsorption_height.png"
)

# 2. Nearest Hg-Au distance
make_bar_plot(
    "nearest_Hg_Au_distance_A",
    "Nearest Hg–Au distance (Å)",
    "NequIP: Nearest Hg–Au distance",
    "06_nearest_Hg_Au_distance.png"
)

# 3. Coordination number
make_bar_plot(
    "coordination_number_3.5A",
    "Coordination number",
    "NequIP: Hg coordination number",
    "07_coordination_number.png"
)

# 4. Au RMS displacement
make_bar_plot(
    "Au_RMS_displacement_A",
    "Au RMS displacement (Å)",
    "NequIP: Substrate RMS displacement",
    "08_Au_RMS_displacement.png"
)

# 5. Maximum Au displacement
make_bar_plot(
    "Au_maximum_displacement_A",
    "Maximum Au displacement (Å)",
    "NequIP: Maximum substrate displacement",
    "09_Au_maximum_displacement.png"
)

# 6. Layer-resolved Au displacement
plt.figure(figsize=(8, 5.5))

for site in sites:
    rows = [
        row for row in layers
        if row["site"] == site
    ]

    rows.sort(key=lambda row: int(row["layer"]))

    layer_numbers = [
        int(row["layer"]) for row in rows
    ]

    rms_values = [
        float(row["RMS_displacement_A"]) * 1000
        for row in rows
    ]

    plt.plot(
        layer_numbers,
        rms_values,
        marker="o",
        linewidth=2,
        label=site.capitalize()
    )

plt.xlabel("Au layer (1 = bottom, 6 = top)")
plt.ylabel("Layer RMS displacement (mÅ)")
plt.title("NequIP: Layer-resolved substrate displacement")
plt.xticks(range(1, 7))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{FIGURES}/10_layer_resolved_displacement.png",
    dpi=300
)
plt.savefig(
    f"{FIGURES}/10_layer_resolved_displacement.pdf"
)
plt.close()

print("\nStructural plots generated successfully.")

for filename in sorted(os.listdir(FIGURES)):
    if filename.endswith(".png"):
        print(filename)

print("\nFigures saved in:", FIGURES)

