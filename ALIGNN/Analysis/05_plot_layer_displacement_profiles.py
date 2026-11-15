
import csv
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_FILE = "analysis/tables/ALIGNN_layer_displacements.csv"
OUTPUT_DIR = "analysis/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load structural data
# --------------------------------------------------

with open(DATA_FILE, "r", newline="") as f:
    layer_data = list(csv.DictReader(f))

for row in layer_data:
    for key in [
        "layer",
        "mean_displacement_A",
        "rms_displacement_A",
        "max_displacement_A"
    ]:
        row[key] = float(row[key])

# --------------------------------------------------
# Configurations and adsorption sites
# --------------------------------------------------

configurations = [
    "4x4x4_vac12",
    "4x4x6_vac12",
    "5x5x6_vac12",
    "5x5x6_vac16"
]

sites = ["ontop", "bridge", "hollow"]

# Different line styles and markers help distinguish
# configurations, especially overlapping vacuum tests.

styles = {
    "4x4x4_vac12": {
        "linestyle": "-",
        "marker": "o",
        "label": "4x4x4, 12 Å"
    },
    "4x4x6_vac12": {
        "linestyle": "--",
        "marker": "s",
        "label": "4x4x6, 12 Å"
    },
    "5x5x6_vac12": {
        "linestyle": "-.",
        "marker": "^",
        "label": "5x5x6, 12 Å"
    },
    "5x5x6_vac16": {
        "linestyle": ":",
        "marker": "D",
        "label": "5x5x6, 16 Å"
    }
}

# --------------------------------------------------
# Generate layer-resolved displacement figure
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 6),
    sharey=False
)

for ax, site in zip(axes, sites):

    for config in configurations:

        rows = [
            row for row in layer_data
            if row["configuration"] == config
            and row["site"] == site
        ]

        rows.sort(key=lambda row: row["layer"])

        if not rows:
            raise ValueError(
                f"No layer data found for {config}, {site}"
            )

        layers = [
            row["layer"] for row in rows
        ]

        displacements = [
            row["mean_displacement_A"] for row in rows
        ]

        style = styles[config]

        ax.plot(
            displacements,
            layers,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
            markerfacecolor="none"
            if config == "5x5x6_vac16" else None,
            label=style["label"]
        )

    ax.set_title(site.capitalize(), fontsize=13)
    ax.set_xlabel("Mean Au displacement (Å)", fontsize=11)
    ax.set_ylabel("Au layer (bottom to top)", fontsize=11)

    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(
        fontsize=8,
        loc="best",
        frameon=True
    )

fig.suptitle(
    "ALIGNN: Layer-resolved substrate relaxation",
    fontsize=15
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

# --------------------------------------------------
# Save PNG and PDF
# --------------------------------------------------

png_file = os.path.join(
    OUTPUT_DIR,
    "ALIGNN_layer_displacement_profiles_v2.png"
)

pdf_file = os.path.join(
    OUTPUT_DIR,
    "ALIGNN_layer_displacement_profiles_v2.pdf"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.close()

print("\nImproved ALIGNN layer-displacement figure generated.")
print(f"PNG: {png_file}")
print(f"PDF: {pdf_file}")

