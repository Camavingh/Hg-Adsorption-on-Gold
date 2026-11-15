
import csv
import os
import matplotlib.pyplot as plt

os.makedirs("analysis/figures", exist_ok=True)

# --------------------------------------------------
# Read structural data
# --------------------------------------------------
with open(
    "analysis/tables/ALIGNN_structural_analysis.csv",
    "r",
    newline=""
) as f:
    structural = list(csv.DictReader(f))

with open(
    "analysis/tables/ALIGNN_layer_displacements.csv",
    "r",
    newline=""
) as f:
    layer_data = list(csv.DictReader(f))

for row in structural:
    for key in [
        "nearest_Hg_Au_A",
        "adsorption_height_A",
        "rms_Au_displacement_A"
    ]:
        row[key] = float(row[key])

for row in layer_data:
    for key in [
        "layer",
        "mean_displacement_A",
        "rms_displacement_A",
        "max_displacement_A"
    ]:
        row[key] = float(row[key])

# --------------------------------------------------
# Configuration labels
# --------------------------------------------------
configurations = [
    "4x4x4_vac12",
    "4x4x6_vac12",
    "5x5x6_vac12",
    "5x5x6_vac16"
]

labels = [
    "4x4x4\n12 Å",
    "4x4x6\n12 Å",
    "5x5x6\n12 Å",
    "5x5x6\n16 Å"
]

sites = ["ontop", "bridge", "hollow"]


def get_values(column, site):

    values = []

    for config in configurations:

        matches = [
            row for row in structural
            if row["configuration"] == config
            and row["site"] == site
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Missing or duplicate result: {config}, {site}"
            )

        values.append(matches[0][column])

    return values


# --------------------------------------------------
# Figure 1: Nearest Hg-Au distance
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for site in sites:

    plt.plot(
        range(4),
        get_values("nearest_Hg_Au_A", site),
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(range(4), labels)
plt.xlabel("Slab configuration")
plt.ylabel("Nearest Hg–Au distance (Å)")
plt.title("ALIGNN: Nearest Hg–Au distance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "analysis/figures/ALIGNN_Hg_Au_distance.png",
    dpi=300
)
plt.savefig(
    "analysis/figures/ALIGNN_Hg_Au_distance.pdf"
)
plt.close()


# --------------------------------------------------
# Figure 2: Adsorption height
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for site in sites:

    plt.plot(
        range(4),
        get_values("adsorption_height_A", site),
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(range(4), labels)
plt.xlabel("Slab configuration")
plt.ylabel("Adsorption height (Å)")
plt.title("ALIGNN: Hg adsorption height")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "analysis/figures/ALIGNN_adsorption_height.png",
    dpi=300
)
plt.savefig(
    "analysis/figures/ALIGNN_adsorption_height.pdf"
)
plt.close()


# --------------------------------------------------
# Figure 3: Au substrate RMS displacement
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for site in sites:

    plt.plot(
        range(4),
        get_values("rms_Au_displacement_A", site),
        marker="o",
        linewidth=2,
        markersize=7,
        label=site.capitalize()
    )

plt.xticks(range(4), labels)
plt.xlabel("Slab configuration")
plt.ylabel("Au RMS displacement (Å)")
plt.title("ALIGNN: Substrate relaxation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "analysis/figures/ALIGNN_Au_RMS_displacement.png",
    dpi=300
)
plt.savefig(
    "analysis/figures/ALIGNN_Au_RMS_displacement.pdf"
)
plt.close()


# --------------------------------------------------
# Figure 4: Layer-resolved displacement profiles
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 6),
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

        layers = [
            row["layer"] for row in rows
        ]

        displacements = [
            row["mean_displacement_A"] for row in rows
        ]

        ax.plot(
            displacements,
            layers,
            marker="o",
            linewidth=2,
            label=config
        )

    ax.set_title(site.capitalize())
    ax.set_xlabel("Mean Au displacement (Å)")
    ax.set_ylabel("Au layer (bottom to top)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)

fig.suptitle(
    "ALIGNN: Layer-resolved substrate relaxation"
)

plt.tight_layout()

plt.savefig(
    "analysis/figures/ALIGNN_layer_displacement_profiles.png",
    dpi=300
)
plt.savefig(
    "analysis/figures/ALIGNN_layer_displacement_profiles.pdf"
)
plt.close()


print("\nALIGNN structural plots generated.")

for filename in sorted(os.listdir("analysis/figures")):
    if filename.startswith("ALIGNN_"):
        print(filename)

