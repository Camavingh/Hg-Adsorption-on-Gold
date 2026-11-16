
import os
import csv
import numpy as np

from ase.io import read

BASE = "convergence/consistent_geometry/outputs"
STRUCTURES = f"{BASE}/structures"
ANALYSIS = "convergence/consistent_geometry/analysis"

TABLES = f"{ANALYSIS}/tables"
os.makedirs(TABLES, exist_ok=True)

CUTOFF = 3.5

# Relaxed clean slab reference
clean = read(
    f"{STRUCTURES}/clean_Au100_5x5x6_NequIP.extxyz"
)

clean_positions = clean.positions.copy()
clean_symbols = np.array(clean.get_chemical_symbols())

au_indices = np.where(clean_symbols == "Au")[0]

clean_au = clean_positions[au_indices]

# Identify the six Au layers
layer_z = np.sort(
    np.unique(np.round(clean_au[:, 2], 3))
)

print("Number of Au atoms:", len(au_indices))
print("Number of layers:", len(layer_z))
print("Clean slab layer heights:", layer_z)

sites = ["ontop", "bridge", "hollow"]

results = []
distance_results = []
layer_results = []

for site in sites:

    print("\nAnalyzing:", site)

    filename = (
        f"{STRUCTURES}/Hg_Au100_5x5x6_{site}.extxyz"
    )

    ads = read(filename)

    positions = ads.positions
    symbols = np.array(ads.get_chemical_symbols())

    hg_index = np.where(symbols == "Hg")[0][0]
    hg_position = positions[hg_index]

    ads_au_indices = np.where(symbols == "Au")[0]
    ads_au_positions = positions[ads_au_indices]

    # 1. Adsorption height above the clean top Au layer
    top_z = clean_au[:, 2].max()
    adsorption_height = hg_position[2] - top_z

    # 2. Hg-Au distances
    distances = np.linalg.norm(
        ads_au_positions - hg_position,
        axis=1
    )

    sorted_distances = np.sort(distances)

    coordination = int(np.sum(distances <= CUTOFF))

    nearest_distances = sorted_distances[
        sorted_distances <= CUTOFF
    ]

    for i, distance in enumerate(sorted_distances, start=1):
        distance_results.append({
            "site": site,
            "Au_neighbor_rank": i,
            "Hg_Au_distance_A": distance
        })

    # 3. Au displacement relative to relaxed clean slab
    # Atom order is preserved between clean and adsorbed structures
    displacement = (
        ads.positions[au_indices] - clean_positions[au_indices]
    )

    displacement_magnitude = np.linalg.norm(
        displacement,
        axis=1
    )

    rms_displacement = np.sqrt(
        np.mean(displacement_magnitude**2)
    )

    max_displacement = displacement_magnitude.max()

    # 4. Layer-resolved displacement
    for layer_number, z_value in enumerate(layer_z, start=1):

        layer_mask = np.isclose(
            clean_au[:, 2],
            z_value,
            atol=0.01
        )

        layer_displacements = displacement_magnitude[layer_mask]

        layer_rms = np.sqrt(
            np.mean(layer_displacements**2)
        )

        layer_max = layer_displacements.max()

        layer_results.append({
            "site": site,
            "layer": layer_number,
            "clean_layer_z_A": z_value,
            "RMS_displacement_A": layer_rms,
            "maximum_displacement_A": layer_max
        })

    results.append({
        "site": site,
        "Hg_x_A": hg_position[0],
        "Hg_y_A": hg_position[1],
        "Hg_z_A": hg_position[2],
        "adsorption_height_A": adsorption_height,
        "coordination_number_3.5A": coordination,
        "nearest_Hg_Au_distance_A": sorted_distances[0],
        "mean_Hg_Au_distance_within_cutoff_A": (
            np.mean(nearest_distances)
        ),
        "Au_RMS_displacement_A": rms_displacement,
        "Au_maximum_displacement_A": max_displacement
    })

    print("Hg position:", hg_position)
    print("Adsorption height:", adsorption_height, "Å")
    print("Coordination number:", coordination)
    print("Nearest Hg-Au distance:", sorted_distances[0], "Å")
    print("Hg-Au distances within cutoff:", nearest_distances)
    print("Au RMS displacement:", rms_displacement, "Å")
    print("Au maximum displacement:", max_displacement, "Å")

# Write structural summary
summary_file = f"{TABLES}/nequip_structural_summary.csv"

with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

# Write all Hg-Au distances
distance_file = f"{TABLES}/nequip_hg_au_distances.csv"

with open(distance_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=distance_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(distance_results)

# Write layer-resolved displacement
layer_file = f"{TABLES}/nequip_layer_displacements.csv"

with open(layer_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=layer_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(layer_results)

print("\n" + "=" * 60)
print("NEQUIP STRUCTURAL ANALYSIS COMPLETED")
print("=" * 60)

for filename in [summary_file, distance_file, layer_file]:
    print("Saved:", filename)

