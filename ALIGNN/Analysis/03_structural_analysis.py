
import csv
import os
import numpy as np

from ase.io import read
from ase.geometry import find_mic

# --------------------------------------------------
# Parameters
# --------------------------------------------------
HG_AU_CUTOFF = 3.5  # Angstrom
LAYER_GAP = 1.0     # Angstrom; separates atomic layers

os.makedirs("analysis/tables", exist_ok=True)

# --------------------------------------------------
# Configuration mapping
# --------------------------------------------------
configurations = {
    "4x4x4_vac12": {
        "clean": "convergence/outputs/structures/clean_Au100_4x4x4.extxyz",
        "ads": "convergence/outputs/structures/Hg_Au100_4x4x4",
    },
    "4x4x6_vac12": {
        "clean": "convergence/outputs/structures/clean_Au100_4x4x6.extxyz",
        "ads": "convergence/outputs/structures/Hg_Au100_4x4x6",
    },
    "5x5x6_vac12": {
        "clean": "outputs/structures/clean_Au100_ALIGNN.extxyz",
        "ads": "outputs/structures/Hg_Au100",
    },
    "5x5x6_vac16": {
        "clean": "convergence/outputs/structures/clean_Au100_5x5x6_vac16.extxyz",
        "ads": "convergence/outputs/structures/Hg_Au100_5x5x6_vac16",
    },
}

sites = ["ontop", "bridge", "hollow"]

# --------------------------------------------------
# Identify atomic layers from clean slab
# --------------------------------------------------
def identify_layers(clean):

    z = clean.positions[:, 2]
    sorted_indices = np.argsort(z)

    layers = []
    current_layer = [sorted_indices[0]]

    for idx in sorted_indices[1:]:

        previous_idx = current_layer[-1]

        if abs(z[idx] - z[previous_idx]) > LAYER_GAP:
            layers.append(current_layer)
            current_layer = [idx]
        else:
            current_layer.append(idx)

    layers.append(current_layer)

    return layers


# --------------------------------------------------
# Structural analysis
# --------------------------------------------------
overall_results = []
layer_results = []

for configuration, paths in configurations.items():

    print("\n======================================")
    print("Configuration:", configuration)
    print("======================================")

    clean = read(paths["clean"])
    n_au = len(clean)

    if not all(symbol == "Au" for symbol in clean.get_chemical_symbols()):
        raise ValueError("Clean reference contains non-Au atoms")

    layers = identify_layers(clean)

    print("Au atoms:", n_au)
    print("Number of layers:", len(layers))

    for site in sites:

        ads_file = f"{paths['ads']}_{site}.extxyz"

        if not os.path.exists(ads_file):
            raise FileNotFoundError(ads_file)

        ads = read(ads_file)

        if len(ads) != n_au + 1:
            raise ValueError(
                f"Unexpected atom count in {ads_file}"
            )

        if ads[-1].symbol != "Hg":
            raise ValueError(
                f"Last atom is not Hg in {ads_file}"
            )

        # ------------------------------------------
        # Hg-Au distances using minimum image
        # ------------------------------------------
        hg_index = n_au

        distances = ads.get_distances(
            hg_index,
            np.arange(n_au),
            mic=True
        )

        sorted_distances = np.sort(distances)

        nearest_distance = sorted_distances[0]

        coordination = int(
            np.sum(distances <= HG_AU_CUTOFF)
        )

        # ------------------------------------------
        # Adsorption height
        # ------------------------------------------
        top_au_z = np.max(ads.positions[:n_au, 2])
        hg_z = ads.positions[hg_index, 2]

        adsorption_height = hg_z - top_au_z

        # ------------------------------------------
        # Au displacement relative to clean slab
        # ------------------------------------------
        displacement_vectors = (
            ads.positions[:n_au] - clean.positions
        )

        displacement_vectors, _ = find_mic(
            displacement_vectors,
            clean.cell,
            pbc=clean.pbc
        )

        displacements = np.linalg.norm(
            displacement_vectors,
            axis=1
        )

        mean_displacement = np.mean(displacements)
        rms_displacement = np.sqrt(
            np.mean(displacements**2)
        )
        max_displacement = np.max(displacements)

        # ------------------------------------------
        # Layer-resolved displacement
        # ------------------------------------------
        for layer_number, indices in enumerate(layers, start=1):

            layer_displacements = displacements[indices]

            layer_results.append({
                "configuration": configuration,
                "site": site,
                "layer": layer_number,
                "number_of_Au_atoms": len(indices),
                "mean_displacement_A": np.mean(layer_displacements),
                "rms_displacement_A": np.sqrt(
                    np.mean(layer_displacements**2)
                ),
                "max_displacement_A": np.max(
                    layer_displacements
                )
            })

        # ------------------------------------------
        # Save overall results
        # ------------------------------------------
        overall_results.append({
            "model": "ALIGNN",
            "configuration": configuration,
            "site": site,
            "nearest_Hg_Au_A": nearest_distance,
            "coordination_cutoff_A": HG_AU_CUTOFF,
            "coordination_number": coordination,
            "nearest_6_distances_A": ";".join(
                f"{d:.6f}" for d in sorted_distances[:6]
            ),
            "adsorption_height_A": adsorption_height,
            "mean_Au_displacement_A": mean_displacement,
            "rms_Au_displacement_A": rms_displacement,
            "max_Au_displacement_A": max_displacement
        })

        print(
            f"{site:8s} | "
            f"d(Hg-Au) = {nearest_distance:.4f} Å | "
            f"CN = {coordination} | "
            f"height = {adsorption_height:.4f} Å | "
            f"Au RMS = {rms_displacement:.5f} Å"
        )


# --------------------------------------------------
# Write overall structural results
# --------------------------------------------------
overall_file = (
    "analysis/tables/ALIGNN_structural_analysis.csv"
)

with open(overall_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=overall_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(overall_results)


# --------------------------------------------------
# Write layer-resolved results
# --------------------------------------------------
layer_file = (
    "analysis/tables/ALIGNN_layer_displacements.csv"
)

with open(layer_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=layer_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(layer_results)


# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n======================================")
print("ALIGNN STRUCTURAL ANALYSIS COMPLETED")
print("======================================")

print("Structures analysed:", len(overall_results))
print("Layer records:", len(layer_results))

print("\nSaved files:")
print(overall_file)
print(layer_file)

print("\nHg-Au coordination cutoff:", HG_AU_CUTOFF, "Å")
print("Layer separation threshold:", LAYER_GAP, "Å")

