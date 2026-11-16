
#!/usr/bin/env python3

"""
Structural analysis of Hg adsorption on Au(100) using MACE-MP.

Calculates:
- Hg adsorption height relative to the clean slab's top-layer mean z
- Nearest Hg-Au distance
- Hg-Au coordination number (3.5 Å cutoff)
- Au substrate RMS displacement
- Maximum Au displacement
- Layer-resolved RMS and maximum displacement

Reference:
    Relaxed clean Au(100) slab.

Assumption:
    Au atoms retain the same ordering in the clean and adsorbed
    trajectory files, with Hg as the final atom.
"""

import csv
from pathlib import Path

import numpy as np
from ase.io import read


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

STRUCTURE_DIR = Path("outputs/structures")
TABLE_DIR = Path("analysis/tables")

CLEAN_FILE = STRUCTURE_DIR / "clean_slab_Au100.traj"

ADSORBED_FILES = {
    "Ontop": STRUCTURE_DIR / "HgAu100_ontop.traj",
    "Bridge": STRUCTURE_DIR / "HgAu100_bridge.traj",
    "Hollow": STRUCTURE_DIR / "HgAu100_hollow.traj",
}

COORDINATION_CUTOFF = 3.5  # Angstrom
LAYER_TOLERANCE = 0.5      # Angstrom


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def identify_layers(z_positions, tolerance=0.5):
    """
    Group Au atoms into atomic layers based on their reference z positions.
    Layers are numbered from bottom to top.
    """

    sorted_indices = np.argsort(z_positions)
    layers = []
    current_layer = [sorted_indices[0]]

    for index in sorted_indices[1:]:

        current_mean = np.mean(z_positions[current_layer])

        if abs(z_positions[index] - current_mean) <= tolerance:
            current_layer.append(index)
        else:
            layers.append(current_layer)
            current_layer = [index]

    layers.append(current_layer)

    return layers


def validate_structures(clean, adsorbed, site):

    if len(clean) != 150:
        raise ValueError(
            f"Expected 150 atoms in clean slab; found {len(clean)}"
        )

    if len(adsorbed) != 151:
        raise ValueError(
            f"{site}: expected 151 atoms; found {len(adsorbed)}"
        )

    if clean.get_chemical_symbols() != ["Au"] * 150:
        raise ValueError("Clean reference structure is not 150 Au atoms.")

    if adsorbed.get_chemical_symbols()[:150] != ["Au"] * 150:
        raise ValueError(
            f"{site}: Au atom ordering or composition is unexpected."
        )

    if adsorbed.get_chemical_symbols()[-1] != "Hg":
        raise ValueError(
            f"{site}: expected Hg to be the final atom."
        )

    if not np.allclose(clean.cell.array, adsorbed.cell.array, atol=1e-6):
        raise ValueError(
            f"{site}: clean and adsorbed cells do not match."
        )


# ------------------------------------------------------------
# MAIN STRUCTURAL ANALYSIS
# ------------------------------------------------------------

def main():

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MACE-MP STRUCTURAL ANALYSIS: Hg/Au(100)")
    print("=" * 65)

    clean = read(CLEAN_FILE)
    clean_au_positions = clean.get_positions()

    # Identify layers using the relaxed clean-slab reference.
    reference_z = clean_au_positions[:, 2]
    layers = identify_layers(reference_z, LAYER_TOLERANCE)

    top_layer_indices = layers[-1]
    top_layer_mean_z = np.mean(
        reference_z[top_layer_indices]
    )

    print(f"Clean Au atoms: {len(clean)}")
    print(f"Identified Au layers: {len(layers)}")
    print(f"Au atoms per layer: {[len(layer) for layer in layers]}")
    print(f"Top-layer mean z: {top_layer_mean_z:.6f} Å")
    print(f"Coordination cutoff: {COORDINATION_CUTOFF:.2f} Å")

    structural_rows = []
    layer_rows = []

    # --------------------------------------------------------
    # Analyse each adsorption site
    # --------------------------------------------------------

    for site, filename in ADSORBED_FILES.items():

        if not filename.exists():
            raise FileNotFoundError(
                f"Missing structure: {filename}"
            )

        adsorbed = read(filename)
        validate_structures(clean, adsorbed, site)

        positions = adsorbed.get_positions()

        # Au atoms are the first 150 atoms; Hg is the last atom.
        au_positions = positions[:150]
        hg_index = len(adsorbed) - 1

        # Adsorption height relative to clean slab top-layer mean.
        hg_height = positions[hg_index, 2] - top_layer_mean_z

        # Hg-Au distances using minimum image convention.
        distances = adsorbed.get_distances(
            hg_index,
            list(range(150)),
            mic=True
        )

        nearest_distance = float(np.min(distances))

        coordination_number = int(
            np.count_nonzero(distances <= COORDINATION_CUTOFF)
        )

        # Au substrate displacement relative to clean slab.
        displacement_vectors = au_positions - clean_au_positions
        displacement_magnitudes = np.linalg.norm(
            displacement_vectors,
            axis=1
        )

        rms_displacement = float(
            np.sqrt(np.mean(displacement_magnitudes ** 2))
        )

        maximum_displacement = float(
            np.max(displacement_magnitudes)
        )

        structural_rows.append({
            "Site": site,
            "Adsorption_Height_A": hg_height,
            "Nearest_Hg_Au_Distance_A": nearest_distance,
            "Coordination_Number_3.5A": coordination_number,
            "Au_RMS_Displacement_A": rms_displacement,
            "Au_Max_Displacement_A": maximum_displacement,
            "Top_Layer_Mean_Z_A": top_layer_mean_z,
            "Coordination_Cutoff_A": COORDINATION_CUTOFF,
        })

        # ----------------------------------------------------
        # Layer-resolved displacement
        # ----------------------------------------------------

        for layer_number, indices in enumerate(layers, start=1):

            layer_displacements = displacement_magnitudes[indices]

            layer_rms = float(
                np.sqrt(np.mean(layer_displacements ** 2))
            )

            layer_max = float(
                np.max(layer_displacements)
            )

            layer_rows.append({
                "Site": site,
                "Layer": layer_number,
                "Position": (
                    "Bottom" if layer_number == 1
                    else "Top" if layer_number == len(layers)
                    else "Intermediate"
                ),
                "Number_of_Au_Atoms": len(indices),
                "Reference_Mean_Z_A": float(
                    np.mean(reference_z[indices])
                ),
                "RMS_Displacement_A": layer_rms,
                "Maximum_Displacement_A": layer_max,
            })

        print(f"\n{site}")
        print(f"  Hg adsorption height: {hg_height:.6f} Å")
        print(f"  Nearest Hg-Au distance: {nearest_distance:.6f} Å")
        print(f"  Coordination number: {coordination_number}")
        print(f"  Au RMS displacement: {rms_displacement:.6f} Å")
        print(f"  Au maximum displacement: {maximum_displacement:.6f} Å")

    # --------------------------------------------------------
    # Save structural summary
    # --------------------------------------------------------

    structural_file = TABLE_DIR / "MACE_structural_analysis.csv"

    with open(structural_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(structural_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(structural_rows)

    # --------------------------------------------------------
    # Save layer-resolved data
    # --------------------------------------------------------

    layer_file = TABLE_DIR / "MACE_layer_displacements.csv"

    with open(layer_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(layer_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    print("\n" + "=" * 65)
    print("STRUCTURAL ANALYSIS COMPLETED")
    print("=" * 65)
    print(f"Structural summary: {structural_file}")
    print(f"Layer displacements: {layer_file}")
    print(f"Structural records: {len(structural_rows)}")
    print(f"Layer records: {len(layer_rows)}")


if __name__ == "__main__":
    main()

