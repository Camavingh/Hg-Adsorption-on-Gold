import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

input_dir = "results_Au200_GPU_5x5"
output_dir = "structural_figures_improved"
os.makedirs(output_dir, exist_ok=True)

sites = ["ontop", "bridge", "hollow"]

for site in sites:

    atoms = read(os.path.join(
        input_dir, f"HgAu100_{site}.vasp"
    ))

    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    cell = atoms.cell.array

    au_indices = np.where(symbols == "Au")[0]
    hg_indices = np.where(symbols == "Hg")[0]

    hg_position = positions[hg_indices[0]]
    au_positions = positions[au_indices]

    # Identify the topmost Au layer
    top_z = np.max(au_positions[:, 2])
    top_indices = au_indices[
        au_positions[:, 2] >= top_z - 0.5
    ]

    # In-plane minimum-image distances from Hg
    inv_cell = np.linalg.inv(cell)

    def minimum_image_xy(position):
        delta = position - hg_position
        fractional = delta @ inv_cell
        fractional[0] -= np.round(fractional[0])
        fractional[1] -= np.round(fractional[1])
        return fractional @ cell

    # Local top-layer Au positions around Hg
    top_local = np.array([
        minimum_image_xy(positions[i])
        for i in top_indices
    ])

    distances = np.linalg.norm(top_local[:, :2], axis=1)
    local_mask = distances < 4.2
    top_local = top_local[local_mask]

    # Side-view atoms: top two Au layers
    second_layer_z = np.sort(
        np.unique(np.round(au_positions[:, 2], 2))
    )[-2]

    side_indices = au_indices[
        au_positions[:, 2] >= second_layer_z - 0.5
    ]

    side_local = np.array([
        minimum_image_xy(positions[i])
        for i in side_indices
    ])

    # Create top and side views
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5)
    )

    # ---------------- TOP VIEW ----------------
    ax = axes[0]

    ax.scatter(
        top_local[:, 0], top_local[:, 1],
        s=180, color="gold",
        edgecolors="black", linewidths=0.9,
        label="Au"
    )

    ax.scatter(
        0, 0, s=240,
        color="royalblue",
        edgecolors="black", linewidths=1.1,
        label="Hg", zorder=5
    )

    ax.set_xlabel("x relative to Hg (Å)")
    ax.set_ylabel("y relative to Hg (Å)")
    ax.set_title("Top view")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.grid(alpha=0.2)
    ax.legend()

    # ---------------- SIDE VIEW ----------------
    ax = axes[1]

    ax.scatter(
        side_local[:, 0],
        positions[side_indices, 2],
        s=180, color="gold",
        edgecolors="black", linewidths=0.9,
        label="Au"
    )

    ax.scatter(
        0, hg_position[2],
        s=240, color="royalblue",
        edgecolors="black", linewidths=1.1,
        label="Hg", zorder=5
    )

    ax.set_xlabel("x relative to Hg (Å)")
    ax.set_ylabel("z (Å)")
    ax.set_title("Side view")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.suptitle(
        f"Hg adsorption on Au(200): {site.capitalize()} site",
        fontsize=15
    )

    plt.tight_layout()

    output_file = os.path.join(
        output_dir, f"Hg_Au200_{site}_views.png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")

print("Improved structural figures generated.")
