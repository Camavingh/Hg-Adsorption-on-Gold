
def get_adsorption_position(slab, site):

    # Find the topmost Au layer
    top_z = slab.positions[:, 2].max()

    top_indices = np.where(
        np.isclose(slab.positions[:, 2], top_z, atol=0.01)
    )[0]

    top_atoms = slab.positions[top_indices]

    # Select a reference atom on the top layer
    ref = top_atoms[
        np.argmin(top_atoms[:, 0] + top_atoms[:, 1])
    ]

    # 5x5 surface: divide supercell length by 5
    a_surface = np.linalg.norm(slab.cell[0]) / 5.0

    if site == "ontop":
        position = (ref[0], ref[1])

    elif site == "bridge":
        position = (
            ref[0] + a_surface / 2,
            ref[1]
        )

    elif site == "hollow":
        position = (
            ref[0] + a_surface / 2,
            ref[1] + a_surface / 2
        )

    else:
        raise ValueError(f"Unknown adsorption site: {site}")

    return position

