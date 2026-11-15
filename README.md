# Hg Adsorption on Au(100) using MACE-MP

Atomistic modelling of mercury (Hg) adsorption on the Au(100) surface using the pretrained MACE-MP machine-learning interatomic potential and the Atomic Simulation Environment (ASE).

## Project Overview

This project investigates the adsorption of a single Hg atom at three high-symmetry sites on the Au(100) surface:

- **Ontop:** Directly above a surface Au atom.
- **Bridge:** Between two surface Au atoms.
- **Hollow:** Fourfold hollow site on Au(100).

The main calculation uses a six-layer Au(100) slab with a 5 × 5 surface supercell. The bottom Au layer is fixed while the remaining atoms are allowed to relax.

The calculations use a pretrained MACE-MP model; the model is not trained from scratch in this project.

## Computational Methodology

| Parameter | Value |
|---|---|
| Surface | Au(100) |
| Main surface supercell | 5 × 5 |
| Number of Au layers | 6 |
| Number of Au atoms | 150 |
| Vacuum spacing | 12 Å |
| Fixed atoms | Bottom Au layer (25 atoms) |
| Adsorbate | Single Hg atom |
| Initial Hg height | 2.5 Å |
| Interatomic potential | MACE-MP medium |
| Calculator device | CUDA |
| Optimizer | BFGS |
| Force convergence | 0.005 eV/Å |

An isolated Hg atom in a cubic 15 Å cell is used as the atomic reference. The clean slab and adsorbate–slab systems are relaxed using the same calculator and force criterion.

### Adsorption Energy

The adsorption energy is calculated as:

\[
E_{\mathrm{ads}} =
E_{\mathrm{Hg+slab}} -
E_{\mathrm{slab}} -
E_{\mathrm{Hg}}
\]

A more negative adsorption energy indicates a more energetically favourable adsorption site within this computational model.

## Results

### Adsorption Energies on Au(100)

| Adsorption site | Adsorption energy (eV) |
|---|---:|
| Ontop | -0.607448 |
| Bridge | -0.774535 |
| **Hollow** | **-0.901817** |

The fourfold hollow site has the most negative adsorption energy among the three sites investigated.

### Supercell and Vacuum Convergence

| Calculation | Ontop (eV) | Bridge (eV) | Hollow (eV) |
|---|---:|---:|---:|
| 4 × 4 × 4 | -0.606068 | -0.771262 | -0.898261 |
| 4 × 4 × 6 | -0.607354 | -0.772938 | -0.900269 |
| 5 × 5 × 6 | -0.607448 | -0.774535 | -0.901817 |
| 5 × 5 × 6, 16 Å vacuum | -0.607448 | -0.774535 | -0.901817 |

The results show close agreement between the tested slab dimensions. Increasing the vacuum spacing from 12 Å to 16 Å does not change the reported adsorption energies at the displayed precision.

These tests assess numerical convergence, not the accuracy of MACE-MP for the Hg–Au system.

## Repository Contents

| File | Description |
|---|---|
| `Hg_on_Au200_GPU_5x5.py` | Main 5 × 5 × 6 Au(100) calculation |
| `Hg_on_Au200_GPU_5x5_vac16.py` | Calculation with 16 Å vacuum |
| `Hg_on_Au200_GPU_6layers.py` | Six-layer slab calculation variant |
| `adsorption_energy_results.csv` | Adsorption energies from convergence calculations |
| `plot_adsorption_structures.py` | Structure plotting script |
| `readme.rst` | Earlier project notes |

The main script saves relaxed structures and ASE trajectories in `results_Au200_GPU_5x5/`.

## Requirements

- Python
- ASE
- MACE
- NumPy
- CUDA-enabled PyTorch installation for GPU execution

Install the main packages:

```bash
python -m pip install ase mace-torch numpy
```

The main script specifies `device='cuda'`, so a compatible CUDA-enabled PyTorch installation and NVIDIA GPU are required to run it as written.

## Running the Calculation

Clone the repository:

```bash
git clone https://github.com/Camavingh/Hg-Adsorption-on-Gold.git
cd Hg-Adsorption-on-Gold
```

Run the main calculation:

```bash
python Hg_on_Au200_GPU_5x5.py
```

The script calculates the relaxed adsorption energies for the ontop, bridge, and hollow sites and saves the corresponding structures and trajectories.

## Scope and Limitations

- The results are specific to the pretrained MACE-MP medium model and the computational settings described above.
- Only three high-symmetry adsorption sites are compared.
- Numerical convergence does not independently establish the accuracy of the potential.
- Further validation against suitable reference calculations or experimental data is required.
- Additional machine-learning interatomic potentials will be compared after their corresponding calculations are completed.

## Author

**Debarun Datta**  
BS-MS Physics, IISER Kolkata
