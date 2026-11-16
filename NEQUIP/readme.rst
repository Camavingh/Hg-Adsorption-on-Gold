==================================
Hg Adsorption on Au(100) using NequIP
==================================

Project Overview
================

This project investigates the adsorption of an isolated mercury (Hg) atom
on the Au(100) surface using the NequIP machine-learning interatomic
potential (MLIP).

Three adsorption configurations are investigated:

* Ontop
* Bridge
* Hollow

The project includes baseline adsorption calculations, slab-size
convergence, vacuum convergence, energetic analysis, structural analysis,
and comparison with other machine-learning interatomic potentials.

Machine-Learning Potential
=========================

Model: NequIP-OAM-M

The calculations use a compiled NequIP foundation model:

    NequIP-OAM-M.nequip.pt2

The model is loaded through the ASE NequIP calculator:

    NequIPCalculator.from_compiled_model()

The model file is stored separately from this repository. Update the
MODEL path in the calculation scripts to point to the compiled model
on your system.

Computational Environment
=========================

The calculations were performed using:

* Python 3.11 environment
* NequIP 0.19.1
* PyTorch 2.7.1+cu126
* CUDA-enabled NVIDIA GPU
* Atomic Simulation Environment (ASE)
* BFGS geometry optimization

The calculations used CUDA acceleration.

Computational Methodology
=========================

1. Clean Au(100) slab
---------------------

A clean Au(100) slab is constructed and relaxed using the NequIP model.

The first 25 Au atoms are fixed during the clean-slab relaxation.

2. Hg adsorption
----------------

A single Hg atom is placed at each of the following adsorption sites:

* Ontop
* Bridge
* Hollow

The adsorption structures are optimized using BFGS.

The isolated Hg atom is used as the reference for calculating adsorption
energies.

3. Optimization settings
------------------------

The main optimization settings are:

+------------------------------+----------------+
| Parameter                    | Value          |
+==============================+================+
| Optimizer                    | BFGS           |
+------------------------------+----------------+
| Maximum force threshold      | 0.005 eV/Ang   |
+------------------------------+----------------+
| Maximum optimization steps  | 500            |
+------------------------------+----------------+
| Device                       | CUDA           |
+------------------------------+----------------+

4. Adsorption energy
--------------------

The adsorption energy is calculated as:

.. math::

   E_{ads} = E_{Hg/Au} - E_{Au} - E_{Hg}

where:

* E_{Hg/Au} is the total energy of the relaxed adsorption system.
* E_{Au} is the energy of the relaxed clean Au slab.
* E_{Hg} is the energy of the isolated Hg atom.

A negative adsorption energy indicates energetically favourable
adsorption relative to the separated references.

Convergence Study
=================

Four slab configurations were investigated:

+---------------+-------------+------------+
| Configuration | Slab layers | Vacuum (A) |
+===============+=============+============+
| 4x4x4         | 4           | 12         |
+---------------+-------------+------------+
| 4x4x6         | 6           | 12         |
+---------------+-------------+------------+
| 5x5x6         | 6           | 12         |
+---------------+-------------+------------+
| 5x5x6         | 6           | 16         |
+---------------+-------------+------------+

The 5x5x6 slab with 12 A vacuum is used as the reference configuration
for the main comparison.

Results
=======

Baseline adsorption energies
-----------------------------

The following values correspond to the 5x5x6 Au(100) slab with 12 A
vacuum:

+----------------+-----------------------+
| Adsorption site| Adsorption energy (eV)|
+================+=======================+
| Ontop          | -0.255582             |
+----------------+-----------------------+
| Bridge         | -0.348741             |
+----------------+-----------------------+
| Hollow         | -0.488248             |
+----------------+-----------------------+

The hollow site is the most stable adsorption configuration predicted
by NequIP-OAM-M among the three investigated sites.

Convergence results
-------------------

The relative adsorption energies are calculated with respect to the
lowest-energy adsorption site within each slab configuration.

+---------------+-------------+-------------+-------------+
| Configuration | Ontop (eV) | Bridge (eV) | Hollow (eV) |
+===============+=============+=============+=============+
| 4x4x4, 12 A    | 0.240692    | 0.144664    | 0.000000    |
+---------------+-------------+-------------+-------------+
| 4x4x6, 12 A    | 0.231270    | 0.138580    | 0.000000    |
+---------------+-------------+-------------+-------------+
| 5x5x6, 12 A    | 0.232696    | 0.139557    | 0.000000    |
+---------------+-------------+-------------+-------------+
| 5x5x6, 16 A    | 0.232701    | 0.139570    | 0.000000    |
+---------------+-------------+-------------+-------------+

The preferred adsorption site remains hollow across all four
configurations.

Analysis
========

The project includes the following analyses:

1. Energetic convergence analysis.
2. Relative adsorption energy analysis.
3. Vacuum convergence analysis.
4. Adsorption height analysis.
5. Nearest Hg-Au distance analysis.
6. Coordination number analysis.
7. Au atomic RMS displacement analysis.
8. Au maximum displacement analysis.
9. Layer-resolved displacement analysis.

Directory Structure
===================

::

    ML-IP_nequip/
    |
    |-- scripts/
    |   |-- 01_clean_Au100.py
    |   `-- 02_Hg_adsorption_Au100.py
    |
    |-- outputs/
    |   |-- structures/
    |   `-- logs/
    |
    |-- convergence/
    |   `-- consistent_geometry/
    |       |-- scripts/
    |       |   |-- 01-12: Structure building,
    |       |   |          slab relaxation and adsorption
    |       |   |-- 13: Energetic convergence analysis
    |       |   |-- 14: Structural analysis
    |       |   |-- 15: Structural plotting
    |       |   `-- 16: Energetic analysis
    |       |
    |       |-- outputs/
    |       |   |-- structures/
    |       |   `-- logs/
    |       |
    |       `-- analysis/
    |           |-- figures/
    |           `-- tables/
    |
    |-- analysis/
    |   |-- 03_build_master_dataset.py
    |   |-- figures/
    |   `-- tables/
    |
    `-- readme.rst

Analysis Scripts
================

Baseline calculations
---------------------

    python scripts/01_clean_Au100.py
    python scripts/02_Hg_adsorption_Au100.py

Convergence calculations
------------------------

The scripts in convergence/consistent_geometry/scripts/ are numbered
in execution order.

For each configuration, execute the structure-building, clean-slab
relaxation and adsorption scripts in sequence:

* 01-03: 5x5x6, 12 A vacuum
* 04-06: 4x4x4, 12 A vacuum
* 07-09: 4x4x6, 12 A vacuum
* 10-12: 5x5x6, 16 A vacuum

Run the analysis scripts after the corresponding calculations:

    python convergence/consistent_geometry/scripts/13_analyze_convergence.py

    python convergence/consistent_geometry/scripts/14_structural_analysis.py

    python convergence/consistent_geometry/scripts/15_plot_structural_analysis.py

    python convergence/consistent_geometry/scripts/16_energetic_analysis.py

Master dataset
---------------

Build the NequIP master dataset:

    python analysis/03_build_master_dataset.py

Run the analysis script from the project root directory.

Output Files
============

The project stores calculation outputs, logs, relaxed structures,
analysis tables and figures separately.

Important analytical tables include:

* NequIP_master_dataset.csv
* nequip_convergence_combined.csv
* nequip_structural_summary.csv
* nequip_hg_au_distances.csv
* nequip_layer_displacements.csv
* nequip_energetic_convergence_summary.csv
* nequip_energetic_site_analysis.csv
* nequip_site_energy_gaps.csv

Figures are exported in PNG and PDF formats.

Reproducibility Notes
=====================

* Use the same compiled NequIP-OAM-M model for all calculations.
* Update the model path in the scripts for the local installation.
* Run calculations from the project root directory unless a script
  explicitly specifies otherwise.
* Preserve the same slab constraints and optimization settings when
  reproducing the calculations.
* The compiled foundation model is not included in this repository.
* Initial structures and all required calculation inputs must be
  available before running the corresponding scripts.

Scientific Scope and Limitations
================================

The results represent predictions from the NequIP-OAM-M potential.
Agreement between different machine-learning potentials should not
be interpreted as proof of agreement with experimental values or
first-principles reference calculations.

The adsorption energies from different potentials have different
energy references. Relative adsorption energies within each model
are therefore particularly useful for comparing predicted site
preferences.

The present calculations investigate a single Hg atom on Au(100)
and do not establish coverage-dependent adsorption behaviour.
EOFcd ~/work/projects/open-collection/theoretical_chemistry/projects/adsorptions_with_ase/Hg_on_gold/ML-IP_nequip
cat > readme.rst <<'EOF'
==================================
Hg Adsorption on Au(100) using NequIP
==================================
