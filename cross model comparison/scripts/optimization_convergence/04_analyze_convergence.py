
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# Directories
# ==================================================

BASE = "data/optimization_convergence"
FIG_DIR = "figures/optimization_convergence"

os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["MACE", "ALIGNN", "NequIP"]
SITES = ["ontop", "bridge", "hollow"]

MODEL_LABELS = {
    "MACE": "MACE-MP",
    "ALIGNN": "ALIGNN-FF",
    "NequIP": "NequIP-OAM-M"
}


# ==================================================
# Load and standardize convergence data
# ==================================================

all_data = []

for model in MODELS:

    for site in SITES:

        pattern = os.path.join(
            BASE,
            model,
            f"{model}_{site}_convergence.csv"
        )

        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(pattern)

        df = pd.read_csv(matches[0])

        # Standardize column names
        if model in ["MACE", "ALIGNN"]:

            df = df.rename(columns={
                "Model": "model",
                "Site": "site",
                "Step": "step",
                "Energy_eV": "energy_eV",
                "Delta_E_eV": "relative_energy_eV",
                "Max_Force_eV_per_A": "max_force_eV_A"
            })

        else:

            df = df.rename(columns={
                "step": "step",
                "energy_eV": "energy_eV",
                "relative_energy_eV": "relative_energy_eV",
                "max_force_eV_A": "max_force_eV_A"
            })

        df["model"] = MODEL_LABELS[model]
        df["site"] = site

        # Remove duplicated initial record in MACE and ALIGNN.
        # The first two records are identical initial configurations.
        if model in ["MACE", "ALIGNN"]:

            if len(df) > 1:

                first = df.iloc[0]
                second = df.iloc[1]

                same_initial = (
                    np.isclose(
                        first["energy_eV"],
                        second["energy_eV"]
                    )
                    and np.isclose(
                        first["relative_energy_eV"],
                        second["relative_energy_eV"]
                    )
                    and np.isclose(
                        first["max_force_eV_A"],
                        second["max_force_eV_A"]
                    )
                )

                if same_initial:

                    df = df.drop(df.index[1:2]).copy()

                    # Reindex to actual optimization steps
                    df["step"] = np.arange(len(df))

        df = df[[
            "model",
            "site",
            "step",
            "energy_eV",
            "relative_energy_eV",
            "max_force_eV_A"
        ]]

        all_data.append(df)


# Combine all models and sites
combined = pd.concat(
    all_data,
    ignore_index=True
)

combined = combined.sort_values(
    ["site", "model", "step"]
)

combined.to_csv(
    os.path.join(BASE, "combined_optimization_convergence.csv"),
    index=False
)

print("Combined convergence data saved.")


# ==================================================
# Plot energy convergence
# ==================================================

for site in SITES:

    plt.figure(figsize=(9, 6))

    for model in MODEL_LABELS.values():

        subset = combined[
            (combined["site"] == site)
            & (combined["model"] == model)
        ]

        plt.plot(
            subset["step"],
            subset["relative_energy_eV"],
            linewidth=2,
            label=model
        )

    plt.xlabel("Optimization step", fontsize=12)
    plt.ylabel("Energy change from initial structure (eV)", fontsize=12)
    plt.title(
        f"Energy convergence: {site.capitalize()} adsorption",
        fontsize=14
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"energy_convergence_{site}"

    plt.savefig(
        os.path.join(FIG_DIR, filename + ".png"),
        dpi=300
    )

    plt.savefig(
        os.path.join(FIG_DIR, filename + ".pdf")
    )

    plt.close()


# ==================================================
# Plot force convergence
# ==================================================

for site in SITES:

    plt.figure(figsize=(9, 6))

    for model in MODEL_LABELS.values():

        subset = combined[
            (combined["site"] == site)
            & (combined["model"] == model)
        ]

        plt.plot(
            subset["step"],
            subset["max_force_eV_A"],
            linewidth=2,
            label=model
        )

    plt.axhline(
        0.005,
        linestyle="--",
        linewidth=1.5,
        label="Convergence threshold"
    )

    plt.xlabel("Optimization step", fontsize=12)
    plt.ylabel("Maximum force (eV/Å)", fontsize=12)

    plt.title(
        f"Force convergence: {site.capitalize()} adsorption",
        fontsize=14
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"force_convergence_{site}"

    plt.savefig(
        os.path.join(FIG_DIR, filename + ".png"),
        dpi=300
    )

    plt.savefig(
        os.path.join(FIG_DIR, filename + ".pdf")
    )

    plt.close()


# ==================================================
# Generate convergence summary
# ==================================================

summary = []

for model in MODEL_LABELS.values():

    for site in SITES:

        subset = combined[
            (combined["model"] == model)
            & (combined["site"] == site)
        ].sort_values("step")

        final = subset.iloc[-1]

        summary.append({
            "Model": model,
            "Site": site,
            "Optimization_steps": int(final["step"]),
            "Initial_energy_eV": subset.iloc[0]["energy_eV"],
            "Final_energy_eV": final["energy_eV"],
            "Energy_change_eV": final["relative_energy_eV"],
            "Final_max_force_eV_A": final["max_force_eV_A"],
            "Converged": final["max_force_eV_A"] <= 0.005
        })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(BASE, "optimization_convergence_summary.csv"),
    index=False
)


# ==================================================
# Print final summary
# ==================================================

print("\n" + "=" * 85)
print("CROSS-MODEL OPTIMIZATION CONVERGENCE SUMMARY")
print("=" * 85)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nFigures saved in:", FIG_DIR)
print("Combined data saved in:", BASE)
print("\nAnalysis completed successfully.")

