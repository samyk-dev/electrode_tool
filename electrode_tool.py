import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Constants ===
ELECTRODE_AREA = 1.3273  # cm²
ALUMINUM_MASS = 0.005525  # grams

tool_choice = st.radio(
    "Select Tool",
    options=["Slurry Calculator", "Blade Height Recommender", "Capacity Match Tool"],
    index=0
)

if tool_choice == "Slurry Calculator":

    # === Slurry Calculator ===
    def calculate_slurry_recipe(active_mass, ratio, binder_solution_pct, solid_pct, use_solution):
        total_ratio = sum(ratio)
        active_frac = ratio[0] / total_ratio
        carbon_frac = ratio[1] / total_ratio
        binder_frac = ratio[2] / total_ratio

        carbon_mass = (carbon_frac / active_frac) * active_mass
        binder_mass = (binder_frac / active_frac) * active_mass

        if use_solution:
            binder_solution_mass = binder_mass / (binder_solution_pct / 100)
        else:
            binder_solution_mass = binder_mass

        total_solids = active_mass + carbon_mass + binder_mass
        total_slurry_mass = total_solids / (solid_pct / 100)
        solvent_mass = total_slurry_mass - total_solids

        return {
            "Active Mass (g)": active_mass,
            "Carbon Mass (g)": carbon_mass,
            "Binder Mass (g)": binder_mass,
            "Binder Solution Mass (g)": binder_solution_mass if use_solution else None,
            "Solvent Mass (g)": solvent_mass,
            "Total Solids (g)": total_solids,
            "Total Slurry Mass (g)": total_slurry_mass,
            "Initial Solid %": solid_pct
        }

    # === App Layout ===
    st.title("Battery Lab Tool: Slurry + Blade Height")
    st.markdown("Designed for cathode electrode slurry and coating optimization.")

    st.header("Electrode Slurry Recipe")

    # === Inputs ===
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Mass Ratio (must add up to 100%)")
        active_ratio = st.number_input("Active Material % in Formula", min_value=0.0, max_value=100.0, value=96.0, step=0.1)
        carbon_ratio = st.number_input("Carbon % in Formula", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        binder_ratio = st.number_input("Binder % in Formula", min_value=0.0, max_value=100.0, value=2.0, step=0.1)


        total_ratio = active_ratio + carbon_ratio + binder_ratio
        ratio = None

        if abs(total_ratio - 100.0) > 0.01:
            st.error(f"Mass ratio values must add up to 100%. Current total: {total_ratio:.2f}%")
        else:
            ratio = [active_ratio, carbon_ratio, binder_ratio]

        if ratio is not None:
            def get_ingredient_mixture(name, default_entries):
                st.markdown(f"#### {name}")
                count = st.number_input(f"Number of {name} components", min_value=1, max_value=5, value=len(default_entries), key=f"{name}_count")

                names = []
                weights = []
                for i in range(count):
                    col1, col2 = st.columns(2)
                    with col1:
                        comp_name = st.text_input(f"{name} #{i+1} name", value=default_entries[i][0] if i < len(default_entries) else "", key=f"{name}_{i}_name")
                    with col2:
                        comp_weight = st.number_input(f"{name} #{i+1} %", min_value=0.0, max_value=100.0, value=default_entries[i][1] if i < len(default_entries) else 0.0, key=f"{name}_{i}_weight")
                    names.append(comp_name)
                    weights.append(comp_weight)

                total = sum(weights)
                if abs(total - 100.0) > 0.01:
                    st.error(f"{name} components must add up to 100%. Current total: {total:.2f}%")
                    return None, None
                return names, weights

            active_names, active_weights = get_ingredient_mixture("Active Material", [("NVP", 100.0)])
            carbon_names, carbon_weights = get_ingredient_mixture("Carbon", [("Super P", 100.0)])
            binder_names, binder_weights = get_ingredient_mixture("Binder", [("CMC", 60.0), ("SBR", 40.0)])

            if None in [active_weights, carbon_weights, binder_weights]:
                st.stop()
        else:
            st.stop()




        active_pct = active_ratio / 100.0
        active_mass = st.number_input("Active Material Mass (g)", min_value=0.0, step=0.1, key="active_mass")
    with col2:
        use_solution = st.checkbox("Using Binder Solution?", value=True)
        binder_solution_pct = st.number_input("Binder % in Solution", min_value=0.1, max_value=100.0, value=5.0)
        solid_pct = st.number_input("Target Solid Content (%)", min_value=0.1, max_value=100.0, value=30.0)

    # === Calculate Base Slurry ===
    try:
        if ratio is None:
            st.stop()  # Stop if ratio is invalid

        recipe = calculate_slurry_recipe(active_mass, ratio, binder_solution_pct, solid_pct, use_solution)

        st.subheader("Initial Slurry Recipe")

        import pandas as pd

        # Prepare a clean dictionary for display (exclude None values)
        display_recipe = {k: v for k, v in recipe.items() if v is not None}
        # Format floats nicely
        for k in display_recipe:
            display_recipe[k] = round(display_recipe[k], 4)

        df_recipe = pd.DataFrame.from_dict(display_recipe, orient='index', columns=["Mass (g)"])
        st.table(df_recipe)

            # Show binder breakdown
        if use_solution and binder_weights:
            st.markdown("**Binder Composition Breakdown:**")
            for name, wt in zip(binder_names, binder_weights):
                binder_mass_component = recipe["Binder Mass (g)"] * (wt / 100)
                st.markdown(f"- {name}: **{binder_mass_component:.4f} g**")


    except Exception as e:
        st.error(f"Error calculating recipe: {e}")

    # === Initialize dilution state ===
    if "dilutions" not in st.session_state:
        st.session_state.dilutions = []

    st.subheader("Oh crap, my slurry is too viscous")

    # Initial dilution input
    initial_target_solid = st.number_input(
        "Initial Desired Solid Content (%)", 
        min_value=0.1, max_value=100.0, value=30.0, step=0.1,
        key="initial_target_solid",
        help="First target to reduce viscosity by lowering solid content."
    )

    if st.button("Calculate Required Additional Solvent"):
        if recipe and initial_target_solid < recipe["Initial Solid %"]:
            total_solids = recipe["Total Solids (g)"]
            original_mass = recipe["Total Slurry Mass (g)"]

            new_total_mass = total_solids / (initial_target_solid / 100)
            added_solvent = new_total_mass - original_mass

            st.session_state.dilutions.append(added_solvent)
            st.success(f"Add **{added_solvent:.4f} g** solvent to reach {initial_target_solid:.2f}% solids.")
        else:
            st.error("Enter a desired solid % lower than the original value and ensure recipe is valid.")

    # === Still Too Viscous ===
    st.subheader("Crap, my slurry is STILL too viscous ")

    further_target_solid = st.number_input(
        "New Target Solid Content (%)",
        min_value=0.1, max_value=100.0, value=25.0, step=0.1,
        key="further_target_solid",
        help="Use this if you've already diluted once, but it's still too thick."
    )

    if st.button("Recalculate Additional Solvent for New Target"):
        if recipe:
            total_solids = recipe["Total Solids (g)"]
            current_mass = recipe["Total Slurry Mass (g)"] + sum(st.session_state.dilutions)
            current_solid_pct = (total_solids / current_mass) * 100

            if further_target_solid < current_solid_pct:
                required_mass = total_solids / (further_target_solid / 100)
                additional_solvent = required_mass - current_mass

                st.session_state.dilutions.append(additional_solvent)
                st.success(f"Add **{additional_solvent:.4f} g** more solvent to reach {further_target_solid:.2f}% solids.")
            else:
                st.info(f"You're already below {further_target_solid:.2f}% solids. No further solvent needed.")
        else:
            st.error("No valid slurry recipe to adjust.")

    # === Optional Reset Button ===
    if st.button("🔄 Reset All Dilutions"):
        st.session_state.dilutions = []
        st.success("Dilution history reset.")

    # === Show Total Dilutions and New Solid % ===
    if st.session_state.dilutions:
        st.subheader("➕ Dilution History")

        for i, amt in enumerate(st.session_state.dilutions):
            st.write(f"Dilution {i+1}: {amt:.4f} g")

        total_solids = recipe["Total Solids (g)"]
        total_mass = recipe["Total Slurry Mass (g)"] + sum(st.session_state.dilutions)
        new_solid_pct = (total_solids / total_mass) * 100

        st.markdown(f"**Total Solvent Added:** {sum(st.session_state.dilutions):.4f} g")
        st.markdown(f"**New Solid Content:** {new_solid_pct:.2f}%")

    # === True Recipe Tracker (Actual Measured Inputs) ===
    st.subheader("True Slurry Recipe Tracker (Post-Mixing Actual Measurements)")

    col1, col2 = st.columns(2)
    with col1:
        actual_active_mass = st.number_input("Measured Active Material Mass (g)", min_value=0.0, step=0.0001)
        actual_carbon_mass = st.number_input("Measured Carbon Mass (g)", min_value=0.0, step=0.00001)
    with col2:
        actual_binder_mass = st.number_input("Measured Binder Mass (g)", min_value=0.0, step=0.00001)
        actual_solvent_mass = st.number_input("Measured Solvent Mass (g)", min_value=0.0, step=0.00001)

    if any(m > 0 for m in [actual_active_mass, actual_carbon_mass, actual_binder_mass, actual_solvent_mass]):
        total_solids = actual_active_mass + actual_carbon_mass + actual_binder_mass
        total_slurry = total_solids + actual_solvent_mass
        solid_pct = (total_solids / total_slurry) * 100 if total_slurry > 0 else 0.0

        # Individual percentages (based on total solids)
        active_pct = (actual_active_mass / total_solids) * 100 if total_solids > 0 else 0.0
        carbon_pct = (actual_carbon_mass / total_solids) * 100 if total_solids > 0 else 0.0
        binder_pct = (actual_binder_mass / total_solids) * 100 if total_solids > 0 else 0.0

        st.markdown("Actual Slurry Composition")
        st.markdown(f"- **Total Slurry Mass:** `{total_slurry:.4f} g`")
        st.markdown(f"- **Total Solids:** `{total_solids:.4f} g`")
        st.markdown(f"- **Solid Content:** `{solid_pct:.3f}%`")
        st.markdown(f"- **Active Material % of Solids:** `{active_pct:.3f}%`")
        st.markdown(f"- **Carbon % of Solids:** `{carbon_pct:.3f}%`")
        st.markdown(f"- **Binder % of Solids:** `{binder_pct:.3f}%`")
    else:
        st.info("Enter actual measured masses to calculate the true recipe.")

    with st.expander("📘 Wait, how did you get that? Show math and assumptions"):
        st.markdown("### Assumptions and Inputs")
        st.markdown(f"- Active material mass: `{active_mass:.4f} g`")
        st.markdown(f"- Ratio (active:carbon:binder): `{active_ratio:.1f}:{carbon_ratio:.1f}:{binder_ratio:.1f}`")
        st.markdown(f"- Binder solution %: `{binder_solution_pct:.2f}%`")
        st.markdown(f"- Target solid content: `{solid_pct:.2f}%`")
        st.markdown(f"- Using binder solution: `{use_solution}`")

        st.markdown("### Step-by-Step Calculation")
        st.markdown(f"- Total ratio = `{active_ratio + carbon_ratio + binder_ratio:.1f}` (normalized to 100%)")
        st.markdown(f"- Normalized fractions:\n  - Active: `{active_ratio/100:.4f}`\n  - Carbon: `{carbon_ratio/100:.4f}`\n  - Binder: `{binder_ratio/100:.4f}`")

        carbon_mass_calc = f"({carbon_ratio}/{active_ratio}) × {active_mass:.4f} = {recipe['Carbon Mass (g)']:.4f} g"
        binder_mass_calc = f"({binder_ratio}/{active_ratio}) × {active_mass:.4f} = {recipe['Binder Mass (g)']:.4f} g"
        binder_solution_calc = f"{recipe['Binder Mass (g)']:.4f} / ({binder_solution_pct/100:.2f}) = {recipe['Binder Solution Mass (g)']:.4f} g" if use_solution else "N/A (not using solution)"
        solids_calc = f"{active_mass:.4f} + {recipe['Carbon Mass (g)']:.4f} + {recipe['Binder Mass (g)']:.4f} = {recipe['Total Solids (g)']:.4f} g"
        slurry_mass_calc = f"{recipe['Total Solids (g)']:.4f} / ({solid_pct:.2f}/100) = {recipe['Total Slurry Mass (g)']:.4f} g"
        solvent_calc = f"{recipe['Total Slurry Mass (g)']:.4f} - {recipe['Total Solids (g)']:.4f} = {recipe['Solvent Mass (g)']:.4f} g"

        st.markdown("#### Intermediate Steps:")
        st.markdown(f"- **Carbon mass:** {carbon_mass_calc}")
        st.markdown(f"- **Binder mass:** {binder_mass_calc}")
        st.markdown(f"- **Binder solution mass:** {binder_solution_calc}")
        st.markdown(f"- **Total solids:** {solids_calc}")
        st.markdown(f"- **Total slurry mass:** {slurry_mass_calc}")
        st.markdown(f"- **Solvent mass:** {solvent_calc}")



# === Blade Height Tool ===
if tool_choice == "Blade Height Recommender":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    st.header("Blade Height Recommender")
    st.markdown("Please note that this tool is still in early development, and may not have enough data to validate")

    # === Constants ===
    ALUMINUM_MASS = 0.005525  # g
    ELECTRODE_AREA = 1.3273  # cm²

    # === Electrode Mass Data ===
    colors = {
        'NaVP': 'blue',
        'LVP': 'green',
        'Li3V2(PO4)3': 'orange',
        'Na3V2(PO4)3': 'red'
    }

    # === Electrode Data ===
    electrode_data = {
        'NaVP': {
            10: np.array([0.0129, 0.0127, 0.0125, 0.0126, 0.0121, 0.0121, 0.0128, 0.0127, 0.0130]),
            13: np.array([0.0158, 0.0157, 0.0155, 0.0155, 0.0153, 0.0153, 0.0164, 0.0155, 0.0153]),
            16: np.array([0.0181, 0.0183, 0.0177, 0.0181, 0.0180, 0.0176, 0.0179, 0.0173, 0.0177]),
            19: np.array([0.0207, 0.0204, 0.0203, 0.0208, 0.0203, 0.0208, 0.0208, 0.0209, 0.0215]),
            22: np.array([0.0159, 0.0156, 0.0185, 0.0236, 0.0225, 0.0233, 0.0236, 0.0227, 0.0235]),
            25: np.array([0.0234, 0.0226, 0.0229, 0.0248, 0.0250, 0.0251, 0.0251, 0.0252, 0.0257])
        },
        'LVP': {
            10: np.array([0.0095, 0.0090, 0.0098, 0.0089, 0.0091, 0.0104, 0.0093, 0.0089, 0.0094]),
            13: np.array([0.0121, 0.0104, 0.0109, 0.0122, 0.0143, 0.0116, 0.0115, 0.0119, 0.0114]),
            16: np.array([0.0143, 0.0140, 0.0144, 0.0148, 0.0133, 0.0137, 0.0145, 0.0140, 0.0136]),
            22: np.concatenate([
                np.array([0.0166, 0.0169, 0.0152, 0.0179, 0.0173, 0.0168, 0.0162, 0.0175, 0.0176]),
                np.array([0.0176, 0.0177, 0.0181, 0.0192, 0.0181, 0.0158, 0.0190, 0.0188, 0.0181])
            ]),
            25: np.concatenate([
                np.array([0.0108, 0.0132, 0.0180, 0.0167, 0.0140, 0.0097, 0.0114, 0.0143, 0.0154]),
                np.array([0.0215, 0.0213, 0.0217, 0.0202, 0.0209, 0.0211, 0.0193, 0.0201, 0.0202])
            ])
        },
        'Li3V2(PO4)3': {
            10: np.array([0.0102, 0.0086, 0.0092, 0.0092, 0.0083, 0.0086, 0.0086, 0.0084, 0.0086]),
            13: np.array([0.0115, 0.0100, 0.0098, 0.0115, 0.0102, 0.0098, 0.0108, 0.0104, 0.0101]),
            16: np.array([0.0122, 0.0126, 0.0129, 0.0124, 0.0127, 0.0118, 0.0124, 0.0125, 0.0129]),
            19: np.array([0.0108, 0.0115, 0.0115, 0.0099, 0.0113, 0.0123, 0.0106, 0.0113, 0.0130]),
            20: np.array([0.0179, 0.0192, 0.0160, 0.0119, 0.0141, 0.0136, 0.0127, 0.0130, 0.0124]),
            22: np.array([0.0157, 0.0155, 0.0153, 0.0156, 0.0150, 0.0150, 0.0150, 0.0156, 0.0142]),
            25: np.concatenate([
                np.array([0.0181, 0.0162, 0.0169, 0.0182, 0.0173, 0.0163, 0.0177, 0.0152, 0.0154]),
                np.array([0.0198, 0.0194, 0.0199, 0.0135, 0.0150, 0.0144, 0.0139, 0.0151, 0.0148])
            ]),
            30: np.array([0.0225, 0.0198, 0.0211, 0.0118, 0.0134, 0.0129, 0.0107, 0.0167, 0.0153]),
            35: np.array([0.0263, 0.0258, 0.0243, 0.0199, 0.0152, 0.0156, 0.0106, 0.0145, 0.0140])
        },
        'Na3V2(PO4)3': {
            15: np.array([0.0094, 0.0095, 0.0093, 0.0098, 0.0090, 0.0091, 0.0099, 0.0097, 0.0099]),
            20: np.array([0.0118, 0.0111, 0.0110, 0.0111, 0.0115, 0.0118, 0.0119, 0.0119, 0.0119]),
            25: np.array([0.0140, 0.0137, 0.0134, 0.0132, 0.0130, 0.0130, 0.0138, 0.0137, 0.0138]),
            30: np.array([0.0163, 0.0159, 0.0157, 0.0161, 0.0155, 0.0152, 0.0162, 0.0160, 0.0160])
        }
    }

    # === User Inputs ===
    material = st.selectbox("Select Material", list(electrode_data.keys()))
    active_ratio = st.number_input("Active Material % in Formula", min_value=0.0, max_value=100.0, value=96.0, step=0.1)
    target_loading = st.number_input("Target Mass Loading (mg/cm²)", value=2.0, step=0.1)

    samples = electrode_data[material]
    thicknesses, mass_loadings = [], []

    for thickness, masses in samples.items():
        avg_mass = np.mean(masses)
        mass_loading = (avg_mass - ALUMINUM_MASS) * (active_ratio / 100.0) / ELECTRODE_AREA * 1000
        thicknesses.append(thickness)
        mass_loadings.append(mass_loading)

    slope, intercept, *_ = linregress(thicknesses, mass_loadings)
    required_thickness = (target_loading - intercept) / slope if slope != 0 else None

    min_m = min(mass_loadings)
    max_m = max(mass_loadings)

    if min_m <= target_loading <= max_m:
        confidence = "High"
        uncertainty = 0.10
    elif target_loading >= min_m * 0.75 and target_loading <= max_m * 1.25:
        confidence = "Medium"
        uncertainty = 0.25
    else:
        confidence = "Low"
        uncertainty = 0.50

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    fit_x = np.linspace(0, 60, 200)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, color=colors[material], linestyle='--', label=f"{material} Fit")
    ax.scatter(thicknesses, mass_loadings, color=colors[material], label="Measured Data")

    if required_thickness:
        ax.axhline(target_loading, linestyle=':', color='black', label='Target Mass Loading')
        ax.axvline(required_thickness, linestyle=':', color='gray', label=f"Recommended Height: {required_thickness:.1f} μm")

    # Confidence zones
    margin = 0.25 * (max_m - min_m)
    low_conf_min = min_m - 2 * margin
    low_conf_max = max_m + 2 * margin
    med_conf_min = min_m - margin
    med_conf_max = max_m + margin

    ax.axhspan(low_conf_min, med_conf_min, color='red', alpha=0.08)
    ax.axhspan(med_conf_max, low_conf_max, color='red', alpha=0.08)
    ax.axhspan(med_conf_min, min_m, color='orange', alpha=0.08)
    ax.axhspan(max_m, med_conf_max, color='orange', alpha=0.08)
    ax.axhspan(min_m, max_m, color='green', alpha=0.08)

    ax.text(61, (min_m + max_m) / 2, "High Confidence", va='center', ha='left', fontsize=9, color='green', alpha=0.8)
    ax.text(61, (med_conf_min + min_m) / 2, "Medium", va='center', ha='left', fontsize=9, color='darkorange', alpha=0.8)
    ax.text(61, (max_m + med_conf_max) / 2, "Medium", va='center', ha='left', fontsize=9, color='darkorange', alpha=0.8)
    ax.text(61, (low_conf_min + med_conf_min) / 2, "Low", va='center', ha='left', fontsize=9, color='red', alpha=0.8)
    ax.text(61, (med_conf_max + low_conf_max) / 2, "Low", va='center', ha='left', fontsize=9, color='red', alpha=0.8)

    ax.set_xlabel("Blade Height (10 μm)")
    ax.set_ylabel("Mass Loading (mg/cm²)")
    ax.set_title(f"Mass Loading vs Blade Height — {material}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # === Output ===
    if required_thickness:
        st.success(f"Recommended Blade Height: **{required_thickness:.2f} (10 μm)**")
        st.info(f"**Confidence:** {confidence}  \nEstimated Uncertainty: ±{uncertainty:.2f} (10 μm)")
    else:
        st.error("Could not calculate blade height (slope is zero).")

    with st.expander("📘 Wait, how did you get that? Show math and assumptions"):
        st.markdown("### Formula Used")
        st.markdown("Linear regression: `mass_loading = slope × blade_height + intercept`")
        st.markdown(f"- Computed slope: `{slope:.4f}`")
        st.markdown(f"- Intercept: `{intercept:.4f}`")
        st.markdown(f"- Target mass loading: `{target_loading:.2f} mg/cm²`")
        st.markdown(f"**Required blade height:** `({target_loading} - {intercept:.4f}) / {slope:.4f} = {required_thickness:.2f} (10 μm)`")



if tool_choice == "Capacity Match Tool":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    st.header("Capacity Match Tool")
    st.markdown("Match cathode and anode capacities based on desired ratio and calculate required blade height.")

    # === Constants ===
    REFERENCE_DIAMETER_MM = 13
    ELECTRODE_REFERENCE_AREA = np.pi * (REFERENCE_DIAMETER_MM / 2) ** 2 / 100  # cm²

    AL_FOIL_MASS = 0.005525  # g, Aluminum foil for cathode
    CU_FOIL_MASS = 0.011644  # g, Copper foil for anode

    # === Hardcoded Active Material Ratios (decimal fractions) ===
    material_active_ratios = {
        # Cathodes
        'NaVP': 0.96,
        'LVP': 0.96,
        'Li3V2(PO4)3': 0.96,
        'Na3V2(PO4)3': 0.96,
        # Anodes
        'Graphite': 0.80,
        # Add more anodes if needed
    }

    colors = {
        'NaVP': 'blue',
        'LVP': 'green',
        'Li3V2(PO4)3': 'orange',
        'Na3V2(PO4)3': 'red',
        'Graphite': 'gray',
    }

    # === Electrode Data ===
    cathode_data = {
        'NaVP': {
            10: np.array([0.0129, 0.0127, 0.0125, 0.0126, 0.0121, 0.0121, 0.0128, 0.0127, 0.0130]),
            13: np.array([0.0158, 0.0157, 0.0155, 0.0155, 0.0153, 0.0153, 0.0164, 0.0155, 0.0153]),
            16: np.array([0.0181, 0.0183, 0.0177, 0.0181, 0.0180, 0.0176, 0.0179, 0.0173, 0.0177]),
            19: np.array([0.0207, 0.0204, 0.0203, 0.0208, 0.0203, 0.0208, 0.0208, 0.0209, 0.0215]),
            22: np.array([0.0159, 0.0156, 0.0185, 0.0236, 0.0225, 0.0233, 0.0236, 0.0227, 0.0235]),
            25: np.array([0.0234, 0.0226, 0.0229, 0.0248, 0.0250, 0.0251, 0.0251, 0.0252, 0.0257])
        },
        'LVP': {
            10: np.array([0.0095, 0.0090, 0.0098, 0.0089, 0.0091, 0.0104, 0.0093, 0.0089, 0.0094]),
            13: np.array([0.0121, 0.0104, 0.0109, 0.0122, 0.0143, 0.0116, 0.0115, 0.0119, 0.0114]),
            16: np.array([0.0143, 0.0140, 0.0144, 0.0148, 0.0133, 0.0137, 0.0145, 0.0140, 0.0136]),
            22: np.concatenate([
                np.array([0.0166, 0.0169, 0.0152, 0.0179, 0.0173, 0.0168, 0.0162, 0.0175, 0.0176]),
                np.array([0.0176, 0.0177, 0.0181, 0.0192, 0.0181, 0.0158, 0.0190, 0.0188, 0.0181])
            ]),
            25: np.concatenate([
                np.array([0.0108, 0.0132, 0.0180, 0.0167, 0.0140, 0.0097, 0.0114, 0.0143, 0.0154]),
                np.array([0.0215, 0.0213, 0.0217, 0.0202, 0.0209, 0.0211, 0.0193, 0.0201, 0.0202])
            ])
        },
        'Li3V2(PO4)3': {
            10: np.array([0.0102, 0.0086, 0.0092, 0.0092, 0.0083, 0.0086, 0.0086, 0.0084, 0.0086]),
            13: np.array([0.0115, 0.0100, 0.0098, 0.0115, 0.0102, 0.0098, 0.0108, 0.0104, 0.0101]),
            16: np.array([0.0122, 0.0126, 0.0129, 0.0124, 0.0127, 0.0118, 0.0124, 0.0125, 0.0129]),
            19: np.array([0.0108, 0.0115, 0.0115, 0.0099, 0.0113, 0.0123, 0.0106, 0.0113, 0.0130]),
            20: np.array([0.0179, 0.0192, 0.0160, 0.0119, 0.0141, 0.0136, 0.0127, 0.0130, 0.0124]),
            22: np.array([0.0157, 0.0155, 0.0153, 0.0156, 0.0150, 0.0150, 0.0150, 0.0156, 0.0142]),
            25: np.concatenate([
                np.array([0.0181, 0.0162, 0.0169, 0.0182, 0.0173, 0.0163, 0.0177, 0.0152, 0.0154]),
                np.array([0.0198, 0.0194, 0.0199, 0.0135, 0.0150, 0.0144, 0.0139, 0.0151, 0.0148])
            ]),
            30: np.array([0.0225, 0.0198, 0.0211, 0.0118, 0.0134, 0.0129, 0.0107, 0.0167, 0.0153]),
            35: np.array([0.0263, 0.0258, 0.0243, 0.0199, 0.0152, 0.0156, 0.0106, 0.0145, 0.0140])
        },
        'Na3V2(PO4)3': {
            15: np.array([0.0094, 0.0095, 0.0093, 0.0098, 0.0090, 0.0091, 0.0099, 0.0097, 0.0099]),
            20: np.array([0.0118, 0.0111, 0.0110, 0.0111, 0.0115, 0.0118, 0.0119, 0.0119, 0.0119]),
            25: np.array([0.0140, 0.0137, 0.0134, 0.0132, 0.0130, 0.0130, 0.0138, 0.0137, 0.0138]),
            30: np.array([0.0163, 0.0159, 0.0157, 0.0161, 0.0155, 0.0152, 0.0162, 0.0160, 0.0160])
        }
    }

    anode_data = {
        'Graphite': {
            22.5: np.array([0.02172, 0.02047, 0.02013]),
            32.5: np.array([0.02700, 0.02643, 0.02510]),
            42.5: np.array([0.03184, 0.02984, 0.02953]),
            52.5: np.array([0.03592, 0.0497, 0.03747])
        }
    }

    # === User Input Section ===
    st.subheader("1. Input Known Electrode")
    known_side = st.selectbox("Known Side", ["Anode", "Cathode"])

    known_materials = list(anode_data.keys()) if known_side == "Anode" else list(cathode_data.keys())
    known_material = st.selectbox(f"{known_side} Material", known_materials)
    known_diameter_mm = st.number_input(f"{known_side} Diameter (mm)", value=13.0, step=0.01)
    known_area = np.pi * (known_diameter_mm / 2) ** 2 / 100  # convert mm² to cm²
    known_mass_loading = st.number_input(f"{known_side} Mass Loading (mg/cm²)", min_value=0.0, step=0.001)

    st.subheader("2. Target Electrode")
    target_side = "Cathode" if known_side == "Anode" else "Anode"
    target_materials = list(cathode_data.keys()) if target_side == "Cathode" else list(anode_data.keys())
    target_material = st.selectbox(f"{target_side} Material", target_materials)
    target_diameter_mm = st.number_input(f"{target_side} Diameter (mm)", value=13.0, step=0.01)
    target_area = np.pi * (target_diameter_mm / 2) ** 2 / 100  # convert mm² to cm²


    st.subheader("3. Capacity Info")
    known_capacity = st.number_input(f"{known_material} Specific Capacity (mAh/g)", value=100.0, step=0.1)
    target_capacity = st.number_input(f"{target_material} Specific Capacity (mAh/g)", value=100.0, step=0.1)
    anode_ratio = st.number_input("Anode Ratio", value=1.0, step=0.01, format="%.4f")
    cathode_ratio = st.number_input("Cathode Ratio", value=1.0, step=0.01, format="%.4f")

    # === Lookup hardcoded active ratios ===
    known_active_ratio = material_active_ratios.get(known_material, 0.96)
    target_active_ratio = material_active_ratios.get(target_material, 0.96)

    # === Areal Capacity Calculation ===
    known_areal_capacity = (known_mass_loading * known_active_ratio * known_capacity)
    if target_side == "Anode":
        capacity_ratio = anode_ratio
    else:
        capacity_ratio = cathode_ratio
    
    target_areal_capacity = known_areal_capacity * capacity_ratio
    required_mass_loading = target_areal_capacity / target_capacity / target_active_ratio

    # Assign these AFTER mass_loadings is populated:
    target_data = cathode_data if target_side == "Cathode" else anode_data
    foil_mass = AL_FOIL_MASS if target_side == "Cathode" else CU_FOIL_MASS

    samples = target_data[target_material]
    thicknesses, mass_loadings = [], []

    for thickness, masses in samples.items():
        avg_mass = np.mean(masses)
        ml = (avg_mass - foil_mass) * target_active_ratio / ELECTRODE_REFERENCE_AREA * 1000  # mg/cm²
        thicknesses.append(thickness)
        mass_loadings.append(ml)

    min_m = min(mass_loadings)
    max_m = max(mass_loadings)
    target_loading = required_mass_loading



    if min_m <= target_loading <= max_m:
        confidence = "High"
        uncertainty = 0.10
    elif target_loading >= min_m * 0.75 and target_loading <= max_m * 1.25:
        confidence = "Medium"
        uncertainty = 0.25
    else:
        confidence = "Low"
        uncertainty = 0.50

    slope, intercept, *_ = linregress(thicknesses, mass_loadings)
    required_thickness = (required_mass_loading - intercept) / slope if slope != 0 else None

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(10, 6))
    fit_x = np.linspace(0, 60, 200)
    fit_y = slope * fit_x + intercept
    color = colors.get(target_material, 'black')

    ax.plot(fit_x, fit_y, color=color, linestyle='--', label=f"{target_material} Fit")
    ax.scatter(thicknesses, mass_loadings, color=color, label="Measured Data")
    ax.axhline(required_mass_loading, linestyle=':', color='black', label='Target Mass Loading')

    if required_thickness:
        ax.axvline(required_thickness, linestyle=':', color='gray', label=f"Recommended Height: {required_thickness:.1f} μm")

    # Confidence zones for plot
    margin = 0.25 * (max_m - min_m)
    low_conf_min = min_m - 2 * margin
    low_conf_max = max_m + 2 * margin
    med_conf_min = min_m - margin
    med_conf_max = max_m + margin

    ax.axhspan(low_conf_min, med_conf_min, color='red', alpha=0.08)
    ax.axhspan(med_conf_max, low_conf_max, color='red', alpha=0.08)
    ax.axhspan(med_conf_min, min_m, color='orange', alpha=0.08)
    ax.axhspan(max_m, med_conf_max, color='orange', alpha=0.08)
    ax.axhspan(min_m, max_m, color='green', alpha=0.08)

    ax.text(61, (min_m + max_m) / 2, "High Confidence", va='center', ha='left', fontsize=9, color='green', alpha=0.8)
    ax.text(61, (med_conf_min + min_m) / 2, "Medium", va='center', ha='left', fontsize=9, color='darkorange', alpha=0.8)
    ax.text(61, (max_m + med_conf_max) / 2, "Medium", va='center', ha='left', fontsize=9, color='darkorange', alpha=0.8)
    ax.text(61, (low_conf_min + med_conf_min) / 2, "Low", va='center', ha='left', fontsize=9, color='red', alpha=0.8)
    ax.text(61, (med_conf_max + low_conf_max) / 2, "Low", va='center', ha='left', fontsize=9, color='red', alpha=0.8)


    ax.set_xlabel("Blade Height (10 μm)")
    ax.set_ylabel("Mass Loading (mg/cm²)")
    ax.set_title(f"Mass Loading vs Blade Height — {target_material}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # === Output ===
    st.subheader("Results")
    st.markdown(f"**Required {target_side} Mass Loading:** `{required_mass_loading:.2f} mg/cm²`")
    st.markdown(f"**Target Areal Capacity:** `{target_areal_capacity:.2f} mAh/cm²`")
    st.markdown(f"**Assumed {known_side} Active Material %:** `{known_active_ratio*100:.1f}%`")
    st.markdown(f"**Assumed {target_side} Active Material %:** `{target_active_ratio*100:.1f}%`")

    if required_thickness:
        st.success(f"Recommended Blade Height: **{required_thickness:.2f} (10 μm)**")
        st.info(f"**Confidence:** {confidence}  \nEstimated Uncertainty: ±{uncertainty:.2f} (10 μm)")
    else:
        st.error("Could not calculate blade height (slope is zero or invalid data).")

    with st.expander("📘 Wait, how did you get that? Show math and assumptions"):
        st.markdown("### Capacity Matching Formula")
        st.markdown("`Required mass = target_capacity / specific_capacity`")
        st.markdown("`Electrode thickness estimated using linear regression of mass vs blade height`")
        st.markdown("### Assumptions:")
        st.markdown("- Active material % (from formula)")
        st.markdown("- Measured average mass (from dataset)")
        st.markdown("- Area of electrode: `π × (13 mm / 2)² / 100 ≈ 1.3273 cm²`")
        st.markdown("### Example Calculation:")
        st.markdown("- If `target N/P = 1.1` and cathode capacity = `2.0 mAh`, then anode should be `2.2 mAh`.")
        st.markdown("- If anode specific capacity = `350 mAh/g`, required anode mass = `2.2 / 350 ≈ 0.00629 g`")
        st.markdown("- Estimate required thickness from linear regression plot.")
