import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Constants ===
ELECTRODE_AREA = 1.3273  # cm²
ALUMINUM_MASS = 0.005525  # grams

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
    active_ratio = st.number_input("Active %", min_value=0.0, max_value=100.0, value=96.0, step=0.1, key="active_ratio")
    carbon_ratio = st.number_input("Carbon %", min_value=0.0, max_value=100.0, value=2.0, step=0.1, key="carbon_ratio")
    binder_ratio = st.number_input("Binder %", min_value=0.0, max_value=100.0, value=2.0, step=0.1, key="binder_ratio")

    total_ratio = active_ratio + carbon_ratio + binder_ratio
    if abs(total_ratio - 100.0) > 0.01:  # allow a tiny margin of error
        st.error(f"Mass ratio values must add up to 100%. Current total: {total_ratio:.2f}%")
        ratio = None
    else:
        ratio = [active_ratio, carbon_ratio, binder_ratio]


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

except Exception as e:
    st.error(f"Error calculating recipe: {e}")

st.subheader("Oh crap, my slurry is too viscous")

desired_solid_pct = st.number_input(
    "Desired Solid Content After Dilution (%)", 
    min_value=0.1, max_value=100.0, value=20.0, step=0.1,
    help="Set target solid content after adding solvent to reduce viscosity."
)

if st.button("Calculate Required Additional Solvent"):
    if recipe and desired_solid_pct < recipe["Initial Solid %"]:
        total_solids = recipe["Total Solids (g)"]
        total_slurry_mass = recipe["Total Slurry Mass (g)"]
        
        # Calculate required total slurry mass to get desired solid %
        required_total_mass = total_solids / (desired_solid_pct / 100)
        additional_solvent_needed = required_total_mass - total_slurry_mass
        
        if additional_solvent_needed > 0:
            st.success(f"You need to add **{additional_solvent_needed:.4f} g** of additional solvent to reduce solid content to {desired_solid_pct:.2f}%.")
        else:
            st.info("No additional solvent needed; current solid content is already below desired level.")
    else:
        st.error("Please enter a desired solid % lower than the initial solid % and have a valid slurry recipe.")

# === Dilution Tracking ===
st.subheader("➕ Track Additional Solvent Dilutions")
if "dilutions" not in st.session_state:
    st.session_state.dilutions = []

new_dilution = st.number_input("Add Dilution Solvent (g)", value=0.0, key="dilution_input")
if st.button("Add Dilution"):
    if new_dilution > 0:
        st.session_state.dilutions.append(new_dilution)

if st.session_state.dilutions:
    for i, val in enumerate(st.session_state.dilutions):
        st.write(f"Dilution {i+1}: {val:.3f} g")

    # Recalculate new solid %
    total_solids = recipe["Total Solids (g)"]
    total_mass_with_dilutions = recipe["Total Slurry Mass (g)"] + sum(st.session_state.dilutions)
    new_solid_pct = (total_solids / total_mass_with_dilutions) * 100

    st.markdown(f"New Solid Content After Dilutions: **{new_solid_pct:.2f}%**")

# === Blade Height Tool ===
st.header("Blade Height Recommender")

# === Electrode Mass Data ===
colors = {'NaVP': 'blue', 'LVP': 'green', 'Li3V2(PO4)3': 'orange', 'Na3V2(PO4)3': 'red'}

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

material = st.selectbox("Select Material", list(electrode_data.keys()))
target_loading = st.number_input("Target Mass Loading (mg/cm²)", value=2.0, step=0.1)

samples = electrode_data[material]
thicknesses, mass_loadings = [], []

for thickness, masses in samples.items():
    avg_mass = np.mean(masses)
    mass_loading = (avg_mass - ALUMINUM_MASS) * active_pct / ELECTRODE_AREA * 1000
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
