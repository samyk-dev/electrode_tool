# %%
#Electrode tool
 # %%
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# === Electrode Constants ===
electrode_area = 1.3273  # cm²
aluminum_mass = 0.005525  # grams
active_pct = {'NaVP': 0.96, 'LVP': 0.96, 'Li3V2(PO4)3': 0.96, 'Na3V2(PO4)3': 0.96}
colors = {'NaVP': 'blue', 'LVP': 'green', 'Li3V2(PO4)3': 'orange', 'Na3V2(PO4)3': 'red'}

# === Electrode Mass Data ===

# NaVP Samples
NaVP_10 = np.array([0.0129, 0.0127, 0.0125, 0.0126, 0.0121, 0.0121, 0.0128, 0.0127, 0.0130])
NaVP_13 = np.array([0.0158, 0.0157, 0.0155, 0.0155, 0.0153, 0.0153, 0.0164, 0.0155, 0.0153])
NaVP_16 = np.array([0.0181, 0.0183, 0.0177, 0.0181, 0.0180, 0.0176, 0.0179, 0.0173, 0.0177])
NaVP_19 = np.array([0.0207, 0.0204, 0.0203, 0.0208, 0.0203, 0.0208, 0.0208, 0.0209, 0.0215])
NaVP_22 = np.array([0.0159, 0.0156, 0.0185, 0.0236, 0.0225, 0.0233, 0.0236, 0.0227, 0.0235])
NaVP_25 = np.array([0.0234, 0.0226, 0.0229, 0.0248, 0.0250, 0.0251, 0.0251, 0.0252, 0.0257])

# LVP Samples
LVP_10 = np.array([0.0095, 0.0090, 0.0098, 0.0089, 0.0091, 0.0104, 0.0093, 0.0089, 0.0094])
LVP_13 = np.array([0.0121, 0.0104, 0.0109, 0.0122, 0.0143, 0.0116, 0.0115, 0.0119, 0.0114])
LVP_16 = np.array([0.0143, 0.0140, 0.0144, 0.0148, 0.0133, 0.0137, 0.0145, 0.0140, 0.0136])
LVP_22_1 = np.array([0.0166, 0.0169, 0.0152, 0.0179, 0.0173, 0.0168, 0.0162, 0.0175, 0.0176])
LVP_22_2 = np.array([0.0176, 0.0177, 0.0181, 0.0192, 0.0181, 0.0158, 0.0190, 0.0188, 0.0181])
LVP_25_1 = np.array([0.0108, 0.0132, 0.0180, 0.0167, 0.0140, 0.0097, 0.0114, 0.0143, 0.0154])
LVP_25_2 = np.array([0.0215, 0.0213, 0.0217, 0.0202, 0.0209, 0.0211, 0.0193, 0.0201, 0.0202])

# Li3V2(PO4)3 Samples
E_5008_10_1 = np.array([0.0102, 0.0086, 0.0092, 0.0092, 0.0083, 0.0086, 0.0086, 0.0084, 0.0086])
E_5008_13_1 = np.array([0.0115, 0.0100, 0.0098, 0.0115, 0.0102, 0.0098, 0.0108, 0.0104, 0.0101])
E_5008_16_1 = np.array([0.0122, 0.0126, 0.0129, 0.0124, 0.0127, 0.0118, 0.0124, 0.0125, 0.0129])
E_5008_19_1 = np.array([0.0108, 0.0115, 0.0115, 0.0099, 0.0113, 0.0123, 0.0106, 0.0113, 0.0130])
E_5008_20_2 = np.array([0.0179, 0.0192, 0.0160, 0.0119, 0.0141, 0.0136, 0.0127, 0.0130, 0.0124])
E_5008_22_1 = np.array([0.0157, 0.0155, 0.0153, 0.0156, 0.0150, 0.0150, 0.0150, 0.0156, 0.0142])
E_5008_25_1 = np.array([0.0181, 0.0162, 0.0169, 0.0182, 0.0173, 0.0163, 0.0177, 0.0152, 0.0154])
E_5008_25_2 = np.array([0.0198, 0.0194, 0.0199, 0.0135, 0.0150, 0.0144, 0.0139, 0.0151, 0.0148])
E_5008_30_2 = np.array([0.0225, 0.0198, 0.0211, 0.0118, 0.0134, 0.0129, 0.0107, 0.0167, 0.0153])
E_5008_35_2 = np.array([0.0263, 0.0258, 0.0243, 0.0199, 0.0152, 0.0156, 0.0106, 0.0145, 0.0140])

# Na3V2(PO4)3 Samples
E_5001_15 = np.array([0.0094, 0.0095, 0.0093, 0.0098, 0.0090, 0.0091, 0.0099, 0.0097, 0.0099])
E_5001_20 = np.array([0.0118, 0.0111, 0.0110, 0.0111, 0.0115, 0.0118, 0.0119, 0.0119, 0.0119])
E_5001_25 = np.array([0.0140, 0.0137, 0.0134, 0.0132, 0.0130, 0.0130, 0.0138, 0.0137, 0.0138])
E_5001_30 = np.array([0.0163, 0.0159, 0.0157, 0.0161, 0.0155, 0.0152, 0.0162, 0.0160, 0.0160])

# === Data Dictionary ===
electrode_data = {
    'NaVP': {
        10: NaVP_10, 13: NaVP_13, 16: NaVP_16, 19: NaVP_19, 22: NaVP_22, 25: NaVP_25
    },
    'LVP': {
        10: LVP_10, 13: LVP_13, 16: LVP_16,
        22: np.concatenate([LVP_22_1, LVP_22_2]),
        25: np.concatenate([LVP_25_1, LVP_25_2])
    },
    'Li3V2(PO4)3': {
        10: E_5008_10_1, 13: E_5008_13_1, 16: E_5008_16_1, 19: E_5008_19_1,
        20: E_5008_20_2, 22: E_5008_22_1,
        25: np.concatenate([E_5008_25_1, E_5008_25_2]),
        30: E_5008_30_2, 35: E_5008_35_2
    },
    'Na3V2(PO4)3': {
        15: E_5001_15, 20: E_5001_20, 25: E_5001_25, 30: E_5001_30
    }
}

# === Streamlit UI ===
st.title("Blade Height Recommender")
material = st.selectbox("Select Material", list(electrode_data.keys()))

target_mass_loading = st.number_input(
    "Target Mass Loading (mg/cm²)",
    min_value=0.1, max_value=50.0, value=2.0, step=0.1
)

# active_pct_slider = st.slider(
active_pct_slider = st.number_input(
    f"Active Material Percentage for {material} (%)",
    min_value=.0, max_value=100.0, value=96.0, step=0.1
) / 100.0  # Convert to decimal for calculation

samples = electrode_data[material]
thicknesses, mass_loadings = [], []

for thickness, masses in samples.items():
    avg_mass = np.mean(masses)
    mass_loading = (avg_mass - aluminum_mass) * active_pct_slider / electrode_area * 1000
    thicknesses.append(thickness)
    mass_loadings.append(mass_loading)


slope, intercept, *_ = linregress(thicknesses, mass_loadings)
required_thickness = (target_mass_loading - intercept) / slope if slope != 0 else None

min_measured = min(mass_loadings)
max_measured = max(mass_loadings)

# Determine confidence
if min_measured <= target_mass_loading <= max_measured:
    confidence = "High"
    uncertainty = 0.10
elif (
    target_mass_loading >= min_measured * 0.75 and
    target_mass_loading <= max_measured * 1.25
):
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
    ax.axhline(target_mass_loading, linestyle=':', color='black', label='Target Mass Loading')
    ax.axvline(required_thickness, linestyle=':', color='gray', label=f"Recommended Height: {required_thickness:.1f} μm")

# Define mass loading bounds
min_m = min(mass_loadings)
max_m = max(mass_loadings)
margin = 0.25 * (max_m - min_m)

low_conf_min = min_m - 2 * margin
low_conf_max = max_m + 2 * margin
med_conf_min = min_m - margin
med_conf_max = max_m + margin

# Low confidence zones (red)
ax.axhspan(low_conf_min, med_conf_min, color='red', alpha=0.08)
ax.axhspan(med_conf_max, low_conf_max, color='red', alpha=0.08)

# Medium confidence zones (orange)
ax.axhspan(med_conf_min, min_m, color='orange', alpha=0.08)
ax.axhspan(max_m, med_conf_max, color='orange', alpha=0.08)

# High confidence zone (green)
ax.axhspan(min_m, max_m, color='green', alpha=0.08)

# Add confidence zone labels
ax.text(61, (min_m + max_m) / 2, "High Confidence", va='center', ha='left',
        fontsize=9, color='green', alpha=0.8)

ax.text(61, (med_conf_min + min_m) / 2, "Medium", va='center', ha='left',
        fontsize=9, color='darkorange', alpha=0.8)

ax.text(61, (max_m + med_conf_max) / 2, "Medium", va='center', ha='left',
        fontsize=9, color='darkorange', alpha=0.8)

ax.text(61, (low_conf_min + med_conf_min) / 2, "Low", va='center', ha='left',
        fontsize=9, color='red', alpha=0.8)

ax.text(61, (med_conf_max + low_conf_max) / 2, "Low", va='center', ha='left',
        fontsize=9, color='red', alpha=0.8)


ax.set_xlabel("Blade Height (10 μm)")
ax.set_ylabel("Mass Loading (mg/cm²)")
ax.set_title(f"Mass Loading vs Blade Height — {material}")
ax.grid(True)
ax.legend()
fig.tight_layout()

st.pyplot(fig)


# === Output ===

if required_thickness:
    st.success(
        f"Recommended Blade Height: **{required_thickness:.2f} (10 μm)**"
    )
    st.info(
        f"**Confidence:** {confidence}  \n"
        f"Estimated Uncertainty: ±{uncertainty:.2f} mg/cm²"
    )
else:
    st.error("Could not calculate blade height (slope is zero).")