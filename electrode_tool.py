# Battery Lab Tools

# =========================
# Requirements Config
# =========================

import os, io, math, json, datetime
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from scipy.stats import linregress, t

from io import BytesIO
import xlsxwriter
from fpdf import FPDF


# =========================
# Global Config
# =========================
st.set_page_config(page_title="Slurry Calculator", layout="wide")
plt.rcParams.update({"figure.max_open_warning": 0})

# =========================
# Constants
# =========================
ELECTRODE_AREA_13MM = 1.3273  # cm², 13 mm punch (π * (6.5 mm)^2 / 100)
DEFAULT_DB_FILE = "electrode_database.xlsx"

# Foil masses for 13 mm punches measured in earlier data
AL_FOIL_MASS_13MM = 0.005525  # g
CU_FOIL_MASS_13MM = 0.011644  # g
HC_FOIL_MASS_13MM = 0.005933 # g

FOIL_MASS_MAP = {
    "aluminum foil": AL_FOIL_MASS_13MM,
    "aluminium foil": AL_FOIL_MASS_13MM,
    "al foil": AL_FOIL_MASS_13MM,
    "aluminium foil": AL_FOIL_MASS_13MM,
    "carbon-coated al": HC_FOIL_MASS_13MM,
    "carbon coated al": HC_FOIL_MASS_13MM,
    "carbon-coated aluminum": HC_FOIL_MASS_13MM,
    "carbon coated aluminum": HC_FOIL_MASS_13MM,
    "carbon-coated aluminium": HC_FOIL_MASS_13MM,
    "carbon-coated aluminium": HC_FOIL_MASS_13MM,
    "copper foil": CU_FOIL_MASS_13MM,
    "cu foil": CU_FOIL_MASS_13MM,
}

# Colors used in plots
COLORS = {
    'NaVP': 'blue',
    'LVP': 'green',
    'Li3V2(PO4)3': 'orange',
    'Na3V2(PO4)3': 'red',
    'Graphite': 'gray',
    'Hard Carbon': 'purple',
}

# =========================
# Built-in Historical Dataset (backward compatibility)
# Thickness key units are "10 μm" ticks (as in the original data)
# Values are total electrode mass per 13 mm disk in grams (includes foil)
# =========================
CATHODE_DATA = {
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
    },
    # Add more cathodes if available
}

ANODE_DATA = {
    'Graphite': {
        22.5: np.array([0.02172, 0.02047, 0.02013]),
        32.5: np.array([0.02700, 0.02643, 0.02510]),
        42.5: np.array([0.03184, 0.02984, 0.02953]),
        52.5: np.array([0.03592, 0.0497,  0.03747])
    },
    # Add more anodes if available (e.g., Hard Carbon)
}

# Default active material fractions for capacity match based on historical data
MATERIAL_ACTIVE_RATIOS = {
    'NaVP': 0.96,
    'LVP': 0.96,
    'Li3V2(PO4)3': 0.96,
    'Na3V2(PO4)3': 0.96,
    'Graphite': 0.80,
    'Hard Carbon': 0.90,
}

MATERIAL_LIBRARY = {
    "NaVP": {"type": "Cathode", "active_pct": 96.0, "capacity": 110},
    "LVP": {"type": "Cathode", "active_pct": 96.0, "capacity": 120},
    "Li3V2(PO4)3": {"type": "Cathode", "active_pct": 96.0, "capacity": 120},
    "Na3V2(PO4)3": {"type": "Cathode", "active_pct": 96.0, "capacity": 110},
    "Graphite": {"type": "Anode", "active_pct": 80.0, "capacity": 350},
    "Hard Carbon": {"type": "Anode", "active_pct": 90.0, "capacity": 300},
}

SUBSTRATE_LIBRARY = {
    "Aluminum Foil": {"tare_mg_cm2": 4.163, "mass_13mm": AL_FOIL_MASS_13MM, "type": "Cathode"},
    "Copper Foil": {"tare_mg_cm2": 8.775, "mass_13mm": CU_FOIL_MASS_13MM, "type": "Anode"},
    "Carbon-Coated Al": {"tare_mg_cm2": 4.473, "mass_13mm": HC_FOIL_MASS_13MM, "type": "Cathode"},
}


# =========================
# Utilities
# =========================
def mm_diameter_to_area_cm2(d_mm: float) -> float:
    r_cm = (d_mm / 2.0) / 10.0
    return math.pi * r_cm * r_cm

def pick_foil_mass(substrate_str: str) -> float | None:
    """Improved substrate mass lookup with fuzzy matching"""
    if not substrate_str:
        return None
    
    substrate_lower = str(substrate_str).strip().lower()
    
    # Exact match first
    for key, mass in FOIL_MASS_MAP.items():
        if substrate_lower == key:
            return mass
    
    # Fuzzy matching for common variations
    if any(word in substrate_lower for word in ["aluminum", "aluminium", "al"]):
        if any(word in substrate_lower for word in ["carbon", "coated"]):
            return HC_FOIL_MASS_13MM
        else:
            return AL_FOIL_MASS_13MM
    elif any(word in substrate_lower for word in ["copper", "cu"]):
        return CU_FOIL_MASS_13MM
    
    # Try to extract tare value from substrate name if it contains numbers
    import re
    numbers = re.findall(r'\d+\.?\d*', substrate_lower)
    if numbers:
        try:
            # If substrate name contains a number, assume it's mg/cm²
            tare_mg_cm2 = float(numbers[0])
            area_13mm = mm_diameter_to_area_cm2(13.0)
            return (tare_mg_cm2 / 1000.0) * area_13mm
        except:
            pass
    
    return None

def calc_mass_loading_total_and_active(
    disk_mass_g: float,
    diameter_mm: float,
    substrate_mass_g: float,
    active_pct: float | None,
) -> tuple[float, float]:
    """
    Args:
      disk_mass_g: total electrode mass in grams (disk+substrate)
      diameter_mm: electrode diameter in mm
      substrate_mass_g: substrate tare mass in grams (calculated from mg/cm² input)
      active_pct: % of active material in the coating

    Returns:
      total_ml_mg_cm2: mg/cm² of total coating (foil-subtracted)
      active_ml_mg_cm2: mg/cm² of active-only (if active_pct provided)
    """
    area = mm_diameter_to_area_cm2(diameter_mm)
    if area <= 0 or disk_mass_g is None or np.isnan(disk_mass_g):
        return (np.nan, np.nan)

    # Subtract the foil tare directly
    coat_mass_g = max(disk_mass_g - substrate_mass_g, 0.0)

    total_ml = (coat_mass_g * 1000.0) / area
    active_ml = total_ml * (active_pct / 100.0) if active_pct is not None else np.nan
    return (total_ml, active_ml)

def bytes_excel(df: pd.DataFrame, filename="export.xlsx") -> tuple[bytes, str]:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buf.seek(0)
    return buf.read(), filename

def ensure_db_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure database has all required columns with proper names"""
    cols = [
        "Active Material", "Substrate", "Diameter (mm)", "Mass (g)",
        "Solid %", "Blade Height (µm)", "Active Material %",
        "Mass Loading (mg/cm²)", "Active ML (mg/cm²)", 
        "Date Made", "Made By", "Notes"
    ]
    for c in cols:
        if c not in df.columns:
            if c == "Date Made":
                df[c] = ""
            elif c == "Made By":
                df[c] = ""
            elif c == "Notes":
                df[c] = ""
            else:
                df[c] = np.nan
    return df[cols]

def load_database(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df = ensure_db_columns(df)
    return df

def save_database(df: pd.DataFrame, file_path: str):
    df = ensure_db_columns(df)
    df.to_excel(file_path, index=False)

def mass_to_mg_per_cm2(mass_g, punch_diameter_mm=13):
    radius_cm = (punch_diameter_mm / 10) / 2
    area_cm2 = math.pi * radius_cm**2
    return round((mass_g / area_cm2) * 1000, 3)

substrates_dropdown = [
    (f"Aluminum Foil ({mass_to_mg_per_cm2(AL_FOIL_MASS_13MM)} mg/cm² tare)", AL_FOIL_MASS_13MM),
    (f"Copper Foil ({mass_to_mg_per_cm2(CU_FOIL_MASS_13MM)} mg/cm² tare)", CU_FOIL_MASS_13MM),
    (f"High Carbon Aluminum Foil ({mass_to_mg_per_cm2(HC_FOIL_MASS_13MM)} mg/cm² tare)", HC_FOIL_MASS_13MM)
]

def get_substrate_mass_g(substrate_str: str, diameter_mm: float) -> float:
    """Get substrate mass in grams for given diameter"""
    # First try the pick_foil_mass function (for 13mm)
    mass_13mm = pick_foil_mass(substrate_str)
    if mass_13mm is not None:
        # Scale to actual diameter
        area_13mm = mm_diameter_to_area_cm2(13.0)
        area_actual = mm_diameter_to_area_cm2(diameter_mm)
        return mass_13mm * (area_actual / area_13mm)
    
    # Default fallback
    area_actual = mm_diameter_to_area_cm2(diameter_mm)
    return (4.0 / 1000.0) * area_actual  # 4 mg/cm² default

def plot_regression(x, y, xlabel="", ylabel="", material_label="", color='black',
                    target_y=None, recommended_x=None, x_is_ticks=False, alpha_ci=0.05):
    """
    Plots linear regression with:
      - regression line spanning the full relevant x range
      - confidence interval for the mean 
      - prediction interval for new observations
      - optional horizontal/vertical target lines
      - automatic dynamic scaling to include targets or recommended values
      - origin (0,0) always in view
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    # Fit regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = slope * x + intercept

    # Determine dynamic x-axis range: cover data + recommended_x if given
    x_vals = [x.min(), x.max()]
    if recommended_x is not None:
        x_vals.append(recommended_x)
    x_min, x_max = min(x_vals), max(x_vals)
    
    # Ensure origin (0,0) is always in view
    x_min = min(0, x_min) if x_min > 0 else x_min
    x_range = x_max - x_min
    x_plot = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 500)
    y_plot = slope * x_plot + intercept

    # Standard error of estimate
    residuals = y - y_pred
    se = np.sqrt(np.sum(residuals**2) / (n - 2))
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean)**2)

    # Confidence interval for mean prediction
    se_fit = se * np.sqrt(1/n + (x_plot - x_mean)**2 / ssx)
    t_val = t.ppf(1 - alpha_ci/2, df=n-2)
    ci_upper = y_plot + t_val * se_fit
    ci_lower = y_plot - t_val * se_fit

    # Prediction interval for new observation
    se_pred = se * np.sqrt(1 + 1/n + (x_plot - x_mean)**2 / ssx)
    pi_upper = y_plot + t_val * se_pred
    pi_lower = y_plot - t_val * se_pred

    # Determine dynamic y-axis range: cover data + CI/PI + target_y if given
    y_vals = [y.min(), y.max(), ci_lower.min(), ci_upper.max(), pi_lower.min(), pi_upper.max()]
    if target_y is not None:
        y_vals.append(target_y)
    y_min, y_max = min(y_vals), max(y_vals)
    
    # Ensure origin (0,0) is always in view
    y_min = min(0, y_min) if y_min > 0 else y_min
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    # Plotting
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, color=color, label=f"Data: {material_label}")
    plt.plot(x_plot, y_plot, color='red', label=f"Linear Fit (R²={r_value**2:.3f})")
    plt.fill_between(x_plot, ci_lower, ci_upper, color='red', alpha=0.2, label=f'{int((1-alpha_ci)*100)}% CI')
    plt.fill_between(x_plot, pi_lower, pi_upper, color='orange', alpha=0.2, label=f'{int((1-alpha_ci)*100)}% PI')

    # Target lines
    if target_y is not None:
        plt.axhline(target_y, color='green', linestyle='--', label=f"Target ML = {target_y:.2f}")
    if recommended_x is not None:
        plt.axvline(recommended_x, color='blue', linestyle='--', label=f"Recommended X = {recommended_x:.2f}")

    plt.xlim(x_min, x_max + 0.05*x_range)
    plt.ylim(y_min, y_max)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}" + (f" — {material_label}" if material_label else ""))
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

    return slope, intercept, r_value**2


# =========================
# Sidebar: DB file selection
# =========================
st.sidebar.header("Settings")
db_file = st.sidebar.text_input("Database file", value=DEFAULT_DB_FILE)
if st.sidebar.button("Create empty DB if missing"):
    if not os.path.exists(db_file):
        save_database(pd.DataFrame(), db_file)
    st.sidebar.success("Ready.")

# =========================
# Tool Switcher
# =========================
tool = st.sidebar.radio(
    "Select Tool",
    options=["Slurry Calculator", "Blade Height Recommender", "Capacity Match Tool", "Coating Calibration Tool", "Electrode Database Manager"],     format_func=lambda x: x,
    index=0
)

# =========================
# Slurry Calculator
# =========================

def slurry_calculator():

    # Initialize session_state variables
    if "recipe" not in st.session_state:
        st.session_state.recipe = {}
    if "dilutions" not in st.session_state:
        st.session_state.dilutions = []

    st.title("Slurry Calculator")
    st.caption("Plan your cathode/anode slurry with multi-component tracking and dilution history.")

    # ===== Formula & Mass Targets =====
    with st.expander("1️⃣ Formula & Mass Targets", expanded=True):
        st.subheader("Mass Ratios (%)")
        active_ratio = st.number_input("Active Material %", 0.0, 100.0, 96.0, 0.1)
        carbon_ratio = st.number_input("Conductive Carbon %", 0.0, 100.0, 2.0, 0.1)
        binder_ratio = st.number_input("Binder %", 0.0, 100.0, 2.0, 0.1)

        total_ratio = active_ratio + carbon_ratio + binder_ratio
        if abs(total_ratio - 100.0) > 0.01:
            st.error(f"Mass ratio must sum to 100%. Current: {total_ratio:.2f}%")
            st.stop()

        target_mode = st.radio("Target Mode", ["Active Mass (g)", "Total Slurry Mass (g)"])
        if target_mode == "Active Mass (g)":
            active_mass = st.number_input("Active Material Mass (g)", 0.0, step=0.00001, value=1.0)
        else:
            total_slurry_mass = st.number_input("Total Slurry Mass (g)", 0.0, step=0.00001, value=10.0)

        use_solution = st.checkbox("Using Binder Solution?", True)
        binder_solution_pct = st.number_input("Binder % in Solution", 0.1, 100.0, 5.0) if use_solution else None
        solid_pct = st.number_input("Target Solid Content (%)", 0.1, 100.0, 30.0)

    # ===== Component Breakdown =====
    def get_components(name, default_names, default_weights):
        multi = st.checkbox(f"{name} has multiple components?", value=False, key=f"{name}_multi")
        n = st.number_input(f"Number of {name} components", 1, 10,
                            value=len(default_names), key=f"{name}_count") if multi else 1
        names, weights = [], []
        for i in range(n):
            nm = st.text_input(f"{name} #{i+1} name",
                            value=default_names[i] if i < len(default_names) else "",
                            key=f"{name}_{i}_nm")
            wt = st.number_input(f"{name} #{i+1} %",
                                0.0, 100.0,
                                value=default_weights[i] if i < len(default_weights) else 100.0/n,
                                step=0.1, key=f"{name}_{i}_wt")
            names.append(nm)
            weights.append(wt)
        if abs(sum(weights) - 100.0) > 0.01:
            st.error(f"{name} splits must total 100%. Current: {sum(weights):.2f}%")
            return None, None
        return names, weights

    with st.expander("2️⃣ Component Breakdown", expanded=False):
        active_names, active_weights = get_components("Active", ["LVP"], [100.0])
        carbon_names, carbon_weights = get_components("Carbon", ["Super P"], [100.0])
        binder_names, binder_weights = get_components("Binder", ["PVDF"], [100.0])
        solvent_names, solvent_weights = get_components("Solvent", ["NMP"], [100.0])
        if None in (active_names, carbon_names, binder_names, solvent_names):
            st.stop()

    # ===== Slurry Recipe & Masses =====
    with st.expander("3️⃣ Slurry Recipe & Masses", expanded=True):
        # Calculate masses
        active_frac, carbon_frac, binder_frac = active_ratio/100, carbon_ratio/100, binder_ratio/100
        if target_mode == "Active Mass (g)":
            carbon_mass = (carbon_frac / active_frac) * active_mass if active_frac > 0 else 0
            binder_mass = (binder_frac / active_frac) * active_mass if active_frac > 0 else 0
            total_solids = active_mass + carbon_mass + binder_mass
            total_slurry_mass = total_solids / (solid_pct / 100.0)
        else:
            total_solids = total_slurry_mass * (solid_pct / 100.0)
            active_mass = (active_frac / (active_frac + carbon_frac + binder_frac)) * total_solids
            carbon_mass = (carbon_frac / active_frac) * active_mass if active_frac > 0 else 0
            binder_mass = total_solids - active_mass - carbon_mass

        binder_solution_mass = binder_mass / (binder_solution_pct/100.0) if use_solution else binder_mass
        solvent_in_binder = binder_solution_mass - binder_mass if use_solution else 0
        solvent_pure_mass = total_slurry_mass - total_solids - solvent_in_binder
        solvent_total_combined = solvent_pure_mass + solvent_in_binder

        # Store recipe in session_state
        st.session_state.recipe = {
            "Active Mass (g)": active_mass,
            "Carbon Mass (g)": carbon_mass,
            "Binder Mass (g)": binder_mass,
            "Binder Solution Mass (g)": binder_solution_mass if use_solution else None,
            "Solvent Mass Total (g)": solvent_total_combined,
            "Solvent Pure (g)": solvent_pure_mass,
            "Solvent in Binder Solution (g)": solvent_in_binder,
            "Total Solids (g)": total_solids,
            "Total Slurry Mass (g)": total_slurry_mass,
        }

        # Display recipe
        df_recipe = pd.DataFrame.from_dict(
            {k: round(v, 4) for k, v in st.session_state.recipe.items() if v is not None},
            orient="index", columns=["Mass (g)"]
        )
        st.table(df_recipe)

    # ===== Slurry Dilution Tracker =====
    with st.expander("4️⃣ Slurry Dilution Tracker", expanded=True):
        recipe = st.session_state.recipe
        total_solids = recipe["Total Solids (g)"]
        original_mass = recipe["Total Slurry Mass (g)"]

        # Calculate current slurry mass including any previous dilutions
        current_mass = original_mass + sum(st.session_state.dilutions)
        current_solid_pct = (total_solids / current_mass) * 100
        st.info(f"Current slurry mass = {current_mass:.4f} g, Solid % = {current_solid_pct:.2f}%")

        target_solid = st.number_input("Target Solid Content (%) for Dilution", 0.1, 100.0, value=current_solid_pct, step=0.1)

        if st.button("Calculate Solvent Addition"):
            if target_solid < current_solid_pct:
                # Calculate the amount of solvent needed
                new_total_mass = total_solids / (target_solid / 100)
                added_solvent = new_total_mass - current_mass

                st.session_state.dilutions.append(added_solvent)

                # Update current mass and solid % immediately for display
                current_mass += added_solvent
                current_solid_pct = (total_solids / current_mass) * 100

                st.success(f"Add {added_solvent:.4f} g solvent to reach {target_solid:.2f}% solids.")
                st.info(f"New slurry mass = {current_mass:.4f} g, Solid % = {current_solid_pct:.2f}%")
            else:
                st.warning("Target solid % must be lower than current solid %.")

        # Display dilution history
        if st.session_state.dilutions:
            st.subheader("Dilution History")
            for i, amt in enumerate(st.session_state.dilutions, 1):
                st.write(f"Dilution {i}: {amt:.4f} g")
            total_added = sum(st.session_state.dilutions)
            st.markdown(f"**Total Solvent Added:** {total_added:.4f} g")
            st.markdown(f"**New Solid Content:** {(total_solids / (original_mass + total_added))*100:.2f}%")
            
    # ---- Export to Excel ----
    if st.button("Export Recipe to Excel"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Original recipe
            df_recipe.to_excel(writer, sheet_name="Recipe")
            
            # True recipe (if tracked)
            if track_true:
                df_actual = pd.DataFrame.from_dict({k: [v] for k,v in true_recipe.items()})
                df_actual.to_excel(writer, sheet_name="Actual Recipe")
            
            # Dilution history
            if st.session_state.dilution_history:
                dilution_data = {
                    f"Dilution {i+1}": [v] for i, v in enumerate(st.session_state.dilution_history)
                }
                df_dilution = pd.DataFrame(dilution_data)
                df_dilution.to_excel(writer, sheet_name="Dilution History")

            # Create a sheet with total solvent, total solids, and cumulative dilution
            combined_data = {
                "Total Solids (g)": [total_solids],
                "Total Slurry Mass (g)": [total_slurry_mass],
                "Initial Solvent Mass (g)": [solvent_total_combined],
                "Cumulative Added Solvent (g)": [sum(st.session_state.dilution_history)]
            }
            df_combined = pd.DataFrame(combined_data)
            df_combined.to_excel(writer, sheet_name="Combined Overview")

        st.download_button(
            label="Download Excel",
            data=buffer.getvalue(),
            file_name="slurry_recipe.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        
    # ---- Slurry Calculation Details Expander ----
    with st.expander("📘 Slurry Calculation Details"):
        st.markdown("### Slurry Composition Formulas")
        st.markdown(f"""
        **Given:**
        - Active Material %: {active_ratio}%
        - Carbon %: {carbon_ratio}%
        - Binder %: {binder_ratio}%
        - Active Material Mass: {active_mass:.4f} g
        - Binder Solution %: {binder_solution_pct if use_solution else 'N/A'}% ({'used' if use_solution else 'not used'})
        - Target Solid %: {solid_pct}%
        """)

        st.markdown("**Calculations:**")
        st.markdown(f"""
        1. Carbon Mass = (Carbon % / Active %) × Active Mass  
        = ({carbon_ratio} / {active_ratio}) × {active_mass:.4f}  
        = {carbon_mass:.4f} g
        
        2. Binder Mass = (Binder % / Active %) × Active Mass  
        = ({binder_ratio} / {active_ratio}) × {active_mass:.4f}  
        = {binder_mass:.4f} g
        
        3. Binder Solution Mass = {'Binder Mass / (Binder Solution % / 100)' if use_solution else 'N/A'}  
        = {binder_solution_mass:.4f} g
    
        4. Total Solids = Active + Carbon + Binder  
        = {active_mass:.4f} + {carbon_mass:.4f} + {binder_mass:.4f}  
        = {total_solids:.4f} g
        
        5. Total Slurry Mass = Total Solids / (Solid % / 100)  
        = {total_solids:.4f} / ({solid_pct}/100)  
        = {total_slurry_mass:.4f} g
        
        6. Solvent Mass = Total Slurry Mass - Total Solids  
        = {total_slurry_mass:.4f} - {total_solids:.4f}  
        = {solvent_total_combined:.4f} g
        """)

        if use_solution and binder_weights:
            st.markdown("**Binder Composition Breakdown:**")
            for name, wt in zip(binder_names, binder_weights):
                bm = binder_mass * (wt / 100.0)
                st.markdown(f"""
                - {name}: {wt}% of binder → {bm:.4f} g  
                Calculation: {binder_mass:.4f} × ({wt}/100) = {bm:.4f} g
                """)


# =========================
# Blade Height Recommender (DB-aware)
# =========================

def blade_height_recommender():
    st.title("📐 Blade Height Recommender")
    st.caption("Fit mass loading vs blade height from built-in data or your database.")
    db = load_database(db_file)

    # Data source selector
    src = st.radio("Data source", ["Use Database", "Use Built-in Dataset"], index=0, horizontal=True)

    if src == "Use Database":
        if db.empty:
            st.warning("📭 Database is empty. Add some electrodes in the Database Manager first!")
            return

        st.subheader("Select Material & Filters")
        
        # Step 1: Material selection with preview
        col1, col2 = st.columns(2)
        
        with col1:
            available_materials = sorted([m for m in db["Active Material"].dropna().unique() if str(m).strip()])
            if not available_materials:
                st.error("No materials found in database.")
                return
                
            material = st.selectbox("Active Material", available_materials)
            
            # Show quick preview
            material_data = db[db["Active Material"] == material]
            st.info(f"📊 Found **{len(material_data)}** electrodes with {material}")
        
        with col2:
            # Step 2: Substrate selection
            available_substrates = sorted([s for s in material_data["Substrate"].dropna().unique() if str(s).strip()])
            if not available_substrates:
                st.error("No substrates found for this material.")
                return
                
            substrate = st.selectbox("🏗️ Substrate", available_substrates)
            
            # Show substrate preview
            substrate_data = material_data[material_data["Substrate"] == substrate]
            st.info(f"📋 Found **{len(substrate_data)}** with this substrate")

        # Step 3: Quality filters
        st.subheader("Quality Filters")
        col3, col4 = st.columns(2)
        
        with col3:
            active_pct_min = st.slider("Minimum Active Material %", 0.0, 100.0, 90.0, 1.0)
        with col4:
            solid_pct_min = st.slider("Minimum Solid %", 0.0, 100.0, 20.0, 5.0)

        # Auto-calculate missing mass loadings
        db_work = db.copy()
        missing_ml = db_work["Mass Loading (mg/cm²)"].isna() | db_work["Active ML (mg/cm²)"].isna()
        
        if missing_ml.any():
            st.info("🔄 Auto-calculating missing mass loadings...")
            for idx in db_work.index[missing_ml]:
                row = db_work.loc[idx]
                substrate_mass_g = get_substrate_mass_g(
                    row.get("Substrate", ""), 
                    row.get("Diameter (mm)", 13.0)
                )
                
                total_ml, active_ml = calc_mass_loading_total_and_active(
                    disk_mass_g=row.get("Mass (g)", np.nan),
                    diameter_mm=row.get("Diameter (mm)", 13.0),
                    substrate_mass_g=substrate_mass_g,
                    active_pct=row.get("Active Material %", np.nan),
                )
                db_work.at[idx, "Mass Loading (mg/cm²)"] = total_ml
                db_work.at[idx, "Active ML (mg/cm²)"] = active_ml

        # Filter data
        df_filtered = db_work[
            (db_work["Active Material"] == material) &
            (db_work["Substrate"] == substrate) &
            (db_work["Active Material %"].fillna(0) >= active_pct_min) &
            (db_work["Solid %"].fillna(0) >= solid_pct_min) &
            (db_work["Blade Height (µm)"].notna()) &
            (db_work["Active ML (mg/cm²)"].notna()) &
            (db_work["Active ML (mg/cm²)"] > 0) &
            (db_work["Blade Height (µm)"] > 0)
        ].copy()

        if df_filtered.empty:
            st.error("❌ No data points meet your criteria. Try relaxing the filters.")
            return
        
        if len(df_filtered) < 3:
            st.warning(f"⚠️ Only {len(df_filtered)} data points found. Need at least 3 for reliable regression.")
            if len(df_filtered) < 2:
                return

        # Show filtered data preview
        with st.expander(f"📋 Preview Filtered Data ({len(df_filtered)} points)"):
            display_cols = ["Blade Height (µm)", "Active ML (mg/cm²)", "Solid %", "Active Material %", "Date Made"]
            st.dataframe(df_filtered[display_cols].round(3))

        # Regression analysis
        x_ticks = df_filtered["Blade Height (µm)"].astype(float) / 10.0  # Convert to 10µm units
        y_ml = df_filtered["Active ML (mg/cm²)"].astype(float)

        # Target mass loading input
        st.subheader("🎯 Target Mass Loading")
        median_ml = float(y_ml.median())
        target_loading = st.number_input(
            "Target Active Mass Loading (mg/cm²)", 
            min_value=0.0, 
            step=0.1,
            value=median_ml,
            help=f"Current data range: {y_ml.min():.2f} - {y_ml.max():.2f} mg/cm²"
        )

        # Perform regression
        slope, intercept, r_value, _, std_err = linregress(x_ticks, y_ml)
        required_tick = (target_loading - intercept) / slope if slope != 0 else None
        required_um = required_tick * 10.0 if required_tick is not None else None

        # Confidence assessment
        min_m, max_m = y_ml.min(), y_ml.max()
        if min_m <= target_loading <= max_m:
            confidence = "High ✅"
            conf_color = "green"
        elif target_loading >= min_m * 0.8 and target_loading <= max_m * 1.2:
            confidence = "Medium ⚠️"
            conf_color = "orange"
        else:
            confidence = "Low ❌"
            conf_color = "red"

        # Plot regression
        color = COLORS.get(material, 'black')
        plot_regression(
            x_ticks.values, y_ml.values,
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=f"{material} on {substrate}",
            color=color,
            target_y=target_loading,
            recommended_x=required_tick,
            x_is_ticks=True
        )

        # Results
        if required_um is not None and np.isfinite(required_um):
            st.success(f"🎯 **Recommended Blade Height: {required_um:.1f} µm**")
            st.markdown(f"**Confidence Level:** :{conf_color}[{confidence}]")
            
            if confidence.startswith("Low"):
                st.warning("⚠️ Target is outside your data range. Consider adding more data points or adjusting target.")
        else:
            st.error("❌ Cannot calculate recommendation (invalid slope).")

        # Regression details
        with st.expander("📊 Regression Statistics"):
            st.markdown(f"""
            - **Slope:** {slope:.4f} mg/cm² per 10µm
            - **Intercept:** {intercept:.4f} mg/cm²  
            - **R²:** {r_value**2:.4f}
            - **Data Points:** {len(df_filtered)}
            - **Blade Height Range:** {x_ticks.min()*10:.0f} - {x_ticks.max()*10:.0f} µm
            """)

    else:
        # Keep the existing built-in dataset code unchanged
        st.info("Using built-in historical dataset.")
        side = st.selectbox("Side", ["Cathode", "Anode"])
        if side == "Cathode":
            material = st.selectbox("Material", list(CATHODE_DATA.keys()))
            samples = CATHODE_DATA[material]
            foil_mass = AL_FOIL_MASS_13MM
        else:
            material = st.selectbox("Material", list(ANODE_DATA.keys()))
            samples = ANODE_DATA[material]
            foil_mass = CU_FOIL_MASS_13MM

        active_ratio = st.number_input("Active Material % in Formula", 0.0, 100.0, value=96.0, step=0.1)
        thicknesses, mass_loadings = [], []

        for thickness, masses in samples.items():
            avg_m = np.mean(masses)
            ml = (avg_m - foil_mass) * (active_ratio / 100.0) / ELECTRODE_AREA_13MM * 1000.0
            thicknesses.append(thickness)
            mass_loadings.append(ml)

        target_loading = st.number_input("Target Active Mass Loading (mg/cm²)", 0.0, 100.0, value=2.0, step=0.1)
        slope, intercept, r_value, _, std_err = linregress(thicknesses, mass_loadings)
        required_thickness = (target_loading - intercept) / slope if slope != 0 else None

        color = COLORS.get(material, 'black')
        plot_regression(
            np.array(thicknesses), np.array(mass_loadings),
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=material,
            color=color,
            target_y=target_loading,
            recommended_x=required_thickness,
            x_is_ticks=True
        )

        if required_thickness is not None and np.isfinite(required_thickness):
            st.success(f"Recommended Blade Height: **{required_thickness*10.0:.1f} μm**  "
                       f"(= {required_thickness:.2f} × 10 μm)")
        else:
            st.error("Slope is zero or invalid. Cannot compute recommendation.")
            
# =========================
# Capacity Match Tool (DB-aware)
# =========================

def capacity_match_tool():
    st.title("🔋 CapMatch - Capacity Match Tool")
    st.caption("Match cathode/anode areal capacity and estimate blade height using built-in data or your database.")

    db = load_database(db_file)
    side_known = st.selectbox("Known side", ["Anode", "Cathode"])
    side_target = "Cathode" if side_known == "Anode" else "Anode"

    # Known inputs
    known_material = st.text_input(f"{side_known} material (e.g., LVP, Graphite, ...)", value="LVP" if side_known=="Cathode" else "Graphite")
    known_active_ratio = MATERIAL_ACTIVE_RATIOS.get(known_material, 0.96 if side_known=="Cathode" else 0.80)
    known_ml = st.number_input(f"{side_known} Active Mass Loading (mg/cm²)", 0.0, 100.0, value=2.0, step=0.01)
    known_specific_capacity = st.number_input(f"{side_known} Specific Capacity (mAh/g)", 0.0, 1000.0, value=100.0, step=0.1)

    # Ratio
    if side_target == "Anode":
        capacity_ratio = st.number_input("Anode/Cathode ratio (N/P)", 0.1, 5.0, value=1.0, step=0.01, format="%.3f")
    else:
        capacity_ratio = st.number_input("Cathode/Anode ratio (P/N)", 0.1, 5.0, value=1.0, step=0.01, format="%.3f")

    # Target side properties
    target_material = st.text_input(f"{side_target} material", value="Graphite" if side_target=="Anode" else "LVP")
    target_active_ratio = MATERIAL_ACTIVE_RATIOS.get(target_material, 0.96 if side_target=="Cathode" else 0.80)
    target_specific_capacity = st.number_input(f"{side_target} Specific Capacity (mAh/g)", 0.0, 1000.0, value=350.0 if side_target=="Anode" else 120.0, step=0.1)

    # Areal capacities
    known_areal_capacity = known_ml * known_specific_capacity / 1000.0
    target_areal_capacity = known_areal_capacity * capacity_ratio
    required_target_ml = target_areal_capacity / target_specific_capacity * 1000.0

    st.subheader("Areal capacity results")
    st.markdown(f"- Known areal capacity: **{known_areal_capacity:.3f} mAh/cm²**")
    st.markdown(f"- Target areal capacity: **{target_areal_capacity:.3f} mAh/cm²**")
    st.markdown(f"- Required {side_target} active ML: **{required_target_ml:.3f} mg/cm²**")

    # Now estimate blade height for the target
    st.subheader("Estimate target blade height from dataset")
    src = st.radio("Data source", ["Use Database", "Use Built-in Dataset"], index=0, horizontal=True)

    if src == "Use Database":
        if db.empty:
            st.warning("Database empty. Use Electrode Database Manager to add data.")
            return

        # Simplified material selection
        available_materials = [m for m in db["Active Material"].dropna().unique() if str(m).strip().lower() == target_material.strip().lower()]
        
        if not available_materials:
            st.error(f"Material '{target_material}' not found in database. Available materials: {list(db['Active Material'].dropna().unique())}")
            return
            
        # Auto-calculate missing mass loadings
        db_work = db.copy()
        missing_ml = db_work["Active ML (mg/cm²)"].isna()
        if missing_ml.any():
            for idx in db_work.index[missing_ml]:
                row = db_work.loc[idx]
                substrate_mass_g = get_substrate_mass_g(
                    row.get("Substrate", ""), 
                    row.get("Diameter (mm)", 13.0)
                )
                
                total_ml, active_ml = calc_mass_loading_total_and_active(
                    disk_mass_g=row.get("Mass (g)", np.nan),
                    diameter_mm=row.get("Diameter (mm)", 13.0),
                    substrate_mass_g=substrate_mass_g,
                    active_pct=row.get("Active Material %", np.nan),
                )
                db_work.at[idx, "Active ML (mg/cm²)"] = active_ml

        # Filter for target material
        df = db_work[
            (db_work["Active Material"].str.lower() == target_material.lower()) &
            (db_work["Blade Height (µm)"].notna()) &
            (db_work["Active ML (mg/cm²)"].notna()) &
            (db_work["Active ML (mg/cm²)"] > 0)
        ].copy()

        if df.empty:
            st.error("No usable data for this material in database.")
            return

        st.info(f"Found {len(df)} data points for {target_material}")

        x_ticks = df["Blade Height (µm)"].astype(float) / 10.0
        y_active_ml = df["Active ML (mg/cm²)"].astype(float)
        
        slope, intercept, r_value, _, _ = linregress(x_ticks, y_active_ml)
        req_tick = (required_target_ml - intercept) / slope if slope != 0 else None
        req_um = req_tick * 10.0 if req_tick is not None else None

        color = COLORS.get(target_material, 'black')
        plot_regression(
            x_ticks.values, y_active_ml.values,
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=f"{target_material} (DB)", color=color,
            target_y=required_target_ml, recommended_x=req_tick, x_is_ticks=True
        )

        if req_um is not None and np.isfinite(req_um):
            st.success(f"Recommended {side_target} blade height: **{req_um:.1f} μm**")
        else:
            st.error("Regression invalid. Cannot recommend height.")

    else:
        # Keep the existing built-in data code unchanged
        if side_target == "Cathode":
            if target_material not in CATHODE_DATA:
                st.error("Target material not found in built-in cathode dataset.")
                return
            samples = CATHODE_DATA[target_material]
            foil_mass = AL_FOIL_MASS_13MM
        else:
            if target_material not in ANODE_DATA:
                st.error("Target material not found in built-in anode dataset.")
                return
            samples = ANODE_DATA[target_material]
            foil_mass = CU_FOIL_MASS_13MM

        thicknesses, active_mls = [], []
        for t, masses in samples.items():
            avg_m = np.mean(masses)
            active_ml = (avg_m - foil_mass) * (target_active_ratio) / ELECTRODE_AREA_13MM * 1000.0
            thicknesses.append(t)
            active_mls.append(active_ml)

        slope, intercept, *_ = linregress(thicknesses, active_mls)
        req_tick = (required_target_ml - intercept) / slope if slope != 0 else None
        color = COLORS.get(target_material, 'black')
        plot_regression(
            np.array(thicknesses), np.array(active_mls),
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=target_material, color=color,
            target_y=required_target_ml, recommended_x=req_tick, x_is_ticks=True
        )
        if req_tick is not None and np.isfinite(req_tick):
            st.success(f"Recommended {side_target} blade height: **{req_tick*10.0:.1f} μm**")
        else:
            st.error("Regression invalid. Cannot recommend height.")

    with st.expander("📘 Capacity Matching Calculation Details"):
        st.markdown("### Capacity Matching Formulas")
        st.markdown(f"""
        **Known Electrode ({side_known}):**
        - Material: {known_material}
        - Active Ratio: {known_active_ratio:.2f}
        - Mass Loading: {known_ml:.2f} mg/cm²
        - Specific Capacity: {known_specific_capacity:.1f} mAh/g
        
        **Target Electrode ({side_target}):**
        - Material: {target_material}
        - Active Ratio: {target_active_ratio:.2f}
        - Specific Capacity: {target_specific_capacity:.1f} mAh/g
        - Capacity Ratio: {capacity_ratio:.2f} ({'N/P' if side_target=='Anode' else 'P/N'})
        """)
        
        st.markdown("**Calculations:**")
        st.markdown(f"""
        1. Known Areal Capacity = (ML × Active Ratio × Specific Capacity) / 1000  
           = ({known_ml:.2f} × {known_active_ratio:.2f} × {known_specific_capacity:.1f}) / 1000  
           = {known_areal_capacity:.3f} mAh/cm²
        
        2. Target Areal Capacity = Known Areal Capacity × Ratio  
           = {known_areal_capacity:.3f} × {capacity_ratio:.2f}  
           = {target_areal_capacity:.3f} mAh/cm²
        
        3. Required Target ML = (Target Areal Capacity × 1000) / (Active Ratio × Specific Capacity)  
           = ({target_areal_capacity:.3f} × 1000) / ({target_active_ratio:.2f} × {target_specific_capacity:.1f})  
           = {required_target_ml:.3f} mg/cm²
        """)


# =========================
# Coating Calibration Tool
# =========================

def coating_calibration_tool():
    st.title("⚙️ Coating Calibration Tool")
    st.markdown("Create a custom coating matrix to visualize coating uniformity.")

    # --- Grid size ---
    n_rows = st.number_input("Number of rows", 1, 10, 3, step=1)
    n_cols = st.number_input("Number of columns", 1, 10, 3, step=1)

    # --- Globals ---
    with st.expander("Global Parameters", expanded=True):
        g_active = st.text_input("Active Material", "LVP", help="Name of the active material used.")
        g_substrate = st.selectbox("Substrate", [s[0] for s in substrates_dropdown], index=0)
        default_foil_mass = pick_foil_mass(g_substrate)

        use_custom_tare = st.checkbox("Override Tare?", value=False)
        if use_custom_tare:
            g_custom_tare = st.number_input(
                "Custom Tare (mg/cm²)", min_value=0.0, value=float(default_foil_mass), step=0.01,
                help="Optionally override the substrate mass per area."
            )
        else:
            g_custom_tare = default_foil_mass

        g_diameter = st.number_input("Diameter (mm)", min_value=0.1, max_value=100.0, value=13.0, step=0.01)
        g_solid = st.number_input("Solid (%)", 0.0, 100.0, 30.0, step=0.00001, format="%.5f")
        g_active_pct = st.number_input(
            "Active Material %", min_value=0.0, max_value=100.0,
            value=MATERIAL_ACTIVE_RATIOS.get(g_active, 96.0), step=0.1
        )
        g_blade = st.number_input("Blade Height (µm)", 1, 1000, 200, step=1)

    heatmap_mode = st.radio(
        "Select Heatmap Type",
        ["Deviation", "Critical Deviation", "Spec/Out-of-Spec"],
        help="Deviation = % from mean; Critical = flag |deviation| > threshold; Spec = inside range."
    )

    if heatmap_mode == "Critical Deviation":
        crit_value = st.number_input("Critical Deviation (%)", 0.0, 100.0, 5.0, step=0.1)
    elif heatmap_mode == "Spec/Out-of-Spec":
        spec_min = st.number_input("Spec Min Active ML (mg/cm²)", 0.0, 50.0, 1.5, step=0.1, format="%.5f")
        spec_max = st.number_input("Spec Max Active ML (mg/cm²)", 0.0, 100.0, 50.0, step=0.1, format="%.5f")

    # --- Matrix inputs ---
    matrix = {}
    for row in range(n_rows):
        cols_layout = st.columns(n_cols + 1)
        with cols_layout[0]:
            st.markdown(f"**Row {row+1}**")
        for col in range(n_cols):
            with cols_layout[col + 1]:
                mass = st.number_input(
                    "Measured Mass (g)", 0.0, 10.0, 0.02000, step=0.00001, format="%.5f",
                    key=f"m_{row}_{col}"
                )
                radius_cm = g_diameter / 10.0 / 2.0
                area_cm2 = 3.14159265 * radius_cm**2
                total_ml = (mass * 1000.0) / area_cm2  # mg/cm^2
                active_ml = total_ml * (g_active_pct)

                data_flag = "MISSING MASS" if mass <= 0 else ""

                matrix[(row, col)] = {
                    'active': g_active,
                    'substrate': g_substrate,
                    'diameter': g_diameter,
                    'solid': g_solid,
                    'active_pct': g_active_pct,
                    'blade': g_blade,
                    'mass': mass,
                    'total_ml': total_ml,
                    'active_ml': active_ml,
                    'flag': data_flag
                }

    # --- Analysis (auto) ---
    ml_values = np.zeros((n_rows, n_cols), dtype=float)
    for (r, c), d in matrix.items():
        ml_values[r, c] = d['active_ml'] if not np.isnan(d['active_ml']) else 0.0

    valid_mask = ml_values > 0.0
    if np.any(valid_mask):
        mean_ml = float(np.nanmean(ml_values[valid_mask]))
        deviations = ((ml_values - mean_ml) / mean_ml) * 100.0
    else:
        mean_ml = 0.0
        deviations = np.zeros_like(ml_values)

    row_labels = [f"Row {r+1}" for r in range(n_rows)]
    col_labels = [f"Col {c+1}" for c in range(n_cols)]

    # Update flags
    for (r, c), d in matrix.items():
        if heatmap_mode == "Critical Deviation" and abs(deviations[r, c]) > crit_value:
            d['flag'] = (d['flag'] + "; CRITICAL DEV") if d['flag'] else "CRITICAL DEV"
        elif heatmap_mode == "Spec/Out-of-Spec":
            if not (spec_min <= ml_values[r, c] <= spec_max):
                d['flag'] = (d['flag'] + "; OUT OF SPEC") if d['flag'] else "OUT OF SPEC"

    # --- Heatmap with labels + colorbar ---
    st.subheader("Heatmap Plot")

    hover_text = [
        [f"ML={ml_values[r,c]:.2f} mg/cm²<br>Deviation={deviations[r,c]:.2f}%"
         for c in range(n_cols)]
        for r in range(n_rows)
    ]

    if heatmap_mode == "Deviation":
        fig_heat = go.Figure(data=go.Heatmap(
            z=deviations,
            x=col_labels, y=row_labels,
            text=hover_text, hovertemplate="%{text}<extra></extra>",
            # colorscale='portland', ## Alternative colorscale test
            colorscale = [
                [0, 'red'],
                [0.25, 'orange'],
                [0.5, 'limegreen'],
                [0.75, 'orange'],
                [1, 'red']
            ],
            zmid=0.0,
            colorbar=dict(title="% Deviation from Mean")
        ))
        # Percent annotations
        for r in range(n_rows):
            for c in range(n_cols):
                fig_heat.add_annotation(
                    x=col_labels[c], y=row_labels[r],
                    text=f"{deviations[r,c]:.1f}%",
                    showarrow=False, font=dict(color="black", size=15)
                )
        st.plotly_chart(fig_heat, use_container_width=True)

    elif heatmap_mode == "Critical Deviation":
        crit_status = np.where(np.abs(deviations) > crit_value, 1, 0)  # 1 = critical, 0 = OK

        # Binary colorscale: 0 -> green (OK), 1 -> red (Critical)
        colorscale = [[0, 'limegreen'], [1, 'red']]

        fig_crit = go.Figure(data=go.Heatmap(
            z=crit_status,
            x=col_labels,
            y=row_labels,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Critical Status", tickvals=[0,1], ticktext=["OK","Critical"])
        ))

        # Large bold O/X annotations with automatic font color
        for r in range(n_rows):
            for c in range(n_cols):
                cell_value = crit_status[r,c]
                fig_crit.add_annotation(
                    x=col_labels[c],
                    y=row_labels[r],
                    text=("X" if cell_value == 1 else "O"),
                    showarrow=False,
                    font=dict(
                        color="white" if cell_value == 1 else "black",
                        size=24,
                        family="Arial Black" 
                    )
                )

        st.plotly_chart(fig_crit, use_container_width=True)


    else:  # Spec/Out-of-Spec
        # 1 = In-spec (green), 0 = Out-of-spec (red)
        spec_map = np.where((ml_values >= spec_min) & (ml_values <= spec_max), 1, 0).astype(float)

        # Binary colorscale: 0 -> red (Out-of-spec), 1 -> green (In-spec)
        colorscale = [[0, 'red'], [1, 'limegreen']]

        fig_spec = go.Figure(data=go.Heatmap(
            z=spec_map,
            x=col_labels,
            y=row_labels,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Spec Status", tickvals=[0,1], ticktext=["Out","In"])
        ))

        # Large bold O/X annotations with automatic font color
        for r in range(n_rows):
            for c in range(n_cols):
                cell_value = spec_map[r,c]
                fig_spec.add_annotation(
                    x=col_labels[c],
                    y=row_labels[r],
                    text=("O" if cell_value == 1 else "X"),
                    showarrow=False,
                    font=dict(
                        color="white" if cell_value == 0 else "black",  # contrast with background
                        size=24,
                        family="Arial Black" 
                    )
                )

        st.plotly_chart(fig_spec, use_container_width=True)

    # --- 3D Plate Representation ---
    st.subheader("3D Plate Representation")

    plate_width = 101.6  # mm
    plate_height = 101.6  # mm
    radius = g_diameter / 2.0

    x_centers = np.linspace(0.0, plate_width, n_cols)
    y_centers = np.linspace(0.0, plate_height, n_rows)

    X, Y = np.meshgrid(x_centers, y_centers)
    Z = ml_values

    # Z range for colorbar
    z_min = float(np.nanmin(Z)) if np.isfinite(np.nanmin(Z)) else 0.0
    z_max = float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else 1.0
    if z_max == z_min:
        z_min -= 1e-6
        z_max += 1e-6
    levels = np.linspace(z_min, z_max, 12)

    # Z exaggeration slider (affects geometry only), you can edit the value here
    z_exaggeration = 1

    Z_exagg = Z * z_exaggeration  # for geometry

    fig_3d = go.Figure()

    # Surface (geometry scaled, color based on actual Z)
    fig_3d.add_trace(go.Surface(
        x=X, y=Y, z=Z_exagg,
        surfacecolor=Z,            # use original data for color
        colorscale="portland",
        cmin=z_min, cmax=z_max,
        opacity=0.85,
        colorbar=dict(
            title="Active ML (mg/cm²)",
            tickvals=levels,
            ticktext=[f"{v:.2f}" for v in levels]
        ),
        contours=dict(
            z=dict(
                show=True,
                start=z_min,
                end=z_max,
                size=(z_max - z_min)/12,
                color="black",
                width=5
            )
        )
    ))

    # Add electrode disks with exaggerated Z (geometry only)
    theta = np.linspace(0, 2*np.pi, 50)
    for i, y0 in enumerate(y_centers):
        for j, x0 in enumerate(x_centers):
            z_val = Z_exagg[i,j]
            circle_x = x0 + radius*np.cos(theta)
            circle_y = y0 + radius*np.sin(theta)
            circle_z = np.ones_like(circle_x) * z_val

            verts_x = np.concatenate(([x0], circle_x))
            verts_y = np.concatenate(([y0], circle_y))
            verts_z = np.concatenate(([z_val], circle_z))

            I = np.zeros(len(theta), dtype=int)
            J = np.arange(1, len(theta)+1)
            K = np.roll(J, -1)

            fig_3d.add_trace(go.Mesh3d(
                x=verts_x, y=verts_y, z=verts_z,
                i=I, j=J, k=K,
                intensity=np.ones_like(verts_z) * Z[i,j],  # color from original Z
                colorscale="portland",
                cmin=z_min, cmax=z_max,
                showscale=False,
                opacity=1.0
            ))


    base_aspect_z = 0.4  # default for z_exaggeration = 1
    aspect_z = base_aspect_z * z_exaggeration

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Active ML (exaggerated)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=aspect_z),  # Scales with exaggeration
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # --- 2D Contour map ---
    fig_2d = go.Figure(data=go.Contour(
        x=x_centers, y=y_centers, z=Z,
        colorscale="portland",
        contours=dict(
            showlines=True,
            start=z_min, end=z_max, size=(z_max - z_min)/12.0,
            coloring="heatmap"
        ),
        colorbar=dict(
            title="Active ML (mg/cm²)",
            tickvals=levels,
            ticktext=[f"{v:.2f}" for v in levels]
        )
    ))
    fig_2d.update_layout(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    # --- Stats ---
    valid_ml = [v['active_ml'] for v in matrix.values() if not np.isnan(v['active_ml'])]
    if len(valid_ml) > 0 and np.mean(valid_ml) != 0:
        uniformity_score = 100.0 * (1.0 - (np.std(valid_ml) / np.mean(valid_ml)))
        st.metric("Uniformity Score", f"{uniformity_score:.2f}%")
        st.write("**Statistical Summary**")
        st.markdown(
            f"- Mean: {np.mean(valid_ml):.5f} mg/cm²\n"
            f"- Std Dev: {np.std(valid_ml):.5f} mg/cm²\n"
            f"- Range: {np.min(valid_ml):.5f} - {np.max(valid_ml):.5f} mg/cm²\n"
            f"- Max Deviation: {np.max(np.abs(deviations)):.5f}%"
        )

    # --- Export data (auto-refreshed content, manual download) ---
    df = pd.DataFrame([{
        'Row': row + 1,
        'Column': col + 1,
        'Active Material': data['active'],
        'Substrate': data['substrate'],
        'Diameter (mm)': f"{data['diameter']:.5f}",
        'Solid (%)': f"{data['solid']:.5f}",
        'Active Material (%)': f"{data['active_pct']:.5f}",
        'Blade Height (µm)': data['blade'],
        'Mass (g)': f"{data['mass']:.5f}",
        'Total ML (mg/cm²)': f"{data['total_ml']:.5f}",
        'Active ML (mg/cm²)': f"{data['active_ml']:.5f}",
        'Deviation (%)': f"{deviations[row, col]:.2f}",
        'Flag': data['flag']
    } for (row, col), data in matrix.items()])

    # Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        "Download Excel",
        data=excel_buffer,
        file_name="coating_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # CSV
    csv_buffer = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_buffer,
        file_name="coating_report.csv",
        mime="text/csv"
    )


# =========================
# Electrode Database Manager
# =========================

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Active Material",
        "Substrate Name",
        "Substrate Tare (mg/cm²)",
        "Diameter (mm)",
        "Mass (g)",
        "Solid %",
        "Blade Height (µm)",
        "Active Material %",
        "Mass Loading (mg/cm²)",
        "Active ML (mg/cm²)",
        "Date Made",
        "Made By",
        "Notes"
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[required_columns]
    return df

# def electrode_database_manager():
    st.title("🗄️ ElectroDB - Electrode Database Manager")
    st.caption("Streamlined electrode data management for battery researchers")

    # Load database
    db = load_database(db_file)

    # Quick stats at the top
    if not db.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Electrodes", len(db))
        with col2:
            st.metric("🧪 Materials", db["Active Material"].nunique())
        with col3:
            st.metric("🗃️ Substrates", db["Substrate"].nunique())
        with col4:
            recent_count = len(db[db["Date Made"].notna()]) if "Date Made" in db.columns else 0
            st.metric("📅 With Dates", recent_count)

    # Tabbed interface
    tab1, tab2, tab3 = st.tabs(["🔍 Add Electrodes", "📋 View & Edit", "📤 Import/Export"])

    with tab1:
        st.subheader("Add New Electrode")
        
        # Initialize session state for form persistence
        if "form_data" not in st.session_state:
            st.session_state.form_data = {
                "custom_material": False,
                "active_material": "LVP",
                "material_key": "LVP",
                "active_pct": 96.0,
                "electrode_type": "Cathode",
                "substrate": "Aluminum Foil",
                "custom_substrate": False,
                "custom_substrate_name": "",
                "custom_tare": 4.163,
                "diameter": 13.0,
                "mass": 0.02,
                "solid_pct": 30.0,
                "blade_height": 200.0,
                "date_made": datetime.now().date(),
                "made_by": "",
                "notes": ""
            }

        # Essential inputs (always visible)
        col1, col2, col3 = st.columns(3)
        with col1:
            electrode_type = st.selectbox(
                "Electrode Type", 
                ["Cathode", "Anode"],
                index=0 if st.session_state.form_data["electrode_type"] == "Cathode" else 1
            )
            st.session_state.form_data["electrode_type"] = electrode_type
        
        with col2:
            diameter = st.number_input(
                "Diameter (mm)", 
                1.0, 50.0, 
                st.session_state.form_data["diameter"], 
                0.1
            )
            st.session_state.form_data["diameter"] = diameter
        
        with col3:
            mass = st.number_input(
                "Total Mass (g)", 
                0.0, 1.0, 
                st.session_state.form_data["mass"], 
                0.0001, 
                format="%.5f"
            )
            st.session_state.form_data["mass"] = mass

        # Material Properties (expandable)
        with st.expander("🧪 Material Properties", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                material_source = st.radio(
                    "Material Source",
                    ["📚 Library", "✏️ Custom"],
                    index=0 if not st.session_state.form_data["custom_material"] else 1,
                    horizontal=True
                )
                st.session_state.form_data["custom_material"] = (material_source == "✏️ Custom")
                
                if material_source == "✏️ Custom":
                    active_material = st.text_input(
                        "Active Material", 
                        value=st.session_state.form_data["active_material"] if st.session_state.form_data["custom_material"] else "",
                        placeholder="e.g., Custom LFP"
                    )
                    st.session_state.form_data["active_material"] = active_material
                else:
                    available_materials = list(MATERIAL_LIBRARY.keys())
                    try:
                        current_idx = available_materials.index(st.session_state.form_data["material_key"])
                    except (ValueError, KeyError):
                        current_idx = 0
                    
                    material_key = st.selectbox(
                        "Active Material", 
                        available_materials,
                        index=current_idx
                    )
                    st.session_state.form_data["material_key"] = material_key
                    active_material = material_key
                    st.session_state.form_data["active_material"] = active_material
                    
                    if material_key in MATERIAL_LIBRARY:
                        material_info = MATERIAL_LIBRARY[material_key]
                        st.caption(f"📖 {material_info['type']} | {material_info['capacity']} mAh/g | Typical: {material_info['active_pct']}%")
            
            with col2:
                active_pct = st.number_input(
                    "Active Material %", 
                    0.0, 100.0, 
                    st.session_state.form_data["active_pct"], 
                    0.1
                )
                st.session_state.form_data["active_pct"] = active_pct

        # Substrate Properties (expandable)
        with st.expander("🗃️ Substrate Properties", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                substrate_source = st.radio(
                    "Substrate Source",
                    ["📚 Standard", "✏️ Custom"],
                    index=0 if not st.session_state.form_data["custom_substrate"] else 1,
                    horizontal=True
                )
                st.session_state.form_data["custom_substrate"] = (substrate_source == "✏️ Custom")
                
                if substrate_source == "✏️ Custom":
                    custom_substrate_name = st.text_input(
                        "Custom Substrate",
                        value=st.session_state.form_data["custom_substrate_name"],
                        placeholder="e.g., Special Copper"
                    )
                    st.session_state.form_data["custom_substrate_name"] = custom_substrate_name
                    substrate_name = custom_substrate_name
                else:
                    substrate_options = list(SUBSTRATE_LIBRARY.keys())
                    try:
                        current_idx = substrate_options.index(st.session_state.form_data["substrate"])
                    except (ValueError, KeyError):
                        current_idx = 0
                    
                    substrate = st.selectbox(
                        "Standard Substrate",
                        substrate_options,
                        index=current_idx
                    )
                    st.session_state.form_data["substrate"] = substrate
                    substrate_name = substrate
                    
                    if substrate in SUBSTRATE_LIBRARY:
                        substrate_info = SUBSTRATE_LIBRARY[substrate]
                        st.caption(f"📋 {substrate_info['tare_mg_cm2']:.3f} mg/cm² | {substrate_info['type']}")
            
            with col2:
                if substrate_source == "✏️ Custom":
                    custom_tare = st.number_input(
                        "Substrate Tare (mg/cm²)",
                        0.0, 50.0,
                        st.session_state.form_data["custom_tare"],
                        0.001,
                        format="%.5f"
                    )
                    st.session_state.form_data["custom_tare"] = custom_tare
                else:
                    if st.session_state.form_data["substrate"] in SUBSTRATE_LIBRARY:
                        tare_value = SUBSTRATE_LIBRARY[st.session_state.form_data["substrate"]]["tare_mg_cm2"]
                        st.metric("Substrate Tare", f"{tare_value:.3f} mg/cm²")

        # Processing Parameters (expandable)
        with st.expander("⚙️ Processing Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                solid_pct = st.number_input(
                    "Slurry Solid Content (%)", 
                    0.0, 100.0, 
                    st.session_state.form_data["solid_pct"], 
                    0.1
                )
                st.session_state.form_data["solid_pct"] = solid_pct
            
            with col2:
                blade_height = st.number_input(
                    "Blade Height (µm)", 
                    1.0, 1000.0, 
                    st.session_state.form_data["blade_height"], 
                    1.0
                )
                st.session_state.form_data["blade_height"] = blade_height

        # Additional Info (expandable)
        with st.expander("📋 Additional Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                date_made = st.date_input(
                    "Date Made",
                    value=st.session_state.form_data["date_made"]
                )
                st.session_state.form_data["date_made"] = date_made
                
                made_by = st.text_input(
                    "Made By",
                    value=st.session_state.form_data["made_by"],
                    placeholder="Initials"
                )
                st.session_state.form_data["made_by"] = made_by
            
            with col2:
                notes = st.text_area(
                    "Notes", 
                    value=st.session_state.form_data["notes"],
                    placeholder="Optional notes...",
                    height=80
                )
                st.session_state.form_data["notes"] = notes

        # Mass loading preview (always visible but compact)
        try:
            if substrate_source == "✏️ Custom":
                area_cm2 = mm_diameter_to_area_cm2(diameter)
                substrate_mass_g = (custom_tare / 1000.0) * area_cm2
                substrate_display = f"{custom_substrate_name} ({custom_tare:.3f} mg/cm²)"
            else:
                substrate_mass_g = get_substrate_mass_g(substrate_name, diameter)
                if substrate_name in SUBSTRATE_LIBRARY:
                    tare_val = SUBSTRATE_LIBRARY[substrate_name]["tare_mg_cm2"]
                    substrate_display = f"{substrate_name} ({tare_val:.3f} mg/cm²)"
                else:
                    substrate_display = substrate_name
            
            total_ml, active_ml = calc_mass_loading_total_and_active(
                mass, diameter, substrate_mass_g, active_pct
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total ML", f"{total_ml:.2f} mg/cm²")
            with col2:
                st.metric("Active ML", f"{active_ml:.2f} mg/cm²")
            with col3:
                area_cm2 = mm_diameter_to_area_cm2(diameter)
                st.metric("Area", f"{area_cm2:.3f} cm²")
                
            # Expandable calculation details
            with st.expander("📊 Calculation Details", expanded=False):
                st.info(f"""
                **Calculation Summary:**
                - Substrate: {substrate_display}
                - Electrode: Ø{diameter:.1f} mm ({area_cm2:.3f} cm²)
                - Total mass: {mass:.5f} g
                - Substrate mass: {substrate_mass_g*1000:.2f} mg
                - Coating mass: {(mass-substrate_mass_g)*1000:.2f} mg
                - Active material: {active_pct:.1f}% of coating
                """)
                
        except Exception as e:
            st.error(f"⚠️ Calculation error: {e}")
            total_ml, active_ml = 0.0, 0.0

        # Action buttons
        col1, col2 = st.columns([3, 1])
        
        with col1:
            submit_button = st.button(
                "💾 Add to Database", 
                use_container_width=True, 
                type="primary"
            )
        
        with col2:
            reset_button = st.button(
                "🔄 Reset", 
                use_container_width=True
            )

        # Handle buttons
        if reset_button:
            st.session_state.form_data = {
                "custom_material": False,
                "active_material": "LVP",
                "material_key": "LVP", 
                "active_pct": 96.0,
                "electrode_type": "Cathode",
                "substrate": "Aluminum Foil",
                "custom_substrate": False,
                "custom_substrate_name": "",
                "custom_tare": 4.163,
                "diameter": 13.0,
                "mass": 0.02,
                "solid_pct": 30.0,
                "blade_height": 200.0,
                "date_made": datetime.now().date(),
                "made_by": "",
                "notes": ""
            }
            st.rerun()

        if submit_button:
            if not active_material.strip():
                st.error("❌ Please enter an active material name.")
                return
                
            if substrate_source == "✏️ Custom" and not custom_substrate_name.strip():
                st.error("❌ Please enter a custom substrate name.")
                return
                
            try:
                # Calculate mass loadings
                if substrate_source == "✏️ Custom":
                    area_cm2 = mm_diameter_to_area_cm2(diameter)
                    substrate_mass_g = (custom_tare / 1000.0) * area_cm2
                    final_substrate_name = custom_substrate_name
                else:
                    substrate_mass_g = get_substrate_mass_g(substrate_name, diameter)
                    final_substrate_name = substrate_name
                
                total_ml, active_ml = calc_mass_loading_total_and_active(
                    mass, diameter, substrate_mass_g, active_pct
                )
                
                # Create new row
                new_row = {
                    "Active Material": active_material,
                    "Substrate": final_substrate_name,
                    "Diameter (mm)": diameter,
                    "Mass (g)": mass,
                    "Solid %": solid_pct,
                    "Blade Height (µm)": blade_height,
                    "Active Material %": active_pct,
                    "Mass Loading (mg/cm²)": total_ml,
                    "Active ML (mg/cm²)": active_ml,
                    "Date Made": str(date_made),
                    "Made By": made_by,
                    "Notes": notes
                }
                
                # Add to database
                db_new = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
                db_new = ensure_db_columns(db_new)
                save_database(db_new, db_file)
                
                st.success(f"✅ Added {active_material} electrode!")
                
                # Show what was added in expander
                with st.expander("📋 Added Details", expanded=False):
                    st.json(new_row)
                    
            except Exception as e:
                st.error(f"❌ Failed to add electrode: {str(e)}")

    with tab2:
        st.subheader("📋 Database Contents")
        
        if db.empty:
            st.info("🔭 No electrodes yet. Add some in the 'Add Electrodes' tab!")
        else:
            # Compact filters in expander
            with st.expander("🔍 Filters", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    search_material = st.selectbox(
                        "Material", 
                        ["All"] + sorted(db["Active Material"].dropna().unique().tolist())
                    )
                
                with col2:
                    search_substrate = st.selectbox(
                        "Substrate",
                        ["All"] + sorted(db["Substrate"].dropna().unique().tolist())
                    )
                    
                with col3:
                    search_person = st.selectbox(
                        "Made By",
                        ["All"] + sorted([p for p in db["Made By"].dropna().unique() if str(p).strip()])
                    )

            # Apply filters
            filtered_db = db.copy()
            if search_material != "All":
                filtered_db = filtered_db[filtered_db["Active Material"] == search_material]
            if search_substrate != "All":
                filtered_db = filtered_db[filtered_db["Substrate"] == search_substrate]
            if search_person != "All":
                filtered_db = filtered_db[filtered_db["Made By"] == search_person]

            st.caption(f"Showing {len(filtered_db)} of {len(db)} electrodes")

            # Display table
            if not filtered_db.empty:
                display_df = filtered_db.copy()
                
                # Format numeric columns
                numeric_cols = ["Mass (g)", "Mass Loading (mg/cm²)", "Active ML (mg/cm²)"]
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").map(
                            lambda x: f"{x:.3f}" if pd.notnull(x) else ""
                        )
                
                st.dataframe(display_df, use_container_width=True, height=300)

                # Compact delete in expander
                with st.expander("🗑️ Delete Electrode", expanded=False):
                    if len(filtered_db) > 0:
                        delete_options = []
                        for idx, row in filtered_db.iterrows():
                            material = row.get("Active Material", "Unknown")
                            substrate = row.get("Substrate", "Unknown")  
                            date = row.get("Date Made", "No date")
                            delete_options.append(f"Row {idx}: {material} on {substrate} ({date})")
                        
                        to_delete = st.selectbox("Select:", ["Select..."] + delete_options)
                        
                        if to_delete != "Select..." and st.button("🗑️ Delete", type="secondary"):
                            try:
                                row_idx = int(to_delete.split(":")[0].replace("Row ", ""))
                                db_updated = db.drop(row_idx).reset_index(drop=True)
                                save_database(db_updated, db_file)
                                st.success("✅ Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Delete failed: {e}")

    with tab3:
        st.subheader("📤 Import & Export")
        
        # Import section in expander
        with st.expander("📥 Import Database", expanded=False):
            uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
            
            if uploaded_file is not None:
                try:
                    new_db = pd.read_excel(uploaded_file)
                    new_db = ensure_db_columns(new_db)
                    
                    st.write("Preview:")
                    st.dataframe(new_db.head(3))
                    
                    import_mode = st.radio("Mode:", ["Replace database", "Add to existing"])
                    
                    if st.button("📥 Import"):
                        if import_mode == "Replace database":
                            save_database(new_db, db_file)
                            st.success(f"✅ Replaced! Imported {len(new_db)} electrodes.")
                        else:
                            combined_db = pd.concat([db, new_db], ignore_index=True)
                            combined_db = ensure_db_columns(combined_db)
                            save_database(combined_db, db_file)
                            st.success(f"✅ Added {len(new_db)} electrodes!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Import failed: {e}")
        
        # Export section
        with st.expander("📤 Export Database", expanded=True if not db.empty else False):
            if not db.empty:
                export_filtered = st.checkbox("Export filtered data only", value=False)
                
                if export_filtered and 'filtered_db' in locals():
                    export_db = filtered_db
                    st.info(f"Will export {len(export_db)} filtered electrodes")
                else:
                    export_db = db
                    st.info(f"Will export all {len(export_db)} electrodes")

                try:
                    excel_bytes, filename = bytes_excel(export_db, "electrode_database.xlsx")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Excel",
                            data=excel_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col2:
                        csv_data = export_db.to_csv(index=False)
                        st.download_button(
                            label="📄 CSV",
                            data=csv_data,
                            file_name="electrode_database.csv", 
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
            else:
                st.info("🔭 No data to export")

        # Database statistics in expander
        if not db.empty:
            with st.expander("📊 Database Statistics", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Materials:**")
                    material_counts = db["Active Material"].value_counts()
                    for material, count in material_counts.head(5).items():
                        st.write(f"• {material}: {count}")
                
                with col2:
                    st.markdown("**Substrates:**")
                    substrate_counts = db["Substrate"].value_counts()
                    for substrate, count in substrate_counts.head(5).items():
                        st.write(f"• {substrate}: {count}")


def electrode_database_manager():
    st.title("🗄️ ElectroDB - Electrode Database Manager")
    st.caption("Streamlined electrode data management for battery researchers")

    # Load database
    db = load_database(db_file)

    # Quick stats at the top
    if not db.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Electrodes", len(db))
        with col2:
            st.metric("🧪 Materials", db["Active Material"].nunique())
        with col3:
            st.metric("🗃️ Substrates", db["Substrate"].nunique())
        with col4:
            recent_count = len(db[db["Date Made"].notna()]) if "Date Made" in db.columns else 0
            st.metric("📅 With Dates", recent_count)

    # Tabbed interface
    tab1, tab2, tab3 = st.tabs(["🔍 Add Electrodes", "📋 View & Edit", "📤 Import/Export"])

    with tab1:
        st.subheader("Add New Electrode")
        
        # Initialize session state for form persistence
        if "form_data" not in st.session_state:
            st.session_state.form_data = {
                "custom_material": False,
                "active_material": "LVP",
                "material_key": "LVP",
                "active_pct": 96.0,
                "electrode_type": "Cathode",
                "substrate": "Aluminum Foil",
                "custom_substrate": False,
                "custom_substrate_name": "",
                "custom_tare": 4.163,
                "diameter": 13.0,
                "mass": 0.02,
                "solid_pct": 30.0,
                "blade_height": 200.0,
                "date_made": datetime.now().date(),
                "made_by": "",
                "notes": ""
            }

        # --- Essential Inputs ---
        col1, col2, col3 = st.columns(3)
        with col1:
            electrode_type = st.selectbox(
                "Electrode Type", ["Cathode", "Anode"],
                index=0 if st.session_state.form_data["electrode_type"]=="Cathode" else 1
            )
            st.session_state.form_data["electrode_type"] = electrode_type
        with col2:
            diameter = st.number_input(
                "Diameter (mm)", 1.0, 50.0, st.session_state.form_data["diameter"], 0.1
            )
            st.session_state.form_data["diameter"] = diameter
        with col3:
            mass = st.number_input(
                "Total Mass (g)", 0.0, 1.0, st.session_state.form_data["mass"], 0.0001, format="%.5f"
            )
            st.session_state.form_data["mass"] = mass

        # --- Material Properties ---
        with st.expander("🧪 Material Properties", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                material_source = st.radio(
                    "Material Source", ["📚 Library", "✏️ Custom"],
                    index=0 if not st.session_state.form_data["custom_material"] else 1, horizontal=True
                )
                st.session_state.form_data["custom_material"] = (material_source=="✏️ Custom")
                if material_source=="✏️ Custom":
                    active_material = st.text_input(
                        "Active Material", value=st.session_state.form_data["active_material"], placeholder="e.g., Custom LFP"
                    )
                    st.session_state.form_data["active_material"] = active_material
                else:
                    available_materials = list(MATERIAL_LIBRARY.keys())
                    try:
                        current_idx = available_materials.index(st.session_state.form_data["material_key"])
                    except (ValueError, KeyError):
                        current_idx = 0
                    material_key = st.selectbox(
                        "Active Material", available_materials, index=current_idx
                    )
                    st.session_state.form_data["material_key"] = material_key
                    active_material = material_key
                    st.session_state.form_data["active_material"] = active_material
                    if material_key in MATERIAL_LIBRARY:
                        material_info = MATERIAL_LIBRARY[material_key]
                        st.caption(f"📖 {material_info['type']} | {material_info['capacity']} mAh/g | Typical: {material_info['active_pct']}%")
            with col2:
                active_pct = st.number_input(
                    "Active Material %", 0.0, 100.0, st.session_state.form_data["active_pct"], 0.1
                )
                st.session_state.form_data["active_pct"] = active_pct

        # --- Substrate Properties ---
        with st.expander("🗃️ Substrate Properties", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                substrate_source = st.radio(
                    "Substrate Source", ["📚 Standard","✏️ Custom"],
                    index=0 if not st.session_state.form_data["custom_substrate"] else 1, horizontal=True
                )
                st.session_state.form_data["custom_substrate"] = (substrate_source=="✏️ Custom")
                if substrate_source=="✏️ Custom":
                    custom_substrate_name = st.text_input(
                        "Custom Substrate", value=st.session_state.form_data["custom_substrate_name"], placeholder="e.g., Special Copper"
                    )
                    st.session_state.form_data["custom_substrate_name"] = custom_substrate_name
                    substrate_name = custom_substrate_name
                else:
                    substrate_options = list(SUBSTRATE_LIBRARY.keys())
                    try:
                        current_idx = substrate_options.index(st.session_state.form_data["substrate"])
                    except (ValueError, KeyError):
                        current_idx = 0
                    substrate = st.selectbox(
                        "Standard Substrate", substrate_options, index=current_idx
                    )
                    st.session_state.form_data["substrate"] = substrate
                    substrate_name = substrate
                    if substrate in SUBSTRATE_LIBRARY:
                        substrate_info = SUBSTRATE_LIBRARY[substrate]
                        st.caption(f"📋 {substrate_info['tare_mg_cm2']:.3f} mg/cm² | {substrate_info['type']}")
            with col2:
                if substrate_source=="✏️ Custom":
                    custom_tare = st.number_input(
                        "Substrate Tare (mg/cm²)", 0.0, 50.0, st.session_state.form_data["custom_tare"], 0.001, format="%.5f"
                    )
                    st.session_state.form_data["custom_tare"] = custom_tare
                else:
                    if substrate_name in SUBSTRATE_LIBRARY:
                        st.metric("Substrate Tare", f"{SUBSTRATE_LIBRARY[substrate_name]['tare_mg_cm2']:.3f} mg/cm²")

        # --- Processing Parameters ---
        with st.expander("⚙️ Processing Parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                solid_pct = st.number_input("Slurry Solid Content (%)",0.0,100.0,st.session_state.form_data["solid_pct"],0.1)
                st.session_state.form_data["solid_pct"] = solid_pct
            with col2:
                blade_height = st.number_input("Blade Height (µm)",1.0,1000.0,st.session_state.form_data["blade_height"],1.0)
                st.session_state.form_data["blade_height"] = blade_height

        # --- Additional Info ---
        with st.expander("📋 Additional Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                date_made = st.date_input("Date Made", value=st.session_state.form_data["date_made"])
                st.session_state.form_data["date_made"] = date_made
                made_by = st.text_input("Made By", value=st.session_state.form_data["made_by"], placeholder="Initials")
                st.session_state.form_data["made_by"] = made_by
            with col2:
                notes = st.text_area("Notes", value=st.session_state.form_data["notes"], placeholder="Optional notes...", height=80)
                st.session_state.form_data["notes"] = notes

        # --- Mass loading preview ---

        # Initialize defaults
        area_cm2 = 0.0
        tare_mg_cm2 = 0.0
        total_ml = 0.0
        active_ml = 0.0
        try:
            if substrate_source=="✏️ Custom":
                area_cm2 = mm_diameter_to_area_cm2(diameter)
                substrate_mass_g = (custom_tare/1000.0)*area_cm2
                substrate_display = f"{custom_substrate_name} ({custom_tare:.3f} mg/cm²)"
                tare_mg_cm2 = custom_tare
            else:
                substrate_mass_g = get_substrate_mass_g(substrate_name, diameter)
                if substrate_name in SUBSTRATE_LIBRARY:
                    area_cm2 = mm_diameter_to_area_cm2(diameter)
                    tare_val = SUBSTRATE_LIBRARY[substrate_name]["tare_mg_cm2"]
                    substrate_display = f"{substrate_name} ({tare_val:.3f} mg/cm²)"
                    tare_mg_cm2 = tare_val
                else:
                    substrate_display = substrate_name
                    tare_mg_cm2 = 0.0
            total_ml, active_ml = calc_mass_loading_total_and_active(mass, diameter, substrate_mass_g, active_pct)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total ML", f"{total_ml:.2f} mg/cm²")
            with col2: st.metric("Active ML", f"{active_ml:.2f} mg/cm²")
            with col3: st.metric("Area", f"{area_cm2:.3f} cm²")
            with st.expander("📊 Calculation Details", expanded=False):
                st.info(f"""
                **Calculation Summary:**
                - Substrate: {substrate_display}
                - Tare Weight Used: {tare_mg_cm2:.3f} mg/cm²
                - Electrode: Ø{diameter:.1f} mm ({area_cm2:.3f} cm²)
                - Total mass: {mass:.5f} g
                - Substrate mass: {substrate_mass_g*1000:.2f} mg
                - Coating mass: {(mass-substrate_mass_g)*1000:.2f} mg
                - Active material: {active_pct:.1f}% of coating
                """)
        except Exception as e:
            st.error(f"⚠️ Calculation error: {e}")
            total_ml, active_ml, tare_mg_cm2 = 0.0,0.0,0.0

        # --- Buttons ---
        col1, col2 = st.columns([3,1])
        with col1:
            submit_button = st.button("💾 Add to Database", use_container_width=True, type="primary")
        with col2:
            reset_button = st.button("🔄 Reset", use_container_width=True)

        if reset_button:
            st.session_state.form_data = {
                "custom_material": False, "active_material": "LVP", "material_key": "LVP",
                "active_pct": 96.0, "electrode_type": "Cathode", "substrate": "Aluminum Foil",
                "custom_substrate": False, "custom_substrate_name": "", "custom_tare": 4.163,
                "diameter": 13.0, "mass": 0.02, "solid_pct": 30.0, "blade_height": 200.0,
                "date_made": datetime.now().date(), "made_by": "", "notes": ""
            }
            st.rerun()

        if submit_button:
            if not active_material.strip(): st.error("❌ Please enter an active material name."); return
            if substrate_source=="✏️ Custom" and not custom_substrate_name.strip(): st.error("❌ Please enter a custom substrate name."); return
            try:
                if substrate_source=="✏️ Custom":
                    area_cm2 = mm_diameter_to_area_cm2(diameter)
                    substrate_mass_g = (custom_tare/1000.0)*area_cm2
                    final_substrate_name = custom_substrate_name
                    tare_mg_cm2 = custom_tare
                else:
                    substrate_mass_g = get_substrate_mass_g(substrate_name, diameter)
                    final_substrate_name = substrate_name
                    tare_mg_cm2 = SUBSTRATE_LIBRARY[substrate_name]["tare_mg_cm2"] if substrate_name in SUBSTRATE_LIBRARY else 0.0
                total_ml, active_ml = calc_mass_loading_total_and_active(mass, diameter, substrate_mass_g, active_pct)
                new_row = {
                    "Electrode Type": electrode_type,
                    "Active Material": active_material,
                    "Substrate": final_substrate_name,
                    "Tare Weight (mg/cm²)": tare_mg_cm2,
                    "Mass (g)": mass,
                    "Diameter (mm)": diameter,
                    "Solid %": solid_pct,
                    "Blade Height (µm)": blade_height,
                    "Active Material %": active_pct,
                    "Mass Loading (mg/cm²)": total_ml,
                    "Active ML (mg/cm²)": active_ml,
                    "Date Made": str(date_made),
                    "Made By": made_by,
                    "Notes": notes
                }
                db_new = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
                db_new = ensure_db_columns(db_new)
                save_database(db_new, db_file)
                st.success(f"✅ Added {active_material} electrode!")
                with st.expander("📋 Added Details", expanded=False):
                    st.json(new_row)
            except Exception as e:
                st.error(f"❌ Failed to add electrode: {str(e)}")

    # --- Tab 2: View/Edit Display ---
    with tab2:
        st.subheader("📋 Database Contents")
        if db.empty:
            st.info("🔭 No electrodes yet. Add some in the 'Add Electrodes' tab!")
        else:
            # Filters
            with st.expander("🔍 Filters", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    search_material = st.selectbox("Material", ["All"] + sorted(db["Active Material"].dropna().unique().tolist()))
                with col2:
                    search_substrate = st.selectbox("Substrate", ["All"] + sorted(db["Substrate"].dropna().unique().tolist()))
                with col3:
                    search_person = st.selectbox("Made By", ["All"] + sorted([p for p in db["Made By"].dropna().unique() if str(p).strip()]))
            filtered_db = db.copy()
            if search_material != "All": filtered_db = filtered_db[filtered_db["Active Material"]==search_material]
            if search_substrate != "All": filtered_db = filtered_db[filtered_db["Substrate"]==search_substrate]
            if search_person != "All": filtered_db = filtered_db[filtered_db["Made By"]==search_person]
            st.caption(f"Showing {len(filtered_db)} of {len(db)} electrodes")
            if not filtered_db.empty:
                display_df = filtered_db.copy()
                numeric_cols = ["Mass (g)", "Mass Loading (mg/cm²)", "Active ML (mg/cm²)", "Tare Weight (mg/cm²)"]
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").map(lambda x:f"{x:.3f}" if pd.notnull(x) else "")
                st.dataframe(display_df, use_container_width=True, height=300)

                # Delete section
                with st.expander("🗑️ Delete Electrode", expanded=False):
                    if len(filtered_db)>0:
                        delete_options=[]
                        for idx,row in filtered_db.iterrows():
                            material=row.get("Active Material","Unknown")
                            substrate=row.get("Substrate","Unknown")
                            date=row.get("Date Made","No date")
                            delete_options.append(f"Row {idx}: {material} on {substrate} ({date})")
                        to_delete=st.selectbox("Select:", ["Select..."]+delete_options)
                        if to_delete!="Select..." and st.button("🗑️ Delete", type="secondary"):
                            try:
                                row_idx=int(to_delete.split(":")[0].replace("Row ",""))
                                db_updated=db.drop(row_idx).reset_index(drop=True)
                                save_database(db_updated, db_file)
                                st.success("✅ Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Delete failed: {e}")

    # --- Tab 3: Import/Export ---
    with tab3:
        st.subheader("📤 Import & Export")
        # Import
        with st.expander("📥 Import Database", expanded=False):
            uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
            if uploaded_file is not None:
                try:
                    new_db = pd.read_excel(uploaded_file)
                    new_db = ensure_db_columns(new_db)
                    st.dataframe(new_db.head(3))
                    import_mode = st.radio("Mode:", ["Replace database", "Add to existing"])
                    if st.button("📥 Import"):
                        if import_mode=="Replace database": save_database(new_db, db_file)
                        else:
                            combined_db=pd.concat([db,new_db],ignore_index=True)
                            combined_db=ensure_db_columns(combined_db)
                            save_database(combined_db, db_file)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Import failed: {e}")

        # Export
        with st.expander("📤 Export Database", expanded=True if not db.empty else False):
            if not db.empty:
                export_filtered=st.checkbox("Export filtered data only", value=False)
                export_db = filtered_db if export_filtered and 'filtered_db' in locals() else db
                export_columns=[
                    "Electrode Type","Active Material","Substrate","Tare Weight (mg/cm²)",
                    "Mass (g)","Diameter (mm)","Solid %","Blade Height (µm)","Active Material %",
                    "Mass Loading (mg/cm²)","Active ML (mg/cm²)","Date Made","Made By","Notes"
                ]
                export_db = export_db.reindex(columns=export_columns)
                try:
                    excel_bytes, filename = bytes_excel(export_db,"electrode_database.xlsx")
                    col1,col2=st.columns(2)
                    with col1:
                        st.download_button("📥 Excel", excel_bytes, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                    with col2:
                        csv_data=export_db.to_csv(index=False)
                        st.download_button("📄 CSV", csv_data,file_name="electrode_database.csv",mime="text/csv", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
            else:
                st.info("🔭 No data to export")

        # Statistics
        if not db.empty:
            with st.expander("📊 Database Statistics", expanded=False):
                col1,col2=st.columns(2)
                with col1:
                    st.markdown("**Materials:**")
                    material_counts=db["Active Material"].value_counts()
                    for material,count in material_counts.head(5).items(): st.write(f"• {material}: {count}")
                with col2:
                    st.markdown("**Substrates:**")
                    substrate_counts=db["Substrate"].value_counts()
                    for substrate,count in substrate_counts.head(5).items(): st.write(f"• {substrate}: {count}")


# =========================
# Router
# =========================
if tool == "Slurry Calculator":
    slurry_calculator()
elif tool == "Blade Height Recommender":
    blade_height_recommender()
elif tool == "Capacity Match Tool":
    capacity_match_tool()
elif tool == "Coating Calibration Tool":
    coating_calibration_tool()
elif tool == "Electrode Database Manager":
    electrode_database_manager()
