# Battery Lab Tools — Slurry + Blade Height + Capacity Match + Electrode Database

# =========================
# Requirements Config
# =========================

import io
import os
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy
from scipy.stats import linregress
import xlsxwriter
from fpdf import FPDF
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

# =========================
# Global Config
# =========================
st.set_page_config(page_title="Battery Lab Tools", layout="wide")
plt.rcParams.update({"figure.max_open_warning": 0})

# =========================
# Constants
# =========================
ELECTRODE_AREA_13MM = 1.3273  # cm², 13 mm punch (π * (6.5 mm)^2 / 100)
DEFAULT_DB_FILE = "electrode_database.xlsx"

# Foil masses for 13 mm punches measured in the user's earlier data
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
    }
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

# Default active material fractions for capacity match
MATERIAL_ACTIVE_RATIOS = {
    'NaVP': 0.96,
    'LVP': 0.96,
    'Li3V2(PO4)3': 0.96,
    'Na3V2(PO4)3': 0.96,
    'Graphite': 0.80,
    'Hard Carbon': 0.90,
}

# =========================
# Utilities
# =========================
def mm_diameter_to_area_cm2(d_mm: float) -> float:
    r_cm = (d_mm / 2.0) / 10.0
    return math.pi * r_cm * r_cm

def pick_foil_mass(sub: str) -> float | None:
    if not sub:
        return None
    key = sub.strip().lower()
    return FOIL_MASS_MAP.get(key, None)

def calc_mass_loading_total_and_active(
    disk_mass_g: float,
    diameter_mm: float,
    substrate: str,
    active_pct: float | None,
) -> tuple[float, float]:
    """
    Returns:
      total_ml_mg_cm2: mg/cm² of total coating (foil-subtracted)
      active_ml_mg_cm2: mg/cm² of active-only (if active_pct provided)
    """
    area = mm_diameter_to_area_cm2(diameter_mm)
    foil = pick_foil_mass(substrate)
    if area <= 0 or disk_mass_g is None or np.isnan(disk_mass_g):
        return (np.nan, np.nan)
    if foil is None:
        # Cannot subtract foil. Treat disk mass as coating mass. Warn upstream in UI.
        coat_mass_g = disk_mass_g
    else:
        coat_mass_g = max(disk_mass_g - foil, 0.0)
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
    cols = [
        "Active Material", "Substrate", "Diameter (mm)", "Mass (g)",
        "Solid %", "Blade Height (µm)", "Active Material %",
        "Mass Loading (mg/cm²)", "Active ML (mg/cm²)"
    ]
    for c in cols:
        if c not in df.columns:
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

def plot_regression(
    x_vals, y_vals, xlabel, ylabel, material_label=None, color="blue",
    target_y=None, recommended_x=None, x_is_ticks=True
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x_vals, y_vals, color=color, label="Measured")
    slope, intercept, r, p, se = linregress(x_vals, y_vals)
    fit_x = np.linspace(min(x_vals)*0.8, max(x_vals)*1.2, 200)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, linestyle='--', color=color, label=f"Fit: y={slope:.3f}x+{intercept:.3f} (R²={r**2:.3f})")
    if target_y is not None:
        ax.axhline(target_y, linestyle=':', color='black', label=f"Target {ylabel}")
    if recommended_x is not None:
        ax.axvline(recommended_x, linestyle=':', color='gray',
                   label=f"Recommended Height: {recommended_x:.1f} {'(10 μm)' if x_is_ticks else 'µm'}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = f"{ylabel} vs {xlabel}"
    if material_label:
        title += f" — {material_label}"
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    return slope, intercept, r**2

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
    options=["Slurry Calculator", "Blade Height Recommender", "Capacity Match Tool", "Coating Calibration Tool", "Electrode Database Manager"],
    index=0
)

# =========================
# Slurry Calculator
# =========================

def slurry_calculator():
    st.title("Slurry Calculator")
    st.caption("Cathode/Anode slurry planning with true recipe tracking, multi-component breakdown, and dilution history.")

    # ---- Column 1: Formula & Mass Targets ----
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Formula (must sum to 100%)")
        active_ratio = st.number_input("Active Material % in Formula", 0.0, 100.0, 96.0, 0.1)
        carbon_ratio = st.number_input("Conductive Carbon % in Formula", 0.0, 100.0, 2.0, 0.1)
        binder_ratio = st.number_input("Binder % in Formula", 0.0, 100.0, 2.0, 0.1)
        total_ratio = active_ratio + carbon_ratio + binder_ratio
        if abs(total_ratio - 100.0) > 0.01:
            st.error(f"Mass ratio must add to 100%. Current: {total_ratio:.2f}%")
            st.stop()

        target_mode = st.radio("Target Mode", ["Active Mass (g)", "Total Slurry Mass (g)"])
        if target_mode == "Active Mass (g)":
            active_mass = st.number_input("Active Material Mass (g)", 0.0, step=0.01, value=1.0)
        else:
            total_slurry_mass = st.number_input("Total Slurry Mass (g)", 0.0, step=0.01, value=10.0)

        use_solution = st.checkbox("Using Binder Solution?", True)
        binder_solution_pct = None
        if use_solution:
            binder_solution_pct = st.number_input("Binder % in Solution", 0.1, 100.0, 5.0)

        solid_pct = st.number_input("Target Solid Content (%)", 0.1, 100.0, 30.0)

    # ---- Component breakdown function ----
    def get_components(name, default_names, default_weights):
        multi = st.checkbox(f"{name} has multiple components?", value=False, key=f"{name}_multi")
        n = st.number_input(f"Number of {name} components", 1, 10, value=len(default_names), key=f"{name}_count") if multi else 1
        names, weights = [], []
        for i in range(n):
            nm = st.text_input(f"{name} #{i+1} name", value=default_names[i] if i < len(default_names) else "", key=f"{name}_{i}_nm")
            wt = st.number_input(f"{name} #{i+1} %", 0.0, 100.0, value=default_weights[i] if i < len(default_weights) else 100.0/n, step=0.1, key=f"{name}_{i}_wt")
            names.append(nm)
            weights.append(wt)
        if abs(sum(weights) - 100.0) > 0.01:
            st.error(f"{name} splits must total 100%. Current: {sum(weights):.2f}%")
            return None, None
        return names, weights

    with col2:
        st.subheader("Component Details")
        active_names, active_weights = get_components("Active", ["NVP"], [100.0])
        carbon_names, carbon_weights = get_components("Carbon", ["Super P"], [100.0])
        binder_names, binder_weights = get_components("Binder", ["CMC"], [100.0])
        solvent_names, solvent_weights = get_components("Solvent", ["NMP"], [100.0])
        if None in (active_names, carbon_names, binder_names, solvent_names):
            st.stop()

    # ---- Compute masses ----
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
    solvent_pure_mass = total_slurry_mass - total_solids - (binder_solution_mass - binder_mass if use_solution else 0)
    solvent_in_binder = binder_solution_mass - binder_mass if use_solution else 0
    solvent_total_combined = solvent_pure_mass + solvent_in_binder

    # ---- Display recipe ----
    st.subheader("Initial Slurry Recipe")
    recipe_dict = {
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
    clean_dict = {k:v for k,v in recipe_dict.items() if v is not None}
    df_recipe = pd.DataFrame.from_dict({k: round(v,4) for k,v in clean_dict.items()}, orient="index", columns=["Mass (g)"])
    st.table(df_recipe)

    st.subheader("Component Mass Breakdown")
    def display_breakdown(names, weights, total_mass, label):
        st.markdown(f"**{label} components:**")
        for nm, wt in zip(names, weights):
            st.markdown(f"- {nm}: {total_mass*wt/100:.4f} g")
    display_breakdown(active_names, active_weights, active_mass, "Active")
    display_breakdown(carbon_names, carbon_weights, carbon_mass, "Carbon")
    display_breakdown(binder_names, binder_weights, binder_mass, "Binder")
    display_breakdown(solvent_names, solvent_weights, solvent_total_combined, "Solvent (total)")

    # ---- True Recipe Tracker ----
    track_true = st.checkbox("Track True Recipe?")
    true_recipe = {}
    if track_true:
        st.subheader("Input Actual Masses")
        for k in clean_dict.keys():
            val = st.number_input(f"Actual {k}", value=clean_dict[k], step=0.01)
            true_recipe[k] = val
        st.markdown("**Actual component breakdown**")
        display_breakdown(active_names, active_weights, true_recipe['Active Mass (g)'], "Active")
        display_breakdown(carbon_names, carbon_weights, true_recipe['Carbon Mass (g)'], "Carbon")
        display_breakdown(binder_names, binder_weights, true_recipe['Binder Mass (g)'], "Binder")
        display_breakdown(solvent_names, solvent_weights, true_recipe['Solvent Mass Total (g)'], "Solvent (total)")

    # ---- Dilution history tracker ----
    st.subheader("Dilution Tool")
    if 'dilution_history' not in st.session_state:
        st.session_state.dilution_history = []

    new_solid_pct = st.number_input("New Solid % (must be <= original)", 0.1, solid_pct, value=solid_pct)

    if new_solid_pct < solid_pct:
        # Calculate additional solvent needed
        current_total_slurry = total_slurry_mass
        new_total_slurry = total_solids / (new_solid_pct / 100.0)
        additional_solvent = new_total_slurry - current_total_slurry
        st.session_state.dilution_history.append(additional_solvent)
        st.success(f"Add {additional_solvent:.4f} g of solvent to reach {new_solid_pct}% solids.")
    st.write("Dilution history (g solvent added):", st.session_state.dilution_history)


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

            # Optional: Combined visualization sheet
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
    st.title("Blade Height Recommender")
    st.caption("Fit mass loading vs blade height from built-in data or your database.")
    db = load_database(db_file)

    # Data source selector
    src = st.radio("Data source", ["Use Database", "Use Built-in Dataset"], index=0, horizontal=True)

    # Material selection
    if src == "Use Database":
        if db.empty:
            st.warning("Database empty. Upload or add entries in the Electrode Database Manager.")
            st.stop()

        mat_side = st.selectbox("Electrode side", ["Cathode", "Anode"])
        if mat_side == "Cathode":
            candidates = sorted({m for m in db["Active Material"].dropna().unique() if m in CATHODE_DATA.keys() or m})
        else:
            candidates = sorted({m for m in db["Active Material"].dropna().unique() if m in ANODE_DATA.keys() or m})

        material = st.selectbox("Material", candidates)
        substrate = st.text_input("Filter by substrate (optional, exact match recommended)", value="")
        d_lower, d_upper = st.columns(2)
        with d_lower:
            tol_active = st.number_input("Active Material % filter: min", 0.0, 100.0, value=0.0, step=0.1)
        with d_upper:
            tol_solid = st.number_input("Solid % filter: min", 0.0, 100.0, value=0.0, step=0.1)

        # Compute ML for each DB row if missing
        db = db.copy()
        missing_ml = db["Mass Loading (mg/cm²)"].isna() | db["Active ML (mg/cm²)"].isna()
        if missing_ml.any():
            for idx in db.index[missing_ml]:
                row = db.loc[idx]
                total_ml, active_ml = calc_mass_loading_total_and_active(
                    disk_mass_g=row.get("Mass (g)", np.nan),
                    diameter_mm=row.get("Diameter (mm)", np.nan),
                    substrate=row.get("Substrate", ""),
                    active_pct=row.get("Active Material %", np.nan),
                )
                db.at[idx, "Mass Loading (mg/cm²)"] = total_ml
                db.at[idx, "Active ML (mg/cm²)"] = active_ml
            # Persist back
            try:
                save_database(db, db_file)
            except Exception:
                pass

        # Filter rows
        use_col = "Active ML (mg/cm²)"  # better predictor for capacity matching
        df = db[
            (db["Active Material"].astype(str).str.strip() == str(material).strip())
            & (~db["Blade Height (µm)"].isna())
            & (~db[use_col].isna())
        ].copy()

        if substrate.strip():
            df = df[df["Substrate"].astype(str).str.strip().str.lower() == substrate.strip().lower()]
        if tol_active > 0:
            df = df[df["Active Material %"].fillna(0) >= tol_active]
        if tol_solid > 0:
            df = df[df["Solid %"].fillna(0) >= tol_solid]

        if df.empty:
            st.error("No matching rows after filters. Relax filters or add data.")
            st.stop()

        # X: blade height (µm). Convert to "10 µm ticks" to align with earlier visuals
        x_ticks = df["Blade Height (µm)"].astype(float) / 10.0
        y_ml = df[use_col].astype(float)

        target_loading = st.number_input("Target Active Mass Loading (mg/cm²)", min_value=0.0, step=0.1, value=float(np.nan_to_num(y_ml.median(), nan=2.0)))
        slope, intercept, *_ = linregress(x_ticks, y_ml)
        required_tick = (target_loading - intercept) / slope if slope != 0 else None
        required_um = required_tick * 10.0 if required_tick is not None else None

        min_m, max_m = y_ml.min(), y_ml.max()
        if min_m <= target_loading <= max_m:
            confidence, uncertainty = "High", 0.10
        elif target_loading >= min_m * 0.75 and target_loading <= max_m * 1.25:
            confidence, uncertainty = "Medium", 0.25
        else:
            confidence, uncertainty = "Low", 0.50

        color = COLORS.get(material, 'black')
        plot_regression(
            x_ticks.values, y_ml.values,
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=f"{material} (DB)",
            color=color,
            target_y=target_loading,
            recommended_x=required_tick,
            x_is_ticks=True
        )

        if required_um is not None and np.isfinite(required_um):
            st.success(f"Recommended Blade Height: **{required_um:.1f} µm**  "
                       f"(= {required_tick:.2f} × 10 µm)")
            st.info(f"Confidence: **{confidence}** | Estimated Uncertainty: ±{uncertainty*required_um:.1f} µm")
        else:
            st.error("Slope is zero or invalid. Cannot compute recommendation.")

        with st.expander("Show regression numbers"):
            st.write(f"slope = {slope:.6f} (mg/cm² per 10 µm)")
            st.write(f"intercept = {intercept:.6f} (mg/cm²)")
            st.write(f"R² = {linregress(x_ticks, y_ml).rvalue**2:.4f}")

    else:
        # Built-in dataset mode (legacy)
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
        for t, masses in samples.items():
            avg_m = np.mean(masses)
            # Convert to active mass loading mg/cm²
            ml = (avg_m - foil_mass) * (active_ratio / 100.0) / ELECTRODE_AREA_13MM * 1000.0
            thicknesses.append(t)
            mass_loadings.append(ml)

        target_loading = st.number_input("Target Active Mass Loading (mg/cm²)", 0.0, 100.0, value=2.0, step=0.1)
        slope, intercept, *_ = linregress(thicknesses, mass_loadings)
        required_thickness = (target_loading - intercept) / slope if slope != 0 else None

        min_m, max_m = min(mass_loadings), max(mass_loadings)
        if min_m <= target_loading <= max_m:
            confidence, uncertainty = "High", 0.10
        elif target_loading >= min_m * 0.75 and target_loading <= max_m * 1.25:
            confidence, uncertainty = "Medium", 0.25
        else:
            confidence, uncertainty = "Low", 0.50

        color = COLORS.get(material, 'black')
        plot_regression(
            np.array(thicknesses), np.array(mass_loadings),
            xlabel="Blade Height (10 μm)", ylabel="Active ML (mg/cm²)",
            material_label=material, color=color,
            target_y=target_loading, recommended_x=required_thickness, x_is_ticks=True
        )

        if required_thickness is not None and np.isfinite(required_thickness):
            st.success(f"Recommended Blade Height: **{required_thickness*10.0:.1f} µm**  "
                       f"(= {required_thickness:.2f} × 10 µm)")
            st.info(f"Confidence: **{confidence}** | Estimated Uncertainty: ±{(uncertainty*required_thickness*10):.1f} µm")
        else:
            st.error("Could not calculate blade height (slope is zero).")

    if src == "Use Database" and not df.empty:
        with st.expander("📘 Blade Height Calculation Details"):
            st.markdown("### Regression Analysis")
            st.markdown(f"""
            **Dataset:**
            - Material: {material}
            - Substrate: {substrate if substrate else 'Any'}
            - Data points: {len(df)}
            - Active ML range: {y_ml.min():.2f} to {y_ml.max():.2f} mg/cm²
            - Blade height range: {x_ticks.min()*10:.1f} to {x_ticks.max()*10:.1f} µm
            """)
            
            st.markdown("**Linear Regression:**")
            st.markdown(f"""
            Equation: y = slope × x + intercept  
            Where:  
            - y = Active Mass Loading (mg/cm²)  
            - x = Blade Height (in 10µm units)
            
            Calculated:  
            - Slope: {slope:.4f} mg/cm² per 10µm  
            - Intercept: {intercept:.4f} mg/cm²  
            - R²: {r**2:.4f}
            """)
            
            st.markdown(f"""
            **Target Calculation:**  
            Required Active ML = {target_loading:.2f} mg/cm²  
            Solving for x:  
            x = (y - intercept) / slope  
              = ({target_loading:.2f} - {intercept:.4f}) / {slope:.4f}  
              = {required_tick:.2f} (in 10µm units)  
            
            Final Blade Height = {required_tick:.2f} × 10 = {required_um:.1f} µm
            """)



# =========================
# Capacity Match Tool (DB-aware)
# =========================
def capacity_match_tool():
    st.title("Capacity Match Tool")
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
        capacity_ratio = st.number_input("Anode/Cathode ratio (N/P)", 0.1, 5.0, value=1.0, step=0.01, format="%.2f")
    else:
        capacity_ratio = st.number_input("Cathode/Anode ratio (P/N)", 0.1, 5.0, value=1.0, step=0.01, format="%.2f")

    # Target side properties
    target_material = st.text_input(f"{side_target} material", value="Graphite" if side_target=="Anode" else "LVP")
    target_active_ratio = MATERIAL_ACTIVE_RATIOS.get(target_material, 0.96 if side_target=="Cathode" else 0.80)
    target_specific_capacity = st.number_input(f"{side_target} Specific Capacity (mAh/g)", 0.0, 1000.0, value=350.0 if side_target=="Anode" else 120.0, step=0.1)

    # Areal capacities
    known_areal_capacity = known_ml * known_specific_capacity / 1000.0  # mAh/cm²
    target_areal_capacity = known_areal_capacity * capacity_ratio

    # Required target active mass loading
    required_target_ml = target_areal_capacity / target_specific_capacity * 1000.0  # mg/cm² of active
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
            st.stop()

        substrate_filter = st.text_input("Filter by substrate (optional, exact match)", value="")
        df = db[
            (db["Active Material"].astype(str).str.strip().str.lower() == target_material.strip().lower())
            & (~db["Blade Height (µm)"].isna())
        ].copy()

        # compute active ML if missing
        miss = df["Active ML (mg/cm²)"].isna()
        if miss.any():
            for idx in df.index[miss]:
                row = df.loc[idx]
                total_ml, active_ml = calc_mass_loading_total_and_active(
                    disk_mass_g=row.get("Mass (g)", np.nan),
                    diameter_mm=row.get("Diameter (mm)", np.nan),
                    substrate=row.get("Substrate", ""),
                    active_pct=row.get("Active Material %", np.nan),
                )
                df.at[idx, "Mass Loading (mg/cm²)"] = total_ml
                df.at[idx, "Active ML (mg/cm²)"] = active_ml

        if substrate_filter.strip():
            df = df[df["Substrate"].astype(str).str.strip().str.lower() == substrate_filter.strip().lower()]

        df = df.dropna(subset=["Active ML (mg/cm²)", "Blade Height (µm)"])
        if df.empty:
            st.error("No usable rows for regression after filters.")
            st.stop()

        x_ticks = df["Blade Height (µm)"].astype(float) / 10.0
        y_active_ml = df["Active ML (mg/cm²)"].astype(float)
        slope, intercept, *_ = linregress(x_ticks, y_active_ml)
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
            st.success(f"Recommended {side_target} blade height: **{req_um:.1f} µm**")
        else:
            st.error("Regression invalid. Cannot recommend height.")

    else:
        # Built-in data
        if side_target == "Cathode":
            if target_material not in CATHODE_DATA:
                st.error("Target material not found in built-in cathode dataset.")
                st.stop()
            samples = CATHODE_DATA[target_material]
            foil_mass = AL_FOIL_MASS_13MM
        else:
            if target_material not in ANODE_DATA:
                st.error("Target material not found in built-in anode dataset.")
                st.stop()
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
            st.success(f"Recommended {side_target} blade height: **{req_tick*10.0:.1f} µm**")
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
        
        if src == "Use Database" and not df.empty:
            st.markdown("**Blade Height Estimation:**")
            st.markdown(f"""
            Using database regression for {target_material}:
            - Slope: {slope:.4f} mg/cm² per 10µm
            - Intercept: {intercept:.4f} mg/cm²
            - R²: {r**2:.4f}
            
            Required Blade Height Calculation:
            x = (y - intercept) / slope  
              = ({required_target_ml:.3f} - {intercept:.4f}) / {slope:.4f}  
              = {req_tick:.2f} (in 10µm units)  
            
            Final Blade Height = {req_tick:.2f} × 10 = {req_um:.1f} µm
            """)


# =========================
# Coating Calibration Tool
# =========================

def coating_calibration_tool():
    st.title("Coating Calibration Tool")
    st.markdown("Create a custom coating matrix to visualize coating uniformity.")

    batch_mode = st.checkbox(
        "Enable Batch Mode (Compare Multiple Coating Runs)",
        help="Toggle to allow input of multiple batches for comparison."
    )
    n_batches = st.number_input("Number of batches", 1, 5, 1, step=1) if batch_mode else 1

    n_rows = st.number_input("Number of rows", 1, 10, 3, step=1, help="Number of rows in your coating matrix.")
    n_cols = st.number_input("Number of columns", 1, 10, 3, step=1, help="Number of columns in your coating matrix.")

    # Global Parameters in Expander
    with st.expander("Global Parameters", expanded=False):
        g_active = st.text_input("Active Material", "LVP", help="Name of the active material used.")
        g_substrate = st.text_input("Substrate", "Aluminum foil", help="Type of substrate being coated.")
        g_diameter = st.number_input(
            "Diameter (mm)", 0.1, 100.0, 13.0, step=0.00001, format="%.5f",
            help="Diameter of the coated disc in mm."
        )
        g_solid = st.number_input(
            "Solid (%)", 0.0, 100.0, 30.0, step=0.00001, format="%.5f",
            help="Solid content of the coating solution."
        )
        g_active_pct = st.number_input(
            "Active Material (%)", 0.0, 100.0, 96.0, step=0.00001, format="%.5f",
            help="Percentage of active material in the solid content."
        )
        g_blade = st.number_input(
            "Blade Height (µm)", 1, 1000, 200, step=1,
            help="Height of the coating blade used for spreading."
        )

    sensitivity = st.slider(
        "Heatmap Sensitivity (%)", 1, 100, 10,
        help="Controls how sensitive the heatmap colors are to deviations."
    )

    # Heatmap mode selection
    heatmap_mode = st.radio(
        "Select Heatmap Type",
        ["Deviation", "Critical Deviation", "Spec/Out-of-Spec"],
        help="Select the type of heatmap to display: "
             "Deviation shows % deviation from mean; "
             "Critical Deviation highlights cells exceeding the critical threshold; "
             "Spec/Out-of-Spec highlights cells outside the specification range."
    )

    # Show relevant flag input based on selected heatmap mode
    if heatmap_mode == "Critical Deviation":
        crit_value = st.number_input(
            "Critical Deviation (%)",
            0.0, 100.0, 5.0, step=0.1,
            help="Cells with deviation from mean exceeding this percentage will be flagged."
        )
    elif heatmap_mode == "Spec/Out-of-Spec":
        spec_min = st.number_input(
            "Spec Min Active ML (mg/cm²)",
            0.0, 10.0, 1.5, step=0.01,
            help="Minimum acceptable active ML; below this will be flagged."
        )
        spec_max = st.number_input(
            "Spec Max Active ML (mg/cm²)",
            0.0, 10.0, 2.5, step=0.01,
            help="Maximum acceptable active ML; above this will be flagged."
        )

    batches = []
    for batch_num in range(n_batches):
        if batch_mode:
            st.subheader(f"Batch {batch_num + 1}")
        use_global_batch = st.checkbox(f"Use global parameters for Batch {batch_num+1}", True)

        matrix = {}
        for row in range(n_rows):
            cols_layout = st.columns(n_cols + 1)
            with cols_layout[0]:
                st.markdown(f"**Row {row+1}**")
            for col in range(n_cols):
                if not use_global_batch:
                    with cols_layout[col + 1]:
                        active = st.text_input("Active Material", g_active, key=f"act_{batch_num}_{row}_{col}")
                        substrate = st.text_input("Substrate", g_substrate, key=f"sub_{batch_num}_{row}_{col}")
                        diameter = st.number_input(
                            "Diameter (mm)", 0.1, 100.0, g_diameter, step=0.00001, format="%.5f",
                            key=f"dia_{batch_num}_{row}_{col}"
                        )
                        solid = st.number_input(
                            "Solid (%)", 0.0, 100.0, g_solid, step=0.00001, format="%.5f",
                            key=f"sol_{batch_num}_{row}_{col}"
                        )
                        active_pct = st.number_input(
                            "Active Material (%)", 0.0, 100.0, g_active_pct, step=0.00001,
                            format="%.5f", key=f"actp_{batch_num}_{row}_{col}"
                        )
                        blade = st.number_input(
                            "Blade Height (µm)", 1, 1000, g_blade, step=1, key=f"bl_{batch_num}_{row}_{col}"
                        )
                else:
                    active, substrate, diameter, solid, active_pct, blade = g_active, g_substrate, g_diameter, g_solid, g_active_pct, g_blade

                with cols_layout[col + 1]:
                    mass = st.number_input(
                        "Measured Mass (g)", 0.0, 10.0, 0.02000, step=0.00001, format="%.5f",
                        key=f"m_{batch_num}_{row}_{col}"
                    )
                    total_ml, active_ml = calc_mass_loading_total_and_active(mass, diameter, substrate, active_pct)

                    data_flag = ""
                    if mass <= 0:
                        data_flag = "MISSING MASS"

                    matrix[(row, col)] = {
                        'active': active,
                        'substrate': substrate,
                        'diameter': diameter,
                        'solid': solid,
                        'active_pct': active_pct,
                        'blade': blade,
                        'mass': mass,
                        'total_ml': total_ml,
                        'active_ml': active_ml,
                        'flag': data_flag
                    }
        batches.append(matrix)

    if st.button("Generate Analysis"):
        combined_writer = BytesIO()
        with pd.ExcelWriter(combined_writer, engine='openpyxl') as writer:
            for i, matrix in enumerate(batches):
                ml_values = np.zeros((n_rows, n_cols))
                for (row, col), data in matrix.items():
                    ml_values[row, col] = data['active_ml'] if not np.isnan(data['active_ml']) else 0

                mean_ml = np.nanmean(ml_values[ml_values > 0])
                deviations = ((ml_values - mean_ml) / mean_ml) * 100

                row_labels = [f"Row {r+1}" for r in range(n_rows)]
                col_labels = [f"Col {c+1}" for c in range(n_cols)]

                # Auto-flagging based on selected heatmap mode
                for (r, c), data in matrix.items():
                    if heatmap_mode == "Critical Deviation" and abs(deviations[r, c]) > crit_value:
                        data['flag'] = data['flag'] + "; CRITICAL DEV" if data['flag'] else "CRITICAL DEV"
                    elif heatmap_mode == "Spec/Out-of-Spec" and (data['active_ml'] < spec_min or data['active_ml'] > spec_max):
                        data['flag'] = data['flag'] + "; OUT OF SPEC" if data['flag'] else "OUT OF SPEC"

                st.subheader(f"Batch {i+1} Heatmap")

                # Heatmap plotting
                if heatmap_mode == "Deviation":
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=deviations, x=col_labels, y=row_labels,
                        text=np.round(deviations, 2),
                        hovertemplate="Row: %{y}<br>Col: %{x}<br>Deviation: %{text} %<extra></extra>",
                        colorscale='RdBu', zmid=0  # blue-red for deviation
                    ))
                    st.plotly_chart(fig_heat)

                elif heatmap_mode == "Critical Deviation":
                    crit_map = np.full_like(ml_values, np.nan)
                    for r in range(n_rows):
                        for c in range(n_cols):
                            if abs(deviations[r, c]) > crit_value:
                                crit_map[r, c] = deviations[r, c]
                    fig_crit = go.Figure(data=go.Heatmap(
                        z=crit_map, x=col_labels, y=row_labels,
                        text=np.round(crit_map, 2),
                        hovertemplate="Row: %{y}<br>Col: %{x}<br>Deviation: %{text} %<extra></extra>",
                        colorscale='Oranges',  # orange scale for critical deviations
                        showscale=True
                    ))
                    st.plotly_chart(fig_crit)

                else:  # Spec/Out-of-Spec
                    spec_map = np.full_like(ml_values, np.nan)
                    for r in range(n_rows):
                        for c in range(n_cols):
                            if ml_values[r, c] < spec_min or ml_values[r, c] > spec_max:
                                spec_map[r, c] = ml_values[r, c]
                    fig_spec = go.Figure(data=go.Heatmap(
                        z=spec_map, x=col_labels, y=row_labels,
                        text=np.round(spec_map, 2),
                        hovertemplate="Row: %{y}<br>Col: %{x}<br>Active ML: %{text} mg/cm²<extra></extra>",
                        colorscale='RdYlGn_r',  # red-green reversed for out-of-spec emphasis
                        showscale=True
                    ))
                    st.plotly_chart(fig_spec)

                # 3D Surface plot
                mean_ml = np.nanmean(ml_values[ml_values > 0])  # calculate mean excluding zeros/nans
                mean_plane = np.full_like(ml_values, mean_ml)

                fig_3d = go.Figure()

                # Surface of actual measurements
                fig_3d.add_trace(go.Surface(
                    z=ml_values,
                    x=np.arange(1, n_cols+1),
                    y=np.arange(1, n_rows+1),
                    colorscale='Viridis',
                    name='Measured ML',
                    showscale=True
                ))

                # Mean plane
                fig_3d.add_trace(go.Surface(
                    z=mean_plane,
                    x=np.arange(1, n_cols+1),
                    y=np.arange(1, n_rows+1),
                    surfacecolor=np.full_like(mean_plane, mean_ml),
                    colorscale=[[0, 'grey'], [1, 'grey']],  # fixed color
                    showscale=False,
                    opacity=0.4,
                    name='Mean ML'
                ))

                # Adjust layout to compress Y-axis and make Z less exaggerated
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="Columns",
                        yaxis_title="Rows",
                        zaxis_title="Active ML (mg/cm²)",
                        aspectratio=dict(x=1, y=1, z=0.3),  # x, y, and z aspect ratios
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))  # optional nicer angle
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                st.plotly_chart(fig_3d)

                # Uniformity and stats
                valid_ml = [v['active_ml'] for v in matrix.values() if not np.isnan(v['active_ml'])]
                if valid_ml:
                    uniformity_score = 100*(1 - (np.std(valid_ml)/np.mean(valid_ml)))
                    st.metric("Uniformity Score", f"{uniformity_score:.2f}%")
                    st.write("**Statistical Summary**")
                    st.markdown(f"""
                    - Mean: {np.mean(valid_ml):.5f} mg/cm²
                    - Std Dev: {np.std(valid_ml):.5f} mg/cm²
                    - Range: {np.min(valid_ml):.5f} - {np.max(valid_ml):.5f} mg/cm²
                    - Max Deviation: {np.max(np.abs(deviations)):.5f}%
                    """)

                # Excel export
                df_data = []
                for (row, col), data in matrix.items():
                    df_data.append({
                        'Row': row+1,
                        'Column': col+1,
                        'Active Material': data['active'],
                        'Substrate': data['substrate'],
                        'Diameter (mm)': f"{data['diameter']:.5f}",
                        'Solid (%)': f"{data['solid']:.5f}",
                        'Active Material (%)': f"{data['active_pct']:.5f}",
                        'Blade Height (µm)': data['blade'],
                        'Mass (g)': f"{data['mass']:.5f}",
                        'Total ML (mg/cm²)': f"{data['total_ml']:.5f}",
                        'Active ML (mg/cm²)': f"{data['active_ml']:.5f}",
                        'Deviation (%)': f"{deviations[row,col]:.2f}",
                        'Flag': data['flag']
                    })

                df = pd.DataFrame(df_data)

                # Add stats as separate rows at the bottom
                stats_rows = pd.DataFrame([{
                    'Row': 'STAT',
                    'Column': '',
                    'Active Material': '',
                    'Substrate': '',
                    'Diameter (mm)': '',
                    'Solid (%)': '',
                    'Active Material (%)': '',
                    'Blade Height (µm)': '',
                    'Mass (g)': '',
                    'Total ML (mg/cm²)': '',
                    'Active ML (mg/cm²)': f"{np.mean(valid_ml):.5f}",
                    'Deviation (%)': f"Std: {np.std(valid_ml):.5f}, Range: {np.min(valid_ml):.5f}-{np.max(valid_ml):.5f}, Max Dev: {np.max(np.abs(deviations)):.5f}%, Uniformity: {uniformity_score:.2f}%",
                    'Flag': ''
                }])
                df = pd.concat([df, stats_rows], ignore_index=True)

                df.to_excel(writer, sheet_name=f"Batch_{i+1}", index=False)

                st.download_button(
                    "Download All Batches Excel",
                    data=combined_writer.getvalue(),
                    file_name="coating_calibration_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )



# =========================
# Electrode Database Manager
# =========================
def electrode_database_manager():
    st.title("Electrode Database Manager")

    # Load or upload DB
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Load / View")
        db = load_database(db_file)
        if db.empty:
            st.info("Database empty.")
        st.dataframe(db, use_container_width=True)

    with c2:
        st.subheader("Import Excel")
        up = st.file_uploader("Upload an existing database (.xlsx)", type=["xlsx"])
        if up is not None:
            try:
                new_db = pd.read_excel(up)
                new_db = ensure_db_columns(new_db)
                save_database(new_db, db_file)
                st.success("Database replaced from upload.")
                db = new_db
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.subheader("Add New Electrode Entry")
    with st.form("new_entry_form", clear_on_submit=True):  # This is the main form
        c1, c2, c3 = st.columns(3)
        with c1:
            active_material = st.text_input("Active Material", value="LVP")
            substrate = st.text_input("Substrate", value="Aluminum foil")
            diameter_mm = st.number_input("Diameter (mm)", 0.0, 100.0, value=13.0, step=0.01)
        with c2:
            mass_g = st.number_input("Disk Mass (g)", 0.0, 10.0, value=0.0200, step=0.0001)
            solid_pct = st.number_input("Solid %", 0.0, 100.0, value=30.0, step=0.1)
            blade_um = st.number_input("Blade Height (µm)", 0.0, 1000.0, value=200.0, step=1.0)
        with c3:
            active_pct = st.number_input("Active Material %", 0.0, 100.0, value=96.0, step=0.1)

            # auto-calc ML
            total_ml, active_ml = calc_mass_loading_total_and_active(mass_g, diameter_mm, substrate, active_pct)

        st.caption("Mass loading auto-calculated using foil subtraction when substrate is recognized.")
        if pick_foil_mass(substrate) is None:
            st.warning("Substrate not recognized. Foil mass not subtracted. Update substrate text to a known key.")

        submitted = st.form_submit_button("Add entry")
        if submitted:
            new_row = {
                "Active Material": active_material,
                "Substrate": substrate,
                "Diameter (mm)": diameter_mm,
                "Mass (g)": mass_g,
                "Solid %": solid_pct,
                "Blade Height (µm)": blade_um,
                "Active Material %": active_pct,
                "Mass Loading (mg/cm²)": total_ml,
                "Active ML (mg/cm²)": active_ml,
            }
            db = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
            save_database(db, db_file)
            st.success("Row added.")

    # Move the math breakdown outside the form
    with st.expander("📘 Mass Loading Calculation Details"):
        st.markdown("### Mass Loading Formulas")
        # You'll need to get the last entered values from session state or variables
        # For demonstration, I'll show the structure - you'll need to adapt based on your data flow
        st.markdown(f"""
        **Inputs:**
        - Disk Mass: {mass_g if 'mass_g' in locals() else 0.0200:.4f} g
        - Diameter: {diameter_mm if 'diameter_mm' in locals() else 13.0:.2f} mm
        - Substrate: {substrate if 'substrate' in locals() else 'Aluminum foil'}
        - Active Material %: {active_pct if 'active_pct' in locals() else 96.0:.1f}%
        """)
        
        foil_mass = pick_foil_mass(substrate if 'substrate' in locals() else 'Aluminum foil')
        if foil_mass is not None:
            st.markdown(f"""
            **Calculations:**
            1. Area = π × (diameter/2)² / 100  
               = π × ({diameter_mm if 'diameter_mm' in locals() else 13.0:.2f}/2)² / 100  
               = {mm_diameter_to_area_cm2(diameter_mm if 'diameter_mm' in locals() else 13.0):.4f} cm²
            
            2. Coating Mass = Disk Mass - Foil Mass  
               = {mass_g if 'mass_g' in locals() else 0.0200:.4f} - {foil_mass:.6f}  
               = {(mass_g - foil_mass) if 'mass_g' in locals() else (0.0200 - foil_mass):.6f} g
            
            3. Total Mass Loading = (Coating Mass × 1000) / Area  
               = {(mass_g - foil_mass) if 'mass_g' in locals() else (0.0200 - foil_mass):.6f} × 1000 / {mm_diameter_to_area_cm2(diameter_mm if 'diameter_mm' in locals() else 13.0):.4f}  
               = {total_ml if 'total_ml' in locals() else ((0.0200 - foil_mass)*1000/mm_diameter_to_area_cm2(13.0)):.2f} mg/cm²
            
            4. Active Mass Loading = Total ML × (Active % / 100)  
               = {total_ml if 'total_ml' in locals() else ((0.0200 - foil_mass)*1000/mm_diameter_to_area_cm2(13.0)):.2f} × ({active_pct if 'active_pct' in locals() else 96.0}/100)  
               = {active_ml if 'active_ml' in locals() else ((0.0200 - foil_mass)*1000/mm_diameter_to_area_cm2(13.0)*(96.0/100)):.2f} mg/cm²
            """)
        else:
            st.markdown("""
            **Warning:** Substrate not recognized - using total disk mass for calculations.
            
            1. Area = π × (diameter/2)² / 100  
               = π × (13.00/2)² / 100  
               = 1.3273 cm²
            
            2. Total Mass Loading = (Disk Mass × 1000) / Area  
               = (0.0200 × 1000) / 1.3273  
               = 15.07 mg/cm²
            
            3. Active Mass Loading = Total ML × (Active % / 100)  
               = 15.07 × (96/100)  
               = 14.47 mg/cm²
            """)

    # Rest of your function remains the same...
    st.subheader("Edit / Delete Rows")
    if not db.empty:
        db_edit = db.copy()
        db_edit = st.dataframe(db_edit, use_container_width=True)
        # Simple delete by index
        del_idx = st.number_input("Delete row index", min_value=0, max_value=max(len(db)-1, 0), value=0, step=1)
        if st.button("Delete row"):
            db2 = load_database(db_file)
            if 0 <= del_idx < len(db2):
                db2 = db2.drop(db2.index[del_idx]).reset_index(drop=True)
                save_database(db2, db_file)
                st.success(f"Deleted row {del_idx}.")
            else:
                st.error("Invalid index.")

    st.subheader("Export")
    db_current = pd.read_excel(db_file)

    from difflib import get_close_matches

    valid_active_materials = ['LVP', 'NVP', 'Li3V2(PO4)3', 'Na3V2(PO4)3', 'graphite', 'hard-carbon']
    valid_substrates = ['aluminum', 'copper', 'hard-carbon']

    def autocorrect_value(value, valid_list):
        if pd.isna(value):
            return value, None
        value_str = str(value).strip()
        match = get_close_matches(value_str.lower(), [v.lower() for v in valid_list], n=1, cutoff=0.6)
        if match:
            corrected = next(v for v in valid_list if v.lower() == match[0])
            if corrected != value_str:
                return corrected, f"Corrected '{value_str}' to '{corrected}'"
        return value_str, None

    corrections_log = []

    for col, valid_list in [('Active Material', valid_active_materials), ('Substrate', valid_substrates)]:
        for idx, val in enumerate(db_current[col]):
            corrected_val, note = autocorrect_value(val, valid_list)
            db_current.at[idx, col] = corrected_val
            if note:
                corrections_log.append({'Row': idx + 2, 'Column': col, 'Note': note})
    data_bytes, fname = bytes_excel(db_current, filename=os.path.basename(db_file))
    st.download_button("Download current database (.xlsx)", data=data_bytes, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


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
