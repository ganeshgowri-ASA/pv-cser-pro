"""
PV-CSER Pro - Climate Specific Energy Rating System.

A Streamlit application for PV module performance characterization
according to IEC 61853 standards.

Features:
- Power matrix data input (22-point grid: 7 irradiances × 4 temperatures)
- IEC 61853-1 compliant calculations
- 3D surface visualizations for P(G,T)
- Climate profile selection (6 standard IEC profiles)
- CSER rating calculations
- Export capabilities (CSV, Excel)

Reference:
    IEC 61853-1:2011 - Irradiance and temperature performance
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-4:2018 - Standard reference climatic profiles
"""

import io
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Import local modules
from src.utils.constants import (
    IRRADIANCES,
    TEMPERATURES,
    STANDARD_CONDITIONS,
    CLIMATE_PROFILES,
    ClimateProfile,
    TEMP_COEFFICIENTS_BY_TECHNOLOGY,
    create_empty_power_matrix,
    get_climate_profile_names,
)
from src.calculations.iec61853_1 import (
    calculate_temperature_coefficients,
    interpolate_power_at_conditions,
    calculate_cser,
    calculate_monthly_energy,
    validate_power_matrix,
    calculate_efficiency,
    calculate_fill_factor,
    TemperatureCoefficients,
)
from src.visualizations.plots_3d import (
    create_power_surface_plot,
    create_efficiency_surface_plot,
    create_dual_surface_plot,
    create_power_vs_irradiance_plot,
    create_power_vs_temperature_plot,
    create_temperature_coefficients_plot,
    create_monthly_energy_plot,
    create_climate_comparison_plot,
    create_heatmap_plot,
    create_performance_gauge,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PV-CSER Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/pv-cser-pro/docs',
        'Report a bug': 'https://github.com/pv-cser-pro/issues',
        'About': '# PV-CSER Pro\nClimate Specific Energy Rating System\nIEC 61853 Compliant'
    }
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Module information
        'module_name': 'Sample PV Module',
        'manufacturer': 'Generic',
        'model_number': 'PV-400',
        'technology_type': 'Mono-Si',
        'module_area': 1.7,
        'cells_in_series': 60,

        # STC parameters
        'pmax_stc': 400.0,
        'voc_stc': 49.5,
        'isc_stc': 10.2,
        'vmp_stc': 41.0,
        'imp_stc': 9.76,

        # Temperature coefficients (will be calculated)
        'alpha_isc': 0.05,
        'beta_voc': -0.29,
        'gamma_pmax': -0.35,

        # Power matrix data
        'power_matrix': create_empty_power_matrix(),
        'isc_matrix': create_empty_power_matrix(),
        'voc_matrix': create_empty_power_matrix(),
        'matrix_initialized': False,

        # Climate data
        'selected_climate': 'Subtropical Arid',
        'hourly_irradiance': None,
        'hourly_temperature': None,

        # CSER results
        'cser_results': None,
        'monthly_energy': None,

        # Navigation
        'current_page': 'Dashboard',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# Sidebar Navigation
# =============================================================================

def render_sidebar():
    """Render the sidebar with navigation and module info."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=PV-CSER+Pro", width=200)
        st.markdown("---")

        # Navigation
        st.subheader("📍 Navigation")
        pages = [
            ("📊 Dashboard", "Dashboard"),
            ("📝 Data Input", "Data Input"),
            ("🔢 Calculations", "Calculations"),
            ("📈 Visualizations", "Visualizations"),
            ("🌍 Climate/CSER", "Climate/CSER"),
            ("📄 Reports", "Reports"),
        ]

        for icon_label, page_name in pages:
            if st.button(icon_label, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()

        st.markdown("---")

        # Quick Module Info
        st.subheader("📋 Module Info")
        st.caption(f"**Name:** {st.session_state.module_name}")
        st.caption(f"**Pmax:** {st.session_state.pmax_stc:.1f} W")
        st.caption(f"**Technology:** {st.session_state.technology_type}")

        # Matrix status
        matrix_valid, _ = validate_power_matrix(st.session_state.power_matrix, min_valid_points=10)
        status = "✅ Ready" if matrix_valid else "⚠️ Incomplete"
        st.caption(f"**Matrix:** {status}")

        st.markdown("---")
        st.caption("IEC 61853 Compliant")
        st.caption(f"v1.0.0 | {datetime.now().year}")


# =============================================================================
# Dashboard Page
# =============================================================================

def render_dashboard():
    """Render the main dashboard with KPIs and overview."""
    st.title("📊 Dashboard")
    st.markdown("### PV Module Performance Overview")

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Pmax (STC)",
            value=f"{st.session_state.pmax_stc:.1f} W",
            delta=None
        )

    with col2:
        efficiency_stc = calculate_efficiency(
            st.session_state.pmax_stc,
            1000.0,
            st.session_state.module_area
        )
        st.metric(
            label="Efficiency (STC)",
            value=f"{efficiency_stc:.2f}%",
            delta=None
        )

    with col3:
        st.metric(
            label="γ (Pmax)",
            value=f"{st.session_state.gamma_pmax:.3f} %/°C",
            delta=None
        )

    with col4:
        # Matrix completeness
        valid_points = np.sum(~np.isnan(st.session_state.power_matrix))
        total_points = st.session_state.power_matrix.size
        st.metric(
            label="Matrix Points",
            value=f"{valid_points}/{total_points}",
            delta=f"{(valid_points/total_points)*100:.0f}%"
        )

    st.markdown("---")

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Power Matrix Heatmap")
        if np.any(~np.isnan(st.session_state.power_matrix)):
            fig = create_heatmap_plot(
                st.session_state.power_matrix,
                title="P(G,T) Power Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter power matrix data in the Data Input page to see the heatmap.")

    with col_right:
        st.subheader("Module Specifications")

        specs_data = {
            "Parameter": ["Module Name", "Manufacturer", "Technology", "Area", "Cells",
                         "Pmax (STC)", "Voc (STC)", "Isc (STC)", "Vmp (STC)", "Imp (STC)"],
            "Value": [
                st.session_state.module_name,
                st.session_state.manufacturer,
                st.session_state.technology_type,
                f"{st.session_state.module_area:.2f} m²",
                str(st.session_state.cells_in_series),
                f"{st.session_state.pmax_stc:.2f} W",
                f"{st.session_state.voc_stc:.2f} V",
                f"{st.session_state.isc_stc:.2f} A",
                f"{st.session_state.vmp_stc:.2f} V",
                f"{st.session_state.imp_stc:.2f} A",
            ]
        }
        st.dataframe(pd.DataFrame(specs_data), hide_index=True, use_container_width=True)

        st.subheader("Temperature Coefficients")
        coeff_data = {
            "Coefficient": ["α (Isc)", "β (Voc)", "γ (Pmax)"],
            "Value": [
                f"{st.session_state.alpha_isc:+.4f} %/°C",
                f"{st.session_state.beta_voc:+.4f} %/°C",
                f"{st.session_state.gamma_pmax:+.4f} %/°C",
            ]
        }
        st.dataframe(pd.DataFrame(coeff_data), hide_index=True, use_container_width=True)

    # Standard conditions reference
    st.markdown("---")
    st.subheader("📌 IEC 61853 Standard Test Conditions")

    cond_cols = st.columns(5)
    for i, (abbrev, cond) in enumerate(STANDARD_CONDITIONS.items()):
        with cond_cols[i]:
            st.markdown(f"**{abbrev}**")
            st.caption(f"G: {cond.irradiance:.0f} W/m²")
            st.caption(f"T: {cond.temperature:.0f}°C")


# =============================================================================
# Data Input Page
# =============================================================================

def render_data_input():
    """Render the data input page for module parameters and power matrix."""
    st.title("📝 Data Input")

    tab1, tab2, tab3 = st.tabs(["Module Parameters", "Power Matrix", "Import Data"])

    # Tab 1: Module Parameters
    with tab1:
        st.subheader("Module Specifications")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.module_name = st.text_input(
                "Module Name",
                value=st.session_state.module_name
            )
            st.session_state.manufacturer = st.text_input(
                "Manufacturer",
                value=st.session_state.manufacturer
            )
            st.session_state.model_number = st.text_input(
                "Model Number",
                value=st.session_state.model_number
            )
            st.session_state.technology_type = st.selectbox(
                "Technology Type",
                options=list(TEMP_COEFFICIENTS_BY_TECHNOLOGY.keys()),
                index=list(TEMP_COEFFICIENTS_BY_TECHNOLOGY.keys()).index(
                    st.session_state.technology_type
                )
            )

        with col2:
            st.session_state.module_area = st.number_input(
                "Module Area (m²)",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.module_area,
                step=0.01
            )
            st.session_state.cells_in_series = st.number_input(
                "Cells in Series",
                min_value=1,
                max_value=200,
                value=st.session_state.cells_in_series,
                step=1
            )

        st.markdown("---")
        st.subheader("STC Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.session_state.pmax_stc = st.number_input(
                "Pmax (W)",
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.pmax_stc,
                step=0.1
            )
            st.session_state.voc_stc = st.number_input(
                "Voc (V)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.voc_stc,
                step=0.1
            )

        with col2:
            st.session_state.isc_stc = st.number_input(
                "Isc (A)",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.isc_stc,
                step=0.01
            )
            st.session_state.vmp_stc = st.number_input(
                "Vmp (V)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.vmp_stc,
                step=0.1
            )

        with col3:
            st.session_state.imp_stc = st.number_input(
                "Imp (A)",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.imp_stc,
                step=0.01
            )

            # Calculate and display fill factor
            ff = calculate_fill_factor(
                st.session_state.pmax_stc,
                st.session_state.isc_stc,
                st.session_state.voc_stc
            )
            st.metric("Fill Factor", f"{ff*100:.1f}%")

    # Tab 2: Power Matrix Input
    with tab2:
        st.subheader("IEC 61853-1 Power Matrix")
        st.caption("Enter measured Pmax values at each (G, T) condition")

        # Matrix selector
        matrix_type = st.radio(
            "Select Matrix Type",
            options=["Power (Pmax)", "Current (Isc)", "Voltage (Voc)"],
            horizontal=True
        )

        # Get the appropriate matrix
        if matrix_type == "Power (Pmax)":
            current_matrix = st.session_state.power_matrix
            matrix_key = 'power_matrix'
            unit = "W"
        elif matrix_type == "Current (Isc)":
            current_matrix = st.session_state.isc_matrix
            matrix_key = 'isc_matrix'
            unit = "A"
        else:
            current_matrix = st.session_state.voc_matrix
            matrix_key = 'voc_matrix'
            unit = "V"

        # Create editable dataframe
        df_matrix = pd.DataFrame(
            current_matrix,
            index=[f"{g} W/m²" for g in IRRADIANCES],
            columns=[f"{t}°C" for t in TEMPERATURES]
        )

        st.markdown(f"**{matrix_type} Matrix ({unit})**")

        edited_df = st.data_editor(
            df_matrix,
            use_container_width=True,
            num_rows="fixed",
            key=f"matrix_editor_{matrix_type}"
        )

        # Update session state with edited values
        st.session_state[matrix_key] = edited_df.values.astype(np.float64)

        # Quick fill options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Generate Sample Data", use_container_width=True):
                _generate_sample_matrix_data()
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Matrix", use_container_width=True):
                st.session_state[matrix_key] = create_empty_power_matrix()
                st.rerun()

        with col3:
            # Validation status
            is_valid, msg = validate_power_matrix(st.session_state.power_matrix)
            if is_valid:
                st.success("✅ Matrix valid")
            else:
                st.warning(f"⚠️ {msg}")

    # Tab 3: Import Data
    with tab3:
        st.subheader("Import Data from File")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Upload a file with power matrix data"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, index_col=0)
                else:
                    df = pd.read_excel(uploaded_file, index_col=0)

                st.write("Preview:")
                st.dataframe(df)

                if st.button("Import to Power Matrix"):
                    st.session_state.power_matrix = df.values.astype(np.float64)
                    st.success("Data imported successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Template download
        st.markdown("---")
        st.subheader("Download Template")

        template_df = pd.DataFrame(
            create_empty_power_matrix(),
            index=[f"{g}" for g in IRRADIANCES],
            columns=[f"{t}" for t in TEMPERATURES]
        )
        template_df.index.name = "Irradiance (W/m²)"

        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer)

        st.download_button(
            label="📥 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="power_matrix_template.csv",
            mime="text/csv"
        )


def _generate_sample_matrix_data():
    """Generate sample power matrix data based on STC values."""
    pmax_stc = st.session_state.pmax_stc
    isc_stc = st.session_state.isc_stc
    voc_stc = st.session_state.voc_stc
    gamma = st.session_state.gamma_pmax / 100
    alpha = st.session_state.alpha_isc / 100
    beta = st.session_state.beta_voc / 100

    power_matrix = np.zeros((len(IRRADIANCES), len(TEMPERATURES)))
    isc_matrix = np.zeros((len(IRRADIANCES), len(TEMPERATURES)))
    voc_matrix = np.zeros((len(IRRADIANCES), len(TEMPERATURES)))

    for i, g in enumerate(IRRADIANCES):
        for j, t in enumerate(TEMPERATURES):
            # Simple model for demonstration
            g_ratio = g / 1000.0
            t_diff = t - 25.0

            # Power with irradiance scaling and temperature correction
            power_matrix[i, j] = pmax_stc * g_ratio * (1 + gamma * t_diff)

            # Isc scales linearly with irradiance
            isc_matrix[i, j] = isc_stc * g_ratio * (1 + alpha * t_diff)

            # Voc with logarithmic irradiance dependence
            if g > 0:
                voc_matrix[i, j] = voc_stc * (1 + 0.025 * np.log(g_ratio + 0.01)) * (1 + beta * t_diff)

    st.session_state.power_matrix = power_matrix
    st.session_state.isc_matrix = isc_matrix
    st.session_state.voc_matrix = voc_matrix
    st.session_state.matrix_initialized = True


# =============================================================================
# Calculations Page
# =============================================================================

@st.cache_data
def cached_calculate_coefficients(power_matrix, isc_matrix, voc_matrix):
    """Cached temperature coefficient calculation."""
    return calculate_temperature_coefficients(
        power_matrix, isc_matrix, voc_matrix
    )


def render_calculations():
    """Render the calculations page."""
    st.title("🔢 Calculations")

    tab1, tab2, tab3 = st.tabs([
        "Temperature Coefficients",
        "Power Interpolation",
        "Performance Metrics"
    ])

    # Tab 1: Temperature Coefficients
    with tab1:
        st.subheader("Temperature Coefficient Calculation")
        st.caption("Per IEC 61853-1 Clause 9")

        if st.button("Calculate Coefficients from Matrix", type="primary"):
            if np.sum(~np.isnan(st.session_state.power_matrix)) >= 8:
                with st.spinner("Calculating..."):
                    coeffs = cached_calculate_coefficients(
                        st.session_state.power_matrix,
                        st.session_state.isc_matrix,
                        st.session_state.voc_matrix
                    )

                    st.session_state.alpha_isc = coeffs.alpha_isc
                    st.session_state.beta_voc = coeffs.beta_voc
                    st.session_state.gamma_pmax = coeffs.gamma_pmax

                st.success("Coefficients calculated!")
            else:
                st.error("Insufficient matrix data. Need at least 8 valid points.")

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "α (Isc)",
                f"{st.session_state.alpha_isc:+.4f} %/°C",
                help="Short-circuit current temperature coefficient"
            )

        with col2:
            st.metric(
                "β (Voc)",
                f"{st.session_state.beta_voc:+.4f} %/°C",
                help="Open-circuit voltage temperature coefficient"
            )

        with col3:
            st.metric(
                "γ (Pmax)",
                f"{st.session_state.gamma_pmax:+.4f} %/°C",
                help="Maximum power temperature coefficient"
            )

        # Coefficient visualization
        st.markdown("---")
        fig = create_temperature_coefficients_plot(
            st.session_state.alpha_isc,
            st.session_state.beta_voc,
            st.session_state.gamma_pmax
        )
        st.plotly_chart(fig, use_container_width=True)

        # Reference ranges
        st.markdown("---")
        st.subheader("Typical Coefficient Ranges by Technology")

        tech_data = []
        for tech, ranges in TEMP_COEFFICIENTS_BY_TECHNOLOGY.items():
            tech_data.append({
                "Technology": tech,
                "α (Isc) %/°C": f"{ranges.alpha_isc[0]:+.2f} to {ranges.alpha_isc[1]:+.2f}",
                "β (Voc) %/°C": f"{ranges.beta_voc[0]:+.2f} to {ranges.beta_voc[1]:+.2f}",
                "γ (Pmax) %/°C": f"{ranges.gamma_pmax[0]:+.2f} to {ranges.gamma_pmax[1]:+.2f}",
            })

        st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)

    # Tab 2: Power Interpolation
    with tab2:
        st.subheader("Power Interpolation at Custom Conditions")

        col1, col2 = st.columns(2)

        with col1:
            g_target = st.slider(
                "Target Irradiance (W/m²)",
                min_value=50,
                max_value=1200,
                value=800,
                step=10
            )

        with col2:
            t_target = st.slider(
                "Target Temperature (°C)",
                min_value=0,
                max_value=85,
                value=45,
                step=1
            )

        if st.button("Interpolate", type="primary"):
            if np.sum(~np.isnan(st.session_state.power_matrix)) >= 4:
                with st.spinner("Interpolating..."):
                    result = interpolate_power_at_conditions(
                        st.session_state.power_matrix,
                        st.session_state.isc_matrix,
                        st.session_state.voc_matrix,
                        g_target,
                        t_target,
                        module_area=st.session_state.module_area
                    )

                st.markdown("### Results")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Pmax", f"{result.pmax:.2f} W")
                with col2:
                    st.metric("Isc", f"{result.isc:.3f} A")
                with col3:
                    st.metric("Voc", f"{result.voc:.2f} V")
                with col4:
                    st.metric("Efficiency", f"{result.efficiency:.2f}%")

                st.caption(f"Method: {result.method}")
            else:
                st.error("Insufficient matrix data for interpolation.")

    # Tab 3: Performance Metrics
    with tab3:
        st.subheader("Standard Condition Performance")

        if np.sum(~np.isnan(st.session_state.power_matrix)) >= 4:
            results = []

            for abbrev, cond in STANDARD_CONDITIONS.items():
                try:
                    result = interpolate_power_at_conditions(
                        st.session_state.power_matrix,
                        st.session_state.isc_matrix,
                        st.session_state.voc_matrix,
                        cond.irradiance,
                        cond.temperature,
                        module_area=st.session_state.module_area
                    )
                    results.append({
                        "Condition": abbrev,
                        "G (W/m²)": cond.irradiance,
                        "T (°C)": cond.temperature,
                        "Pmax (W)": f"{result.pmax:.2f}",
                        "Efficiency (%)": f"{result.efficiency:.2f}",
                    })
                except Exception:
                    results.append({
                        "Condition": abbrev,
                        "G (W/m²)": cond.irradiance,
                        "T (°C)": cond.temperature,
                        "Pmax (W)": "N/A",
                        "Efficiency (%)": "N/A",
                    })

            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        else:
            st.info("Enter power matrix data to calculate performance metrics.")


# =============================================================================
# Visualizations Page
# =============================================================================

def render_visualizations():
    """Render the visualizations page with 3D plots."""
    st.title("📈 Visualizations")

    if np.sum(~np.isnan(st.session_state.power_matrix)) < 4:
        st.warning("Enter power matrix data to generate visualizations.")
        return

    tab1, tab2, tab3 = st.tabs(["3D Surfaces", "2D Plots", "Heatmaps"])

    # Tab 1: 3D Surface Plots
    with tab1:
        st.subheader("3D Power Surface P(G,T)")

        # Options
        col1, col2 = st.columns(2)
        with col1:
            show_contours = st.checkbox("Show contour projections", value=True)
        with col2:
            plot_height = st.slider("Plot height", 400, 800, 600)

        # Power surface
        fig_power = create_power_surface_plot(
            st.session_state.power_matrix,
            module_name=st.session_state.module_name,
            show_contours=show_contours,
            height=plot_height
        )
        st.plotly_chart(fig_power, use_container_width=True)

        st.markdown("---")

        # Efficiency surface
        st.subheader("3D Efficiency Surface η(G,T)")
        fig_eff = create_efficiency_surface_plot(
            st.session_state.power_matrix,
            st.session_state.module_area,
            module_name=st.session_state.module_name,
            height=plot_height
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    # Tab 2: 2D Plots
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Power vs Irradiance")
            fig = create_power_vs_irradiance_plot(st.session_state.power_matrix)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Power vs Temperature")
            fig = create_power_vs_temperature_plot(st.session_state.power_matrix)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Temperature coefficients
        st.subheader("Temperature Coefficient Behavior")
        fig = create_temperature_coefficients_plot(
            st.session_state.alpha_isc,
            st.session_state.beta_voc,
            st.session_state.gamma_pmax
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Heatmaps
    with tab3:
        st.subheader("Matrix Heatmaps")

        col1, col2 = st.columns(2)

        with col1:
            fig = create_heatmap_plot(
                st.session_state.power_matrix,
                title="Power Matrix (W)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Efficiency heatmap
            eff_matrix = np.zeros_like(st.session_state.power_matrix)
            for i, g in enumerate(IRRADIANCES):
                for j in range(len(TEMPERATURES)):
                    if g > 0 and not np.isnan(st.session_state.power_matrix[i, j]):
                        eff_matrix[i, j] = (
                            st.session_state.power_matrix[i, j] /
                            (g * st.session_state.module_area)
                        ) * 100

            fig = create_heatmap_plot(eff_matrix, title="Efficiency Matrix (%)")
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Climate/CSER Page
# =============================================================================

def render_climate_cser():
    """Render the Climate Specific Energy Rating page."""
    st.title("🌍 Climate Specific Energy Rating")
    st.caption("Per IEC 61853-3 and IEC 61853-4")

    tab1, tab2, tab3 = st.tabs(["Climate Profiles", "CSER Calculation", "Comparison"])

    # Tab 1: Climate Profiles
    with tab1:
        st.subheader("IEC 61853-4 Standard Climate Profiles")

        # Profile selector
        profile_names = get_climate_profile_names()
        selected_profile = st.selectbox(
            "Select Climate Profile",
            options=profile_names,
            index=profile_names.index(st.session_state.selected_climate)
        )
        st.session_state.selected_climate = selected_profile

        # Get profile data
        for profile, data in CLIMATE_PROFILES.items():
            if data.name == selected_profile:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Description:** {data.description}")
                    st.markdown(f"**Representative Location:** {data.representative_location}")

                with col2:
                    st.metric("Annual GHI", f"{data.annual_ghi[0]}-{data.annual_ghi[1]} kWh/m²")
                    st.metric("Avg Temperature", f"{data.avg_temp}°C")
                    st.metric("Temp Range", f"{data.temp_range[0]} to {data.temp_range[1]}°C")
                break

        # Display all profiles
        st.markdown("---")
        st.subheader("All Climate Profiles")

        profiles_data = []
        for profile, data in CLIMATE_PROFILES.items():
            profiles_data.append({
                "Profile": data.name,
                "Type": data.description[:30] + "...",
                "GHI (kWh/m²/yr)": f"{data.annual_ghi[0]}-{data.annual_ghi[1]}",
                "Avg T (°C)": data.avg_temp,
                "Location": data.representative_location.split("/")[0].strip(),
            })

        st.dataframe(pd.DataFrame(profiles_data), hide_index=True, use_container_width=True)

    # Tab 2: CSER Calculation
    with tab2:
        st.subheader("Calculate CSER Rating")

        if np.sum(~np.isnan(st.session_state.power_matrix)) < 10:
            st.warning("Complete the power matrix data before calculating CSER.")
            return

        # Generate synthetic hourly data for selected climate
        st.markdown("### Simulation Parameters")

        col1, col2 = st.columns(2)

        with col1:
            array_tilt = st.number_input("Array Tilt (°)", 0, 90, 30)
            system_losses = st.number_input("System Losses (%)", 0.0, 30.0, 14.0)

        with col2:
            array_azimuth = st.number_input("Array Azimuth (°)", 0, 360, 180)

        if st.button("Calculate CSER", type="primary"):
            with st.spinner("Calculating annual energy yield..."):
                # Generate synthetic hourly data
                hourly_g, hourly_t = _generate_synthetic_climate_data(
                    st.session_state.selected_climate
                )

                # Calculate CSER
                cser_results = calculate_cser(
                    st.session_state.power_matrix,
                    hourly_g,
                    hourly_t,
                    st.session_state.pmax_stc
                )

                # Apply system losses
                loss_factor = 1 - (system_losses / 100)
                cser_results['annual_energy'] *= loss_factor
                cser_results['cser_rating'] *= loss_factor

                st.session_state.cser_results = cser_results

                # Calculate monthly energy
                monthly = calculate_monthly_energy(
                    st.session_state.power_matrix,
                    hourly_g,
                    hourly_t
                )
                monthly = [m * loss_factor for m in monthly]
                st.session_state.monthly_energy = monthly

            st.success("CSER calculation complete!")

        # Display results
        if st.session_state.cser_results:
            st.markdown("---")
            st.subheader("Results")

            results = st.session_state.cser_results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Annual Energy", f"{results['annual_energy']:.1f} kWh")

            with col2:
                st.metric("CSER Rating", f"{results['cser_rating']:.0f} kWh/kWp")

            with col3:
                st.metric("Capacity Factor", f"{results['capacity_factor']*100:.1f}%")

            with col4:
                st.metric("Performance Ratio", f"{results['performance_ratio']*100:.1f}%")

            # Monthly chart
            if st.session_state.monthly_energy:
                st.markdown("---")
                fig = create_monthly_energy_plot(
                    st.session_state.monthly_energy,
                    climate_name=st.session_state.selected_climate
                )
                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Comparison
    with tab3:
        st.subheader("Climate Profile Comparison")

        if st.button("Calculate CSER for All Climates", type="primary"):
            with st.spinner("Calculating for all climate profiles..."):
                comparison_results = {}

                for profile, data in CLIMATE_PROFILES.items():
                    hourly_g, hourly_t = _generate_synthetic_climate_data(data.name)

                    try:
                        results = calculate_cser(
                            st.session_state.power_matrix,
                            hourly_g,
                            hourly_t,
                            st.session_state.pmax_stc
                        )
                        comparison_results[data.name] = results['cser_rating']
                    except Exception:
                        comparison_results[data.name] = 0

                # Store and display
                fig = create_climate_comparison_plot(comparison_results)
                st.plotly_chart(fig, use_container_width=True)

                # Table
                comp_df = pd.DataFrame([
                    {"Climate": k, "CSER (kWh/kWp)": f"{v:.0f}"}
                    for k, v in sorted(comparison_results.items(), key=lambda x: -x[1])
                ])
                st.dataframe(comp_df, hide_index=True, use_container_width=True)


def _generate_synthetic_climate_data(climate_name: str) -> tuple:
    """Generate synthetic hourly irradiance and temperature data for a climate profile."""
    # Get climate data
    profile_data = None
    for profile, data in CLIMATE_PROFILES.items():
        if data.name == climate_name:
            profile_data = data
            break

    if profile_data is None:
        profile_data = list(CLIMATE_PROFILES.values())[0]

    # Generate 8760 hours of synthetic data
    hours = np.arange(8760)
    day_of_year = (hours // 24) % 365

    # Seasonal variation
    seasonal = np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Daily variation (simplified)
    hour_of_day = hours % 24
    daily_pattern = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))

    # Irradiance
    avg_ghi = (profile_data.annual_ghi[0] + profile_data.annual_ghi[1]) / 2
    peak_irradiance = avg_ghi / 365 / 5  # Rough peak W/m²

    irradiance = peak_irradiance * daily_pattern * (1 + 0.3 * seasonal)
    irradiance = np.clip(irradiance, 0, 1200)

    # Add some randomness
    irradiance *= (0.8 + 0.4 * np.random.random(8760))
    irradiance = np.clip(irradiance, 0, 1200)

    # Temperature
    t_avg = profile_data.avg_temp
    t_range = profile_data.temp_range[1] - profile_data.temp_range[0]

    temperature = t_avg + (t_range / 4) * seasonal + 5 * (daily_pattern - 0.3)
    temperature = np.clip(temperature, profile_data.temp_range[0], profile_data.temp_range[1])

    # Cell temperature (higher than ambient when irradiance > 0)
    cell_temp = temperature + 0.03 * irradiance  # NOCT-based approximation

    return irradiance.astype(np.float64), cell_temp.astype(np.float64)


# =============================================================================
# Reports Page
# =============================================================================

def render_reports():
    """Render the reports and export page."""
    st.title("📄 Reports & Export")

    tab1, tab2 = st.tabs(["Generate Report", "Export Data"])

    # Tab 1: Generate Report
    with tab1:
        st.subheader("Module Performance Report")

        st.markdown(f"""
        ### {st.session_state.module_name}
        **Manufacturer:** {st.session_state.manufacturer}
        **Model:** {st.session_state.model_number}
        **Technology:** {st.session_state.technology_type}

        ---

        #### STC Performance
        | Parameter | Value |
        |-----------|-------|
        | Pmax | {st.session_state.pmax_stc:.2f} W |
        | Voc | {st.session_state.voc_stc:.2f} V |
        | Isc | {st.session_state.isc_stc:.2f} A |
        | Vmp | {st.session_state.vmp_stc:.2f} V |
        | Imp | {st.session_state.imp_stc:.2f} A |
        | Efficiency | {calculate_efficiency(st.session_state.pmax_stc, 1000, st.session_state.module_area):.2f}% |

        ---

        #### Temperature Coefficients
        | Coefficient | Value |
        |-------------|-------|
        | α (Isc) | {st.session_state.alpha_isc:+.4f} %/°C |
        | β (Voc) | {st.session_state.beta_voc:+.4f} %/°C |
        | γ (Pmax) | {st.session_state.gamma_pmax:+.4f} %/°C |
        """)

        if st.session_state.cser_results:
            results = st.session_state.cser_results
            st.markdown(f"""
            ---

            #### CSER Results ({st.session_state.selected_climate})
            | Metric | Value |
            |--------|-------|
            | Annual Energy | {results['annual_energy']:.1f} kWh |
            | CSER Rating | {results['cser_rating']:.0f} kWh/kWp |
            | Capacity Factor | {results['capacity_factor']*100:.1f}% |
            | Performance Ratio | {results['performance_ratio']*100:.1f}% |
            """)

        st.markdown(f"""
        ---
        *Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        *IEC 61853 Compliant Analysis*
        """)

    # Tab 2: Export Data
    with tab2:
        st.subheader("Export Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Power Matrix (CSV)")
            df_power = pd.DataFrame(
                st.session_state.power_matrix,
                index=[f"{g}" for g in IRRADIANCES],
                columns=[f"{t}" for t in TEMPERATURES]
            )
            df_power.index.name = "Irradiance (W/m²)"

            csv_buffer = io.StringIO()
            df_power.to_csv(csv_buffer)

            st.download_button(
                label="📥 Download Power Matrix CSV",
                data=csv_buffer.getvalue(),
                file_name=f"power_matrix_{st.session_state.model_number}.csv",
                mime="text/csv"
            )

        with col2:
            st.markdown("#### Module Data (JSON)")

            module_data = {
                "module_name": st.session_state.module_name,
                "manufacturer": st.session_state.manufacturer,
                "model_number": st.session_state.model_number,
                "technology_type": st.session_state.technology_type,
                "module_area": st.session_state.module_area,
                "cells_in_series": st.session_state.cells_in_series,
                "stc_parameters": {
                    "pmax": st.session_state.pmax_stc,
                    "voc": st.session_state.voc_stc,
                    "isc": st.session_state.isc_stc,
                    "vmp": st.session_state.vmp_stc,
                    "imp": st.session_state.imp_stc,
                },
                "temperature_coefficients": {
                    "alpha_isc": st.session_state.alpha_isc,
                    "beta_voc": st.session_state.beta_voc,
                    "gamma_pmax": st.session_state.gamma_pmax,
                },
                "cser_results": st.session_state.cser_results,
                "export_timestamp": datetime.now().isoformat(),
            }

            st.download_button(
                label="📥 Download Module JSON",
                data=json.dumps(module_data, indent=2),
                file_name=f"module_data_{st.session_state.model_number}.json",
                mime="application/json"
            )

        st.markdown("---")

        # Excel export
        st.markdown("#### Complete Workbook (Excel)")

        if st.button("Generate Excel Report"):
            buffer = io.BytesIO()

            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Module info
                info_df = pd.DataFrame({
                    "Parameter": ["Module Name", "Manufacturer", "Model", "Technology",
                                  "Area (m²)", "Cells", "Pmax (W)", "Voc (V)", "Isc (A)"],
                    "Value": [
                        st.session_state.module_name,
                        st.session_state.manufacturer,
                        st.session_state.model_number,
                        st.session_state.technology_type,
                        st.session_state.module_area,
                        st.session_state.cells_in_series,
                        st.session_state.pmax_stc,
                        st.session_state.voc_stc,
                        st.session_state.isc_stc,
                    ]
                })
                info_df.to_excel(writer, sheet_name="Module Info", index=False)

                # Power matrix
                df_power.to_excel(writer, sheet_name="Power Matrix")

                # Temperature coefficients
                coeff_df = pd.DataFrame({
                    "Coefficient": ["α (Isc)", "β (Voc)", "γ (Pmax)"],
                    "Value (%/°C)": [
                        st.session_state.alpha_isc,
                        st.session_state.beta_voc,
                        st.session_state.gamma_pmax,
                    ]
                })
                coeff_df.to_excel(writer, sheet_name="Temp Coefficients", index=False)

            st.download_button(
                label="📥 Download Excel Workbook",
                data=buffer.getvalue(),
                file_name=f"pv_module_report_{st.session_state.model_number}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()

    # Route to current page
    page = st.session_state.current_page

    if page == "Dashboard":
        render_dashboard()
    elif page == "Data Input":
        render_data_input()
    elif page == "Calculations":
        render_calculations()
    elif page == "Visualizations":
        render_visualizations()
    elif page == "Climate/CSER":
        render_climate_cser()
    elif page == "Reports":
        render_reports()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
