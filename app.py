"""
PV-CSER Pro - Climate Specific Energy Rating Application

A comprehensive Streamlit application for PV module energy rating
according to IEC 61853 standards.

Author: PV-CSER Pro Team
Version: 1.0.0
"""

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import local modules
from src.data_input import ModuleDataHandler, ModuleSpecification, PowerMatrixHandler
from src.data_input.validation import DataValidator
from src.calculations import IEC61853Part1, IEC61853Part3, EnergyYieldCalculator
from src.climate import ClimateProfile, ClimateProfileManager, CSERCalculator, CSERResult
from src.visualizations import (
    create_power_surface,
    create_monthly_bar_chart,
    create_cser_comparison_chart,
    create_loss_breakdown_chart,
    create_interactive_power_matrix,
    create_climate_comparison_chart,
    create_energy_yield_chart,
)
from src.visualizations.charts import create_temperature_coefficient_chart
from src.exports import PDFReportGenerator, ExcelExporter


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PV-CSER Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ganeshgowri-ASA/pv-cser-pro',
        'Report a bug': 'https://github.com/ganeshgowri-ASA/pv-cser-pro/issues',
        'About': """
        # PV-CSER Pro
        Climate Specific Energy Rating for PV Modules
        Based on IEC 61853 Standards
        """
    }
)


# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #E2E8F0;
        margin-bottom: 0;
    }

    /* Card styling */
    .metric-card {
        background: #F7FAFC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        color: #1E3A5F;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF6B35;
    }

    /* Section headers */
    .section-header {
        color: #1E3A5F;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F7FAFC;
    }

    /* Button styling */
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #E55A2B;
    }

    /* Success message */
    .success-box {
        background-color: #C6F6D5;
        border-left: 4px solid #38A169;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Warning message */
    .warning-box {
        background-color: #FEEBC8;
        border-left: 4px solid #DD6B20;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Info box */
    .info-box {
        background-color: #E2E8F0;
        border-left: 4px solid #2C5282;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        'module_data': None,
        'power_matrix_data': None,
        'cser_results': None,
        'all_cser_results': None,
        'current_page': 'home',
        'calculation_done': False,
        'db_connected': False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def create_sample_power_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample power matrix for demonstration."""
    irradiance = np.array([100, 200, 400, 600, 800, 1000, 1100])
    temperature = np.array([15, 25, 50, 75])
    pmax_stc = 400.0
    temp_coeff = -0.35

    power_matrix = np.zeros((len(irradiance), len(temperature)))
    for i, g in enumerate(irradiance):
        for j, t in enumerate(temperature):
            p_base = pmax_stc * (g / 1000.0)
            temp_factor = 1 + (temp_coeff / 100) * (t - 25)
            power_matrix[i, j] = p_base * temp_factor
            if g < 400:
                power_matrix[i, j] *= 0.98

    return irradiance, temperature, power_matrix


def display_metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Display a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        <small>{subtitle}</small>
    </div>
    """, unsafe_allow_html=True)


def get_db_connection():
    """Get database connection if configured."""
    try:
        from src.utils.database import DatabaseManager
        db_url = os.getenv('DATABASE_URL') or st.secrets.get('DATABASE_URL', None)
        if db_url:
            db = DatabaseManager(db_url)
            if db.check_connection():
                return db
    except Exception:
        pass
    return None


# =============================================================================
# Page Components
# =============================================================================

def render_header() -> None:
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>☀️ PV-CSER Pro</h1>
        <p>Climate Specific Energy Rating for PV Modules | IEC 61853 Standards</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/solar-panel.png", width=80)
        st.title("Navigation")

        pages = {
            "🏠 Home": "home",
            "📊 Module Data": "module_data",
            "📈 Power Matrix": "power_matrix",
            "🌍 Climate Profiles": "climate",
            "⚡ CSER Calculation": "cser",
            "📉 Visualizations": "visualizations",
            "📁 Export Reports": "export",
            "⚙️ Settings": "settings",
        }

        selected = st.radio(
            "Select Page",
            options=list(pages.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats
        if st.session_state.module_data:
            st.subheader("Current Module")
            module = st.session_state.module_data
            st.write(f"**{module.get('manufacturer', 'N/A')}**")
            st.write(f"{module.get('model_name', 'N/A')}")
            st.write(f"Pmax: {module.get('pmax_stc', 0):.0f} W")

        if st.session_state.cser_results:
            st.subheader("Latest CSER")
            cser = st.session_state.cser_results
            st.metric("CSER", f"{cser.cser:.0f} kWh/kWp")

        st.divider()

        # Database status
        db = get_db_connection()
        if db:
            st.success("✓ Database Connected")
        else:
            st.info("ℹ️ Local Mode (No DB)")

        st.markdown("---")
        st.markdown("**PV-CSER Pro v1.0.0**")
        st.markdown("IEC 61853 Compliant")

    return pages[selected]


def render_home_page() -> None:
    """Render home page."""
    st.markdown("## Welcome to PV-CSER Pro")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **PV-CSER Pro** is a comprehensive application for calculating Climate Specific
        Energy Ratings (CSER) for photovoltaic modules according to **IEC 61853** standards.

        ### Key Features

        - **IEC 61853 Compliance**: Full implementation of Parts 1-4
        - **Power Matrix Analysis**: Upload and analyze module performance data
        - **Climate Profiles**: Standard IEC profiles and custom climate data
        - **Interactive Visualizations**: 3D plots and comprehensive charts
        - **Export Reports**: Generate PDF and Excel reports

        ### Getting Started

        1. **Enter Module Data**: Input your PV module specifications
        2. **Upload Power Matrix**: Provide irradiance vs temperature performance data
        3. **Select Climate Profile**: Choose from IEC standard profiles or upload custom data
        4. **Calculate CSER**: Run the energy rating calculations
        5. **Export Results**: Download comprehensive reports

        Use the sidebar navigation to get started!
        """)

    with col2:
        st.markdown("### Quick Actions")

        if st.button("🆕 New Analysis", use_container_width=True):
            for key in ['module_data', 'power_matrix_data', 'cser_results']:
                st.session_state[key] = None
            st.rerun()

        if st.button("📋 Load Sample Data", use_container_width=True):
            load_sample_data()
            st.success("Sample data loaded!")
            st.rerun()

        st.markdown("### IEC 61853 Standards")
        st.info("""
        - **Part 1**: Power rating at STC
        - **Part 2**: Spectral responsivity
        - **Part 3**: Energy rating methodology
        - **Part 4**: Climate profiles
        """)


def load_sample_data() -> None:
    """Load sample data for demonstration."""
    # Sample module
    st.session_state.module_data = {
        'manufacturer': 'Sample Solar',
        'model_name': 'PV-400M',
        'pmax_stc': 400.0,
        'voc_stc': 48.5,
        'isc_stc': 10.5,
        'vmp_stc': 40.8,
        'imp_stc': 9.8,
        'temp_coeff_pmax': -0.35,
        'temp_coeff_voc': -0.28,
        'temp_coeff_isc': 0.05,
        'module_area': 1.92,
        'cell_type': 'mono-Si',
        'num_cells': 72,
        'nmot': 43.0,
    }

    # Sample power matrix
    irr, temp, power = create_sample_power_matrix()
    st.session_state.power_matrix_data = {
        'irradiance': irr,
        'temperature': temp,
        'power_matrix': power,
    }


def render_module_data_page() -> None:
    """Render module data input page."""
    st.markdown("## 📊 Module Specifications")

    st.markdown("""
    Enter your PV module specifications below. These values are typically found
    in the module datasheet and are used for IEC 61853 calculations.
    """)

    # Initialize with existing data or defaults
    existing = st.session_state.module_data or {}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Identification")
        manufacturer = st.text_input(
            "Manufacturer",
            value=existing.get('manufacturer', ''),
            help="Module manufacturer name"
        )
        model_name = st.text_input(
            "Model Name",
            value=existing.get('model_name', ''),
            help="Module model designation"
        )
        cell_type = st.selectbox(
            "Cell Type",
            options=['mono-Si', 'poly-Si', 'PERC', 'HJT', 'TOPCon', 'CdTe', 'CIGS'],
            index=['mono-Si', 'poly-Si', 'PERC', 'HJT', 'TOPCon', 'CdTe', 'CIGS'].index(
                existing.get('cell_type', 'mono-Si')
            ),
        )

        st.subheader("STC Specifications")
        pmax_stc = st.number_input(
            "Pmax at STC (W)",
            min_value=0.0,
            max_value=1000.0,
            value=float(existing.get('pmax_stc', 400.0)),
            step=1.0,
            help="Maximum power at Standard Test Conditions"
        )
        voc_stc = st.number_input(
            "Voc at STC (V)",
            min_value=0.0,
            max_value=100.0,
            value=float(existing.get('voc_stc', 48.0)),
            step=0.1,
        )
        isc_stc = st.number_input(
            "Isc at STC (A)",
            min_value=0.0,
            max_value=30.0,
            value=float(existing.get('isc_stc', 10.0)),
            step=0.1,
        )
        vmp_stc = st.number_input(
            "Vmp at STC (V)",
            min_value=0.0,
            max_value=100.0,
            value=float(existing.get('vmp_stc', 40.0)),
            step=0.1,
        )
        imp_stc = st.number_input(
            "Imp at STC (A)",
            min_value=0.0,
            max_value=30.0,
            value=float(existing.get('imp_stc', 10.0)),
            step=0.1,
        )

    with col2:
        st.subheader("Temperature Coefficients")
        temp_coeff_pmax = st.number_input(
            "Temperature Coeff. Pmax (%/°C)",
            min_value=-1.0,
            max_value=0.0,
            value=float(existing.get('temp_coeff_pmax', -0.35)),
            step=0.01,
            format="%.2f",
            help="Negative value indicating power decrease with temperature"
        )
        temp_coeff_voc = st.number_input(
            "Temperature Coeff. Voc (%/°C)",
            min_value=-1.0,
            max_value=0.0,
            value=float(existing.get('temp_coeff_voc', -0.30)),
            step=0.01,
            format="%.2f",
        )
        temp_coeff_isc = st.number_input(
            "Temperature Coeff. Isc (%/°C)",
            min_value=-0.1,
            max_value=0.2,
            value=float(existing.get('temp_coeff_isc', 0.05)),
            step=0.01,
            format="%.2f",
        )

        st.subheader("Physical Specifications")
        module_area = st.number_input(
            "Module Area (m²)",
            min_value=0.1,
            max_value=5.0,
            value=float(existing.get('module_area', 1.92)),
            step=0.01,
        )
        num_cells = st.number_input(
            "Number of Cells",
            min_value=1,
            max_value=200,
            value=int(existing.get('num_cells', 72)),
            step=1,
        )
        nmot = st.number_input(
            "NMOT (°C)",
            min_value=30.0,
            max_value=60.0,
            value=float(existing.get('nmot', 45.0)),
            step=0.1,
            help="Nominal Module Operating Temperature"
        )

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Module Data", type="primary", use_container_width=True):
            module_data = {
                'manufacturer': manufacturer,
                'model_name': model_name,
                'cell_type': cell_type,
                'pmax_stc': pmax_stc,
                'voc_stc': voc_stc,
                'isc_stc': isc_stc,
                'vmp_stc': vmp_stc,
                'imp_stc': imp_stc,
                'temp_coeff_pmax': temp_coeff_pmax,
                'temp_coeff_voc': temp_coeff_voc,
                'temp_coeff_isc': temp_coeff_isc,
                'module_area': module_area,
                'num_cells': num_cells,
                'nmot': nmot,
            }

            # Validate
            validator = DataValidator()
            result = validator.validate_module_specs(module_data)

            if result.is_valid:
                st.session_state.module_data = module_data
                st.success("✅ Module data saved successfully!")
            else:
                for error in result.errors:
                    st.error(f"❌ {error}")
                for warning in result.warnings:
                    st.warning(f"⚠️ {warning}")

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.module_data = None
            st.rerun()

    # Show calculated values
    if pmax_stc > 0 and voc_stc > 0 and isc_stc > 0:
        st.divider()
        st.subheader("Calculated Values")

        calc_col1, calc_col2, calc_col3 = st.columns(3)

        ff = (pmax_stc / (voc_stc * isc_stc)) * 100 if (voc_stc * isc_stc) > 0 else 0
        efficiency = (pmax_stc / (module_area * 1000)) * 100 if module_area > 0 else 0

        with calc_col1:
            st.metric("Fill Factor", f"{ff:.1f}%")
        with calc_col2:
            st.metric("Efficiency", f"{efficiency:.1f}%")
        with calc_col3:
            st.metric("Power Density", f"{pmax_stc/module_area:.1f} W/m²")


def render_power_matrix_page() -> None:
    """Render power matrix input page."""
    st.markdown("## 📈 Power Matrix Data")

    st.markdown("""
    Upload your IEC 61853-1 power matrix data or use sample data.
    The power matrix shows module power at various irradiance and temperature combinations.
    """)

    tab1, tab2 = st.tabs(["📤 Upload Data", "📝 Manual Entry"])

    with tab1:
        st.markdown("### Upload Power Matrix File")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="File should have irradiance values in first column and temperature in header row"
        )

        if uploaded_file:
            try:
                handler = PowerMatrixHandler()

                if uploaded_file.name.endswith('.csv'):
                    matrix_data = handler.load_from_csv(uploaded_file.getvalue())
                else:
                    matrix_data = handler.load_from_excel(uploaded_file.getvalue())

                st.success(f"✅ Loaded matrix: {len(matrix_data.irradiance_levels)} irradiance x {len(matrix_data.temperature_levels)} temperature points")

                # Show preview
                st.markdown("#### Data Preview")
                df = matrix_data.to_dataframe()
                st.dataframe(df.style.format("{:.1f}"))

                # Validate
                pmax_stc = st.session_state.module_data.get('pmax_stc') if st.session_state.module_data else None
                is_valid, issues = handler.validate_matrix(matrix_data, pmax_stc)

                if issues:
                    for issue in issues:
                        if issue.startswith("ERROR"):
                            st.error(issue)
                        else:
                            st.warning(issue)

                if st.button("💾 Save Power Matrix", type="primary"):
                    st.session_state.power_matrix_data = {
                        'irradiance': matrix_data.irradiance_levels,
                        'temperature': matrix_data.temperature_levels,
                        'power_matrix': matrix_data.power_matrix,
                    }
                    st.success("Power matrix saved!")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        st.markdown("---")
        st.markdown("#### Use Sample Data")

        if st.button("📋 Load Sample Power Matrix"):
            irr, temp, power = create_sample_power_matrix()
            st.session_state.power_matrix_data = {
                'irradiance': irr,
                'temperature': temp,
                'power_matrix': power,
            }
            st.success("Sample power matrix loaded!")
            st.rerun()

    with tab2:
        st.markdown("### Manual Data Entry")

        st.info("Enter comma-separated values for irradiance and temperature levels.")

        irr_input = st.text_input(
            "Irradiance levels (W/m²)",
            value="100, 200, 400, 600, 800, 1000, 1100",
        )
        temp_input = st.text_input(
            "Temperature levels (°C)",
            value="15, 25, 50, 75",
        )

        try:
            irr_vals = [float(x.strip()) for x in irr_input.split(',')]
            temp_vals = [float(x.strip()) for x in temp_input.split(',')]

            st.markdown("#### Enter Power Values (W)")

            # Create input grid
            power_data = []
            cols = st.columns(len(temp_vals) + 1)

            with cols[0]:
                st.write("**G \\ T**")
            for j, t in enumerate(temp_vals):
                with cols[j + 1]:
                    st.write(f"**{t}°C**")

            for i, g in enumerate(irr_vals):
                cols = st.columns(len(temp_vals) + 1)
                row = []
                with cols[0]:
                    st.write(f"**{g}**")
                for j, t in enumerate(temp_vals):
                    with cols[j + 1]:
                        val = st.number_input(
                            f"P({g},{t})",
                            key=f"p_{i}_{j}",
                            min_value=0.0,
                            label_visibility="collapsed",
                        )
                        row.append(val)
                power_data.append(row)

            if st.button("💾 Save Manual Matrix", type="primary"):
                st.session_state.power_matrix_data = {
                    'irradiance': np.array(irr_vals),
                    'temperature': np.array(temp_vals),
                    'power_matrix': np.array(power_data),
                }
                st.success("Power matrix saved!")

        except Exception as e:
            st.error(f"Error parsing input: {str(e)}")

    # Show current power matrix if available
    if st.session_state.power_matrix_data:
        st.divider()
        st.markdown("### Current Power Matrix")

        data = st.session_state.power_matrix_data
        fig = create_power_surface(
            data['irradiance'],
            data['temperature'],
            data['power_matrix'],
            title="Power Matrix - P(G, T)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_climate_page() -> None:
    """Render climate profiles page."""
    st.markdown("## 🌍 Climate Profiles")

    st.markdown("""
    Select from IEC 61853-4 standard climate profiles or upload custom climate data.
    Climate profiles define the irradiance and temperature conditions for CSER calculation.
    """)

    tab1, tab2 = st.tabs(["📋 Standard Profiles", "📤 Custom Profile"])

    profile_manager = ClimateProfileManager()

    with tab1:
        st.markdown("### IEC 61853-4 Standard Climate Profiles")

        profiles = profile_manager.list_profiles()

        # Display as cards
        cols = st.columns(3)

        for i, profile in enumerate(profiles):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                        <h4 style="color: #1E3A5F;">{profile['name']}</h4>
                        <p><strong>Annual GHI:</strong> {profile['annual_ghi']:.0f} kWh/m²</p>
                        <p><strong>Avg Temp:</strong> {profile['avg_temp']:.1f}°C</p>
                        <p style="font-size: 0.85rem; color: #666;">{profile['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # Profile comparison chart
        st.markdown("### Profile Comparison")

        comparison_data = {}
        for name in profile_manager.get_standard_profiles():
            profile = profile_manager.get_profile(name)
            if profile:
                comparison_data[name] = {
                    'annual_ghi': profile.annual_ghi,
                    'avg_temp': profile.avg_temp,
                }

        fig_ghi = go.Figure()
        fig_ghi.add_trace(go.Bar(
            x=list(comparison_data.keys()),
            y=[d['annual_ghi'] for d in comparison_data.values()],
            marker_color='#FF6B35',
        ))
        fig_ghi.update_layout(
            title="Annual GHI by Climate",
            xaxis_title="Climate Profile",
            yaxis_title="Annual GHI (kWh/m²)",
            height=400,
        )
        st.plotly_chart(fig_ghi, use_container_width=True)

    with tab2:
        st.markdown("### Upload Custom Climate Data")

        st.info("""
        Upload hourly climate data with columns:
        - `ghi`: Global Horizontal Irradiance (W/m²)
        - `temperature`: Ambient Temperature (°C)
        - `wind_speed` (optional): Wind Speed (m/s)

        Data should contain 8760 rows (one year of hourly data).
        """)

        uploaded_climate = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            key="climate_upload",
        )

        if uploaded_climate:
            try:
                df = pd.read_csv(uploaded_climate)
                st.write(f"Loaded {len(df)} rows")

                required_cols = ['ghi', 'temperature']
                missing = [c for c in required_cols if c not in df.columns]

                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.success("Data validated!")

                    profile_name = st.text_input("Profile Name", value="Custom Profile")
                    location = st.text_input("Location", value="")

                    col1, col2 = st.columns(2)
                    with col1:
                        latitude = st.number_input("Latitude", value=0.0, min_value=-90.0, max_value=90.0)
                    with col2:
                        longitude = st.number_input("Longitude", value=0.0, min_value=-180.0, max_value=180.0)

                    if st.button("Create Custom Profile", type="primary"):
                        ghi = df['ghi'].values
                        temp = df['temperature'].values
                        wind = df.get('wind_speed', np.ones(len(ghi))).values

                        if len(ghi) != 8760:
                            st.warning(f"Data has {len(ghi)} hours, expected 8760. Will be padded/truncated.")
                            if len(ghi) < 8760:
                                ghi = np.pad(ghi, (0, 8760 - len(ghi)))
                                temp = np.pad(temp, (0, 8760 - len(temp)))
                                wind = np.pad(wind, (0, 8760 - len(wind)))
                            else:
                                ghi = ghi[:8760]
                                temp = temp[:8760]
                                wind = wind[:8760]

                        profile_manager.add_custom_profile(
                            name=profile_name,
                            ghi=ghi,
                            ambient_temp=temp,
                            wind_speed=wind,
                            location=location,
                            latitude=latitude,
                            longitude=longitude,
                        )
                        st.success(f"Created custom profile: {profile_name}")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def render_cser_page() -> None:
    """Render CSER calculation page."""
    st.markdown("## ⚡ CSER Calculation")

    # Check prerequisites
    if not st.session_state.module_data:
        st.warning("⚠️ Please enter module data first (Module Data page)")
        return

    if not st.session_state.power_matrix_data:
        st.warning("⚠️ Please upload power matrix data first (Power Matrix page)")
        return

    st.success("✅ Module data and power matrix loaded. Ready for CSER calculation.")

    module = st.session_state.module_data
    power_data = st.session_state.power_matrix_data

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Calculation Settings")

        profile_manager = ClimateProfileManager()
        profile_names = profile_manager.get_standard_profiles()

        selected_profiles = st.multiselect(
            "Select Climate Profiles",
            options=profile_names,
            default=profile_names[:3],
            help="Select one or more climate profiles for CSER calculation"
        )

        use_custom_nmot = st.checkbox("Use custom NMOT", value=False)
        if use_custom_nmot:
            nmot = st.slider("NMOT (°C)", 30.0, 60.0, float(module.get('nmot', 45)), 0.5)
        else:
            nmot = module.get('nmot', 45)

    with col2:
        st.markdown("### Module Summary")
        st.write(f"**{module.get('manufacturer')} {module.get('model_name')}**")
        st.write(f"Pmax: {module.get('pmax_stc'):.0f} W")
        st.write(f"Temp Coeff: {module.get('temp_coeff_pmax'):.2f} %/°C")
        st.write(f"NMOT: {nmot:.1f}°C")

    st.divider()

    if st.button("🚀 Calculate CSER", type="primary", use_container_width=True):
        if not selected_profiles:
            st.error("Please select at least one climate profile")
            return

        with st.spinner("Calculating CSER for all selected profiles..."):
            # Create power model from matrix
            def power_model(g: float, t: float) -> float:
                from scipy import interpolate
                interp_func = interpolate.RegularGridInterpolator(
                    (power_data['irradiance'], power_data['temperature']),
                    power_data['power_matrix'],
                    method='linear',
                    bounds_error=False,
                    fill_value=0,
                )
                return float(interp_func((g, t)))

            # Calculate CSER for each profile
            all_results = {}
            progress_bar = st.progress(0)

            for i, profile_name in enumerate(selected_profiles):
                profile = profile_manager.get_profile(profile_name)
                if profile:
                    calculator = CSERCalculator(
                        power_model=power_model,
                        pmax_stc=module.get('pmax_stc', 400),
                        temp_coeff_pmax=module.get('temp_coeff_pmax', -0.35),
                        nmot=nmot,
                    )
                    result = calculator.calculate_cser(profile)
                    all_results[profile_name] = result

                progress_bar.progress((i + 1) / len(selected_profiles))

            st.session_state.all_cser_results = all_results

            if all_results:
                # Store first result as main result
                first_profile = list(all_results.keys())[0]
                st.session_state.cser_results = all_results[first_profile]

        st.success(f"✅ CSER calculated for {len(all_results)} climate profiles!")

        # Display results
        st.markdown("### Results Summary")

        results_df = pd.DataFrame([
            {
                'Climate Profile': name,
                'CSER (kWh/kWp)': result.cser,
                'Annual Energy (kWh)': result.annual_energy,
                'Performance Ratio (%)': result.performance_ratio,
                'Temperature Loss (%)': result.temperature_loss,
            }
            for name, result in all_results.items()
        ])

        results_df = results_df.sort_values('CSER (kWh/kWp)', ascending=False)
        st.dataframe(results_df.style.format({
            'CSER (kWh/kWp)': '{:.0f}',
            'Annual Energy (kWh)': '{:.1f}',
            'Performance Ratio (%)': '{:.1f}',
            'Temperature Loss (%)': '{:.1f}',
        }), use_container_width=True)

        # Comparison chart
        cser_values = {name: result.cser for name, result in all_results.items()}
        fig = create_cser_comparison_chart(cser_values)
        st.plotly_chart(fig, use_container_width=True)


def render_visualizations_page() -> None:
    """Render visualizations page."""
    st.markdown("## 📉 Visualizations")

    if not st.session_state.cser_results and not st.session_state.power_matrix_data:
        st.warning("⚠️ Please complete CSER calculation first")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔳 Power Matrix",
        "📊 Energy Yield",
        "🌡️ Temperature Effects",
        "📉 Loss Analysis"
    ])

    with tab1:
        if st.session_state.power_matrix_data:
            data = st.session_state.power_matrix_data

            view_type = st.radio(
                "View Type",
                options=["3D Surface", "Contour", "Heatmap"],
                horizontal=True,
            )

            if view_type == "3D Surface":
                fig = create_power_surface(
                    data['irradiance'],
                    data['temperature'],
                    data['power_matrix'],
                )
            elif view_type == "Contour":
                from src.visualizations.plots_3d import PowerMatrix3DPlot
                plotter = PowerMatrix3DPlot(
                    data['irradiance'],
                    data['temperature'],
                    data['power_matrix'],
                )
                fig = plotter.create_contour_plot()
            else:
                from src.visualizations.plots_3d import PowerMatrix3DPlot
                plotter = PowerMatrix3DPlot(
                    data['irradiance'],
                    data['temperature'],
                    data['power_matrix'],
                )
                fig = plotter.create_heatmap()

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.session_state.cser_results:
            result = st.session_state.cser_results

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CSER", f"{result.cser:.0f} kWh/kWp")
            with col2:
                st.metric("Annual Energy", f"{result.annual_energy:.1f} kWh")
            with col3:
                st.metric("Performance Ratio", f"{result.performance_ratio:.1f}%")

            # Monthly chart
            fig = create_monthly_bar_chart(
                result.monthly_energy,
                title="Monthly Energy Yield",
                y_label="Energy (kWh)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Monthly CSER
            fig2 = create_monthly_bar_chart(
                result.monthly_cser,
                title="Monthly Specific Yield",
                y_label="Yield (kWh/kWp)",
                color="#2C5282"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if st.session_state.power_matrix_data and st.session_state.module_data:
            data = st.session_state.power_matrix_data
            module = st.session_state.module_data

            # Temperature coefficient visualization
            idx_1000 = np.argmin(np.abs(data['irradiance'] - 1000))
            power_vs_temp = data['power_matrix'][idx_1000, :]

            fig = create_temperature_coefficient_chart(
                data['temperature'],
                power_vs_temp,
                module.get('pmax_stc', 400),
                module.get('temp_coeff_pmax', -0.35),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if st.session_state.cser_results:
            result = st.session_state.cser_results

            losses = {
                'Temperature': result.temperature_loss,
                'Low Irradiance': result.low_irradiance_loss,
                'Other': max(0, (100 - result.performance_ratio) - result.temperature_loss - result.low_irradiance_loss),
            }

            fig = create_loss_breakdown_chart(losses, title="Energy Loss Breakdown")
            st.plotly_chart(fig, use_container_width=True)


def render_export_page() -> None:
    """Render export/reports page."""
    st.markdown("## 📁 Export Reports")

    if not st.session_state.cser_results:
        st.warning("⚠️ Please complete CSER calculation first to generate reports")
        return

    module = st.session_state.module_data or {}
    cser_result = st.session_state.cser_results
    power_data = st.session_state.power_matrix_data

    st.markdown("### Available Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 PDF Report")
        st.write("Generate a comprehensive PDF report with all analysis results.")

        if st.button("📥 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_gen = PDFReportGenerator()
                pdf_buffer = pdf_gen.generate_report(
                    module_data=module,
                    cser_results=cser_result.to_dict(),
                    power_matrix_data=power_data,
                )

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"CSER_Report_{module.get('model_name', 'Module')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with col2:
        st.markdown("#### 📊 Excel Report")
        st.write("Export all data and results to Excel workbook.")

        if st.button("📥 Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel..."):
                excel_exp = ExcelExporter()

                # Prepare comparison data if available
                comparison = None
                if st.session_state.all_cser_results:
                    comparison = {
                        name: result.to_dict()
                        for name, result in st.session_state.all_cser_results.items()
                    }

                excel_buffer = excel_exp.export_complete_report(
                    module_data=module,
                    power_matrix_data=power_data,
                    cser_results=cser_result.to_dict(),
                    climate_comparison=comparison,
                )

                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_buffer,
                    file_name=f"CSER_Data_{module.get('model_name', 'Module')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    st.divider()

    st.markdown("### Quick Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Export Power Matrix Only", use_container_width=True):
            if power_data:
                excel_exp = ExcelExporter()
                buffer = excel_exp.export_power_matrix(
                    power_data['irradiance'],
                    power_data['temperature'],
                    power_data['power_matrix'],
                )
                st.download_button(
                    "⬇️ Download",
                    data=buffer,
                    file_name="power_matrix.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with col2:
        if st.button("📈 Export Monthly Data Only", use_container_width=True):
            if cser_result:
                excel_exp = ExcelExporter()
                buffer = excel_exp.export_monthly_data(
                    cser_result.monthly_energy,
                    [0] * 12,  # Placeholder for irradiation
                    module.get('pmax_stc', 400),
                )
                st.download_button(
                    "⬇️ Download",
                    data=buffer,
                    file_name="monthly_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with col3:
        if st.button("📄 Quick PDF Summary", use_container_width=True):
            if cser_result and module:
                pdf_gen = PDFReportGenerator()
                buffer = pdf_gen.generate_quick_summary(
                    f"{module.get('manufacturer', '')} {module.get('model_name', '')}",
                    cser_result.cser,
                    cser_result.climate_profile,
                )
                st.download_button(
                    "⬇️ Download",
                    data=buffer,
                    file_name="cser_summary.pdf",
                    mime="application/pdf",
                )


def render_settings_page() -> None:
    """Render settings page."""
    st.markdown("## ⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["🗄️ Database", "🎨 Appearance", "ℹ️ About"])

    with tab1:
        st.markdown("### Database Configuration")

        db = get_db_connection()

        if db:
            st.success("✅ Database connected successfully")

            if st.button("Test Connection"):
                if db.check_connection():
                    st.success("Connection verified!")
                else:
                    st.error("Connection failed")
        else:
            st.info("""
            Database connection is optional. To enable PostgreSQL storage:

            1. Set the `DATABASE_URL` environment variable, or
            2. Add it to `.streamlit/secrets.toml`:

            ```toml
            DATABASE_URL = "postgresql://user:password@host:port/database"
            ```

            The app will work in local mode without a database.
            """)

    with tab2:
        st.markdown("### Appearance Settings")

        st.info("Appearance settings will be available in a future update.")

        st.selectbox(
            "Color Theme",
            options=["Default (Solar)", "Ocean Blue", "Forest Green"],
            disabled=True,
        )

        st.selectbox(
            "Chart Style",
            options=["Plotly Default", "Minimal", "Dark"],
            disabled=True,
        )

    with tab3:
        st.markdown("### About PV-CSER Pro")

        st.markdown("""
        **PV-CSER Pro** is a comprehensive application for Climate Specific Energy
        Rating (CSER) calculations for photovoltaic modules.

        #### Version
        - **Version**: 1.0.0
        - **Release Date**: 2024

        #### Standards Compliance
        - IEC 61853-1: Power rating at STC
        - IEC 61853-2: Spectral responsivity
        - IEC 61853-3: Energy rating methodology
        - IEC 61853-4: Climate profiles

        #### Technology Stack
        - **Framework**: Streamlit
        - **Visualization**: Plotly
        - **Database**: PostgreSQL (optional)
        - **Export**: PDF (ReportLab), Excel (openpyxl)

        #### Links
        - [GitHub Repository](https://github.com/ganeshgowri-ASA/pv-cser-pro)
        - [Documentation](https://github.com/ganeshgowri-ASA/pv-cser-pro#readme)
        - [Report Issues](https://github.com/ganeshgowri-ASA/pv-cser-pro/issues)

        ---
        **License**: MIT License

        Made with ❤️ for the PV community
        """)


def render_footer() -> None:
    """Render footer."""
    st.markdown("""
    <div class="footer">
        <p>PV-CSER Pro | IEC 61853 Compliant | v1.0.0</p>
        <p>Made with ❤️ for the PV community</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main() -> None:
    """Main application entry point."""
    render_header()

    current_page = render_sidebar()

    # Route to appropriate page
    if current_page == "home":
        render_home_page()
    elif current_page == "module_data":
        render_module_data_page()
    elif current_page == "power_matrix":
        render_power_matrix_page()
    elif current_page == "climate":
        render_climate_page()
    elif current_page == "cser":
        render_cser_page()
    elif current_page == "visualizations":
        render_visualizations_page()
    elif current_page == "export":
        render_export_page()
    elif current_page == "settings":
        render_settings_page()

    render_footer()


if __name__ == "__main__":
    main()
