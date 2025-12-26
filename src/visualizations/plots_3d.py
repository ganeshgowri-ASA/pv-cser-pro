"""
3D Visualization Module for PV-CSER Pro.

This module provides Plotly-based 3D surface plots for visualizing
PV module performance data according to IEC 61853 standards.

Includes:
- Power surface P(G, T)
- Efficiency surface η(G, T)
- Temperature coefficient visualization
- Climate profile comparisons

Reference:
    IEC 61853-1:2011 - Irradiance and temperature performance measurements
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.typing import NDArray

from src.utils.constants import (
    IRRADIANCES,
    TEMPERATURES,
    CLIMATE_PROFILES,
    ClimateProfile,
    UNITS,
)


# =============================================================================
# Color Schemes
# =============================================================================

COLORSCALES = {
    'power': 'Viridis',
    'efficiency': 'RdYlGn',
    'temperature': 'RdBu_r',
    'irradiance': 'YlOrRd',
    'energy': 'Blues',
}

PLOT_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font_color': '#262730',
    'gridcolor': 'rgba(128,128,128,0.2)',
}


# =============================================================================
# 3D Surface Plots
# =============================================================================

def create_power_surface_plot(
    power_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    module_name: str = "PV Module",
    show_contours: bool = True,
    height: int = 600
) -> go.Figure:
    """
    Create 3D surface plot of Power vs Irradiance and Temperature.

    Visualizes P(G, T) surface from IEC 61853-1 power matrix data.

    Args:
        power_matrix: Pmax values array of shape (n_irradiances, n_temperatures)
        irradiances: List of irradiance levels in W/m²
        temperatures: List of temperature levels in °C
        module_name: Module name for title
        show_contours: Whether to show contour projections
        height: Plot height in pixels

    Returns:
        Plotly Figure object

    Reference:
        IEC 61853-1:2011 Clause 8
    """
    # Create meshgrid for surface
    g_arr = np.array(irradiances)
    t_arr = np.array(temperatures)
    T, G = np.meshgrid(t_arr, g_arr)

    # Create figure
    fig = go.Figure()

    # Add surface
    surface = go.Surface(
        x=T,
        y=G,
        z=power_matrix,
        colorscale=COLORSCALES['power'],
        colorbar=dict(
            title=dict(text=f"Power ({UNITS['power']})", side='right'),
            thickness=20,
            len=0.75,
        ),
        contours=dict(
            z=dict(show=show_contours, usecolormap=True, highlightcolor="white", project_z=True)
        ) if show_contours else None,
        hovertemplate=(
            "Temperature: %{x}°C<br>"
            "Irradiance: %{y} W/m²<br>"
            "Power: %{z:.2f} W<br>"
            "<extra></extra>"
        ),
        name="P(G,T)"
    )
    fig.add_trace(surface)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Power Surface P(G,T) - {module_name}",
            font=dict(size=18),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Temperature (°C)",
                gridcolor=PLOT_THEME['gridcolor'],
                showbackground=True,
                backgroundcolor='rgba(230,230,230,0.3)',
            ),
            yaxis=dict(
                title="Irradiance (W/m²)",
                gridcolor=PLOT_THEME['gridcolor'],
                showbackground=True,
                backgroundcolor='rgba(230,230,230,0.3)',
            ),
            zaxis=dict(
                title="Power (W)",
                gridcolor=PLOT_THEME['gridcolor'],
                showbackground=True,
                backgroundcolor='rgba(230,230,230,0.3)',
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
    )

    return fig


def create_efficiency_surface_plot(
    power_matrix: NDArray[np.float64],
    module_area: float,
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    module_name: str = "PV Module",
    height: int = 600
) -> go.Figure:
    """
    Create 3D surface plot of Efficiency vs Irradiance and Temperature.

    η(G, T) = P(G, T) / (G × A) × 100

    Args:
        power_matrix: Pmax values array of shape (n_irradiances, n_temperatures)
        module_area: Module area in m²
        irradiances: List of irradiance levels in W/m²
        temperatures: List of temperature levels in °C
        module_name: Module name for title
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    g_arr = np.array(irradiances)
    t_arr = np.array(temperatures)
    T, G = np.meshgrid(t_arr, g_arr)

    # Calculate efficiency matrix
    efficiency_matrix = np.zeros_like(power_matrix)
    for i, g in enumerate(irradiances):
        for j in range(len(temperatures)):
            if g > 0 and not np.isnan(power_matrix[i, j]):
                efficiency_matrix[i, j] = (power_matrix[i, j] / (g * module_area)) * 100
            else:
                efficiency_matrix[i, j] = np.nan

    fig = go.Figure()

    surface = go.Surface(
        x=T,
        y=G,
        z=efficiency_matrix,
        colorscale=COLORSCALES['efficiency'],
        colorbar=dict(
            title=dict(text=f"Efficiency ({UNITS['efficiency']})", side='right'),
            thickness=20,
            len=0.75,
        ),
        hovertemplate=(
            "Temperature: %{x}°C<br>"
            "Irradiance: %{y} W/m²<br>"
            "Efficiency: %{z:.2f}%<br>"
            "<extra></extra>"
        ),
        name="η(G,T)"
    )
    fig.add_trace(surface)

    fig.update_layout(
        title=dict(
            text=f"Efficiency Surface η(G,T) - {module_name}",
            font=dict(size=18),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Temperature (°C)", gridcolor=PLOT_THEME['gridcolor']),
            yaxis=dict(title="Irradiance (W/m²)", gridcolor=PLOT_THEME['gridcolor']),
            zaxis=dict(title="Efficiency (%)", gridcolor=PLOT_THEME['gridcolor']),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
    )

    return fig


def create_dual_surface_plot(
    power_matrix: NDArray[np.float64],
    module_area: float,
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    module_name: str = "PV Module",
    height: int = 500
) -> go.Figure:
    """
    Create side-by-side 3D plots of Power and Efficiency surfaces.

    Args:
        power_matrix: Pmax values array
        module_area: Module area in m²
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        module_name: Module name for title
        height: Plot height in pixels

    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=['Power P(G,T)', 'Efficiency η(G,T)'],
        horizontal_spacing=0.05,
    )

    g_arr = np.array(irradiances)
    t_arr = np.array(temperatures)
    T, G = np.meshgrid(t_arr, g_arr)

    # Power surface
    fig.add_trace(
        go.Surface(
            x=T, y=G, z=power_matrix,
            colorscale=COLORSCALES['power'],
            showscale=True,
            colorbar=dict(x=0.45, len=0.75, thickness=15),
        ),
        row=1, col=1
    )

    # Efficiency surface
    efficiency_matrix = np.zeros_like(power_matrix)
    for i, g in enumerate(irradiances):
        for j in range(len(temperatures)):
            if g > 0 and not np.isnan(power_matrix[i, j]):
                efficiency_matrix[i, j] = (power_matrix[i, j] / (g * module_area)) * 100

    fig.add_trace(
        go.Surface(
            x=T, y=G, z=efficiency_matrix,
            colorscale=COLORSCALES['efficiency'],
            showscale=True,
            colorbar=dict(x=1.0, len=0.75, thickness=15),
        ),
        row=1, col=2
    )

    # Update layout
    scene_config = dict(
        xaxis=dict(title="T (°C)"),
        yaxis=dict(title="G (W/m²)"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    )

    fig.update_layout(
        title=dict(text=f"PV Module Characteristics - {module_name}", x=0.5),
        scene=dict(**scene_config, zaxis=dict(title="P (W)")),
        scene2=dict(**scene_config, zaxis=dict(title="η (%)")),
        height=height,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return fig


# =============================================================================
# 2D Plots
# =============================================================================

def create_power_vs_irradiance_plot(
    power_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    height: int = 400
) -> go.Figure:
    """
    Create 2D plot of Power vs Irradiance at different temperatures.

    Args:
        power_matrix: Pmax values array
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for j, temp in enumerate(temperatures):
        power_values = power_matrix[:, j]
        valid = ~np.isnan(power_values)

        fig.add_trace(go.Scatter(
            x=np.array(irradiances)[valid],
            y=power_values[valid],
            mode='lines+markers',
            name=f'{temp}°C',
            line=dict(color=colors[j % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=f"T={temp}°C<br>G=%{{x}} W/m²<br>P=%{{y:.2f}} W<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Power vs Irradiance at Different Temperatures", x=0.5),
        xaxis=dict(title="Irradiance (W/m²)", gridcolor=PLOT_THEME['gridcolor']),
        yaxis=dict(title="Power (W)", gridcolor=PLOT_THEME['gridcolor']),
        legend=dict(title="Temperature", orientation='h', yanchor='bottom', y=1.02),
        height=height,
        hovermode='x unified',
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
    )

    return fig


def create_power_vs_temperature_plot(
    power_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    height: int = 400
) -> go.Figure:
    """
    Create 2D plot of Power vs Temperature at different irradiances.

    Shows temperature coefficient behavior per IEC 61853-1.

    Args:
        power_matrix: Pmax values array
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    colors = ['#440154', '#482878', '#3e4a89', '#31688e', '#26838f', '#1f9d8a', '#6cce5a']

    for i, irr in enumerate(irradiances):
        power_values = power_matrix[i, :]
        valid = ~np.isnan(power_values)

        fig.add_trace(go.Scatter(
            x=np.array(temperatures)[valid],
            y=power_values[valid],
            mode='lines+markers',
            name=f'{irr} W/m²',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=f"G={irr} W/m²<br>T=%{{x}}°C<br>P=%{{y:.2f}} W<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Power vs Temperature at Different Irradiances", x=0.5),
        xaxis=dict(title="Temperature (°C)", gridcolor=PLOT_THEME['gridcolor']),
        yaxis=dict(title="Power (W)", gridcolor=PLOT_THEME['gridcolor']),
        legend=dict(title="Irradiance", orientation='h', yanchor='bottom', y=1.02),
        height=height,
        hovermode='x unified',
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
    )

    return fig


def create_temperature_coefficients_plot(
    alpha_isc: float,
    beta_voc: float,
    gamma_pmax: float,
    temp_range: Tuple[float, float] = (15, 75),
    height: int = 400
) -> go.Figure:
    """
    Create visualization of temperature coefficients.

    Shows normalized parameter variation with temperature relative to STC.

    Args:
        alpha_isc: Isc temperature coefficient (%/°C)
        beta_voc: Voc temperature coefficient (%/°C)
        gamma_pmax: Pmax temperature coefficient (%/°C)
        temp_range: Temperature range for plot
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    temps = np.linspace(temp_range[0], temp_range[1], 100)
    t_ref = 25.0

    # Calculate normalized values
    isc_norm = 1 + (alpha_isc / 100) * (temps - t_ref)
    voc_norm = 1 + (beta_voc / 100) * (temps - t_ref)
    pmax_norm = 1 + (gamma_pmax / 100) * (temps - t_ref)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temps, y=isc_norm * 100,
        mode='lines',
        name=f'Isc (α={alpha_isc:+.3f}%/°C)',
        line=dict(color='#2ca02c', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=temps, y=voc_norm * 100,
        mode='lines',
        name=f'Voc (β={beta_voc:+.3f}%/°C)',
        line=dict(color='#d62728', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=temps, y=pmax_norm * 100,
        mode='lines',
        name=f'Pmax (γ={gamma_pmax:+.3f}%/°C)',
        line=dict(color='#1f77b4', width=3),
    ))

    # Add reference line at STC
    fig.add_vline(x=25, line_dash="dash", line_color="gray", annotation_text="STC (25°C)")
    fig.add_hline(y=100, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=dict(text="Temperature Coefficient Behavior", x=0.5),
        xaxis=dict(title="Temperature (°C)", gridcolor=PLOT_THEME['gridcolor']),
        yaxis=dict(title="Relative Value (% of STC)", gridcolor=PLOT_THEME['gridcolor']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=height,
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
    )

    return fig


# =============================================================================
# CSER and Energy Plots
# =============================================================================

def create_monthly_energy_plot(
    monthly_energy: List[float],
    climate_name: str = "Climate Profile",
    height: int = 400
) -> go.Figure:
    """
    Create bar chart of monthly energy production.

    Args:
        monthly_energy: List of 12 monthly energy values in kWh
        climate_name: Name of climate profile
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=months,
        y=monthly_energy,
        marker_color='#1f77b4',
        text=[f'{e:.1f}' for e in monthly_energy],
        textposition='outside',
        hovertemplate="%{x}: %{y:.2f} kWh<extra></extra>"
    ))

    annual_total = sum(monthly_energy)

    fig.update_layout(
        title=dict(
            text=f"Monthly Energy Production - {climate_name}<br>"
                 f"<sup>Annual Total: {annual_total:.1f} kWh</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Month", gridcolor=PLOT_THEME['gridcolor']),
        yaxis=dict(title="Energy (kWh)", gridcolor=PLOT_THEME['gridcolor']),
        height=height,
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        showlegend=False,
    )

    return fig


def create_climate_comparison_plot(
    cser_results: Dict[str, float],
    height: int = 400
) -> go.Figure:
    """
    Create comparison chart of CSER ratings across climate profiles.

    Args:
        cser_results: Dictionary mapping climate name to CSER rating
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    climates = list(cser_results.keys())
    ratings = list(cser_results.values())

    # Color by rating value
    colors = ['#2ca02c' if r > 1500 else '#ff7f0e' if r > 1200 else '#d62728'
              for r in ratings]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=climates,
        y=ratings,
        marker_color=colors,
        text=[f'{r:.0f}' for r in ratings],
        textposition='outside',
        hovertemplate="%{x}<br>CSER: %{y:.1f} kWh/kWp<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="CSER Comparison Across Climate Profiles", x=0.5),
        xaxis=dict(title="Climate Profile", tickangle=45),
        yaxis=dict(title="CSER Rating (kWh/kWp)"),
        height=height,
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
    )

    return fig


def create_heatmap_plot(
    power_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    title: str = "Power Matrix Heatmap",
    height: int = 400
) -> go.Figure:
    """
    Create 2D heatmap of power matrix.

    Args:
        power_matrix: Pmax values array
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        title: Plot title
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=power_matrix,
        x=[f'{t}°C' for t in temperatures],
        y=[f'{g} W/m²' for g in irradiances],
        colorscale=COLORSCALES['power'],
        colorbar=dict(title="Power (W)"),
        hovertemplate="T: %{x}<br>G: %{y}<br>P: %{z:.2f} W<extra></extra>"
    ))

    # Add text annotations
    for i in range(len(irradiances)):
        for j in range(len(temperatures)):
            val = power_matrix[i, j]
            if not np.isnan(val):
                fig.add_annotation(
                    x=j, y=i,
                    text=f'{val:.1f}',
                    showarrow=False,
                    font=dict(size=10, color='white' if val > np.nanmax(power_matrix) / 2 else 'black')
                )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Temperature"),
        yaxis=dict(title="Irradiance"),
        height=height,
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
    )

    return fig


def create_performance_gauge(
    value: float,
    title: str,
    suffix: str = "",
    reference: Optional[float] = None,
    ranges: Optional[List[Dict]] = None,
    height: int = 250
) -> go.Figure:
    """
    Create a gauge chart for KPI display.

    Args:
        value: Current value to display
        title: Gauge title
        suffix: Unit suffix
        reference: Reference value for delta
        ranges: List of range dicts with 'range' and 'color' keys
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    if ranges is None:
        ranges = [
            {'range': [0, 50], 'color': '#d62728'},
            {'range': [50, 75], 'color': '#ff7f0e'},
            {'range': [75, 100], 'color': '#2ca02c'},
        ]

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta" if reference else "gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=16)),
        number=dict(suffix=suffix, font=dict(size=24)),
        delta=dict(reference=reference) if reference else None,
        gauge=dict(
            axis=dict(range=[ranges[0]['range'][0], ranges[-1]['range'][1]]),
            bar=dict(color='#1f77b4'),
            steps=ranges,
            threshold=dict(
                line=dict(color="black", width=2),
                thickness=0.75,
                value=value
            )
        )
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
    )

    return fig
