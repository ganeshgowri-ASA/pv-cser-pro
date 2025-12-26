"""
Interactive visualization components for PV-CSER Pro.

Provides highly interactive Plotly charts for data exploration.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_interactive_power_matrix(
    irradiance: np.ndarray,
    temperature: np.ndarray,
    power_matrix: np.ndarray,
    pmax_stc: Optional[float] = None,
    height: int = 600,
) -> go.Figure:
    """
    Create fully interactive power matrix visualization.

    Features:
    - Toggle between surface, contour, and heatmap views
    - Show/hide STC marker
    - Adjustable colorscale

    Args:
        irradiance: Irradiance array (W/m²)
        temperature: Temperature array (°C)
        power_matrix: 2D power array (W)
        pmax_stc: Rated power at STC
        height: Chart height

    Returns:
        Plotly Figure with interactive controls
    """
    g_mesh, t_mesh = np.meshgrid(irradiance, temperature, indexing='ij')

    fig = go.Figure()

    # Surface trace
    fig.add_trace(go.Surface(
        x=g_mesh,
        y=t_mesh,
        z=power_matrix,
        colorscale='YlOrRd',
        colorbar=dict(title="Power (W)", x=1.02),
        visible=True,
        name='Surface',
        hovertemplate=(
            "G: %{x:.0f} W/m²<br>"
            "T: %{y:.0f}°C<br>"
            "P: %{z:.1f} W<extra></extra>"
        ),
    ))

    # Contour trace
    fig.add_trace(go.Contour(
        x=temperature,
        y=irradiance,
        z=power_matrix,
        colorscale='YlOrRd',
        visible=False,
        name='Contour',
        contours=dict(showlabels=True),
    ))

    # Heatmap trace
    fig.add_trace(go.Heatmap(
        x=[f"{t}°C" for t in temperature],
        y=[f"{g} W/m²" for g in irradiance],
        z=power_matrix,
        colorscale='YlOrRd',
        visible=False,
        name='Heatmap',
    ))

    # Add buttons for view switching
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.1,
                y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="3D Surface",
                        method="update",
                        args=[
                            {"visible": [True, False, False]},
                            {"scene": dict(
                                xaxis=dict(title="Irradiance (W/m²)"),
                                yaxis=dict(title="Temperature (°C)"),
                                zaxis=dict(title="Power (W)"),
                            )}
                        ],
                    ),
                    dict(
                        label="Contour",
                        method="update",
                        args=[
                            {"visible": [False, True, False]},
                            {"xaxis": dict(title="Temperature (°C)"),
                             "yaxis": dict(title="Irradiance (W/m²)")}
                        ],
                    ),
                    dict(
                        label="Heatmap",
                        method="update",
                        args=[
                            {"visible": [False, False, True]},
                            {"xaxis": dict(title="Temperature"),
                             "yaxis": dict(title="Irradiance")}
                        ],
                    ),
                ],
            ),
        ],
        title=dict(
            text="Interactive Power Matrix Analysis",
            font=dict(size=18),
        ),
        height=height,
        scene=dict(
            xaxis=dict(title="Irradiance (W/m²)"),
            yaxis=dict(title="Temperature (°C)"),
            zaxis=dict(title="Power (W)"),
        ),
    )

    return fig


def create_climate_comparison_chart(
    climate_data: Dict[str, Dict[str, Any]],
    metric: str = "cser",
    height: int = 500,
) -> go.Figure:
    """
    Create interactive climate comparison chart.

    Args:
        climate_data: Dictionary of {climate_name: {cser, annual_ghi, avg_temp, ...}}
        metric: Metric to display ('cser', 'annual_ghi', 'avg_temp')
        height: Chart height

    Returns:
        Plotly Figure object
    """
    profiles = list(climate_data.keys())

    fig = go.Figure()

    # CSER values
    cser_values = [climate_data[p].get('cser', 0) for p in profiles]
    fig.add_trace(go.Bar(
        x=profiles,
        y=cser_values,
        name='CSER (kWh/kWp)',
        marker_color='#FF6B35',
        visible=True,
        text=[f"{v:.0f}" for v in cser_values],
        textposition='outside',
    ))

    # GHI values
    ghi_values = [climate_data[p].get('annual_ghi', 0) for p in profiles]
    fig.add_trace(go.Bar(
        x=profiles,
        y=ghi_values,
        name='Annual GHI (kWh/m²)',
        marker_color='#FFC300',
        visible=False,
        text=[f"{v:.0f}" for v in ghi_values],
        textposition='outside',
    ))

    # Temperature values
    temp_values = [climate_data[p].get('avg_temp', 0) for p in profiles]
    fig.add_trace(go.Bar(
        x=profiles,
        y=temp_values,
        name='Avg Temp (°C)',
        marker_color='#C70039',
        visible=False,
        text=[f"{v:.1f}" for v in temp_values],
        textposition='outside',
    ))

    # Performance ratio
    pr_values = [climate_data[p].get('performance_ratio', 0) for p in profiles]
    fig.add_trace(go.Bar(
        x=profiles,
        y=pr_values,
        name='Performance Ratio (%)',
        marker_color='#900C3F',
        visible=False,
        text=[f"{v:.1f}" for v in pr_values],
        textposition='outside',
    ))

    # Add dropdown for metric selection
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="CSER",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False]},
                            {"yaxis": {"title": "CSER (kWh/kWp)"}}
                        ],
                    ),
                    dict(
                        label="Annual GHI",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False]},
                            {"yaxis": {"title": "Annual GHI (kWh/m²)"}}
                        ],
                    ),
                    dict(
                        label="Temperature",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False]},
                            {"yaxis": {"title": "Average Temperature (°C)"}}
                        ],
                    ),
                    dict(
                        label="Performance Ratio",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True]},
                            {"yaxis": {"title": "Performance Ratio (%)"}}
                        ],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
            ),
        ],
        title=dict(
            text="Climate Profile Comparison",
            font=dict(size=18),
        ),
        xaxis=dict(title="Climate Profile", tickangle=-45),
        yaxis=dict(title="CSER (kWh/kWp)"),
        height=height,
        showlegend=False,
    )

    return fig


def create_energy_yield_chart(
    monthly_energy: List[float],
    monthly_irradiation: List[float],
    title: str = "Energy Yield Analysis",
    height: int = 500,
) -> go.Figure:
    """
    Create comprehensive energy yield chart.

    Args:
        monthly_energy: Monthly energy values (kWh)
        monthly_irradiation: Monthly irradiation values (kWh/m²)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Energy Yield",
            "Monthly Irradiation",
            "Cumulative Energy",
            "Performance Ratio",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Monthly energy
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly_energy,
            marker_color='steelblue',
            name='Energy',
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Monthly irradiation
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly_irradiation,
            marker_color='orange',
            name='Irradiation',
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Cumulative energy
    cumulative = np.cumsum(monthly_energy)
    fig.add_trace(
        go.Scatter(
            x=months,
            y=cumulative,
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            name='Cumulative',
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Performance ratio (simplified calculation)
    pr = [e / (i * 0.2) * 100 if i > 0 else 0
          for e, i in zip(monthly_energy, monthly_irradiation)]

    fig.add_trace(
        go.Scatter(
            x=months,
            y=pr,
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=8),
            name='PR',
            showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
    )

    # Update axes labels
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Irradiation (kWh/m²)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="PR (%)", row=2, col=2)

    return fig


def create_module_comparison_radar(
    modules: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    height: int = 500,
) -> go.Figure:
    """
    Create radar chart for module comparison.

    Args:
        modules: Dictionary of {module_name: {metric: value, ...}}
        metrics: List of metrics to include
        height: Chart height

    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ['pmax_stc', 'efficiency', 'temp_coeff', 'cser']

    # Normalize values for radar chart
    all_values = {m: [] for m in metrics}
    for module_data in modules.values():
        for metric in metrics:
            all_values[metric].append(module_data.get(metric, 0))

    # Normalize each metric to 0-1 scale
    normalized = {}
    for module_name, module_data in modules.items():
        normalized[module_name] = []
        for metric in metrics:
            values = all_values[metric]
            max_val = max(values) if max(values) > 0 else 1
            min_val = min(values)
            val = module_data.get(metric, 0)

            # For temp_coeff, lower (more negative) is worse, so invert
            if metric == 'temp_coeff':
                normalized[module_name].append(1 - (val - min_val) / (max_val - min_val + 0.001))
            else:
                normalized[module_name].append((val - min_val) / (max_val - min_val + 0.001))

    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for idx, (name, values) in enumerate(normalized.items()):
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name=name,
            line=dict(color=colors[idx % len(colors)]),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            ),
        ),
        title=dict(text="Module Comparison", font=dict(size=18)),
        height=height,
        showlegend=True,
    )

    return fig


def create_hourly_profile_chart(
    hours: np.ndarray,
    power: np.ndarray,
    irradiance: np.ndarray,
    temperature: np.ndarray,
    day_of_year: int = 172,  # Summer solstice
    height: int = 400,
) -> go.Figure:
    """
    Create hourly profile chart for a typical day.

    Args:
        hours: Hour indices (0-23)
        power: Hourly power values
        irradiance: Hourly irradiance values
        temperature: Hourly temperature values
        day_of_year: Day of year to display
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Power output
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=power,
            mode='lines',
            name='Power (W)',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)',
        ),
        secondary_y=False,
    )

    # Irradiance
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=irradiance,
            mode='lines',
            name='Irradiance (W/m²)',
            line=dict(color='orange', width=2, dash='dash'),
        ),
        secondary_y=True,
    )

    # Temperature
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=temperature,
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='red', width=2, dash='dot'),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=f"Daily Profile (Day {day_of_year})",
            font=dict(size=16),
        ),
        xaxis=dict(title="Hour of Day"),
        height=height,
        legend=dict(x=0.7, y=1.15, orientation='h'),
    )

    fig.update_yaxes(title_text="Power (W)", secondary_y=False)
    fig.update_yaxes(title_text="Irradiance / Temperature", secondary_y=True)

    return fig
