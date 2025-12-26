"""
2D chart visualizations for PV-CSER Pro.

Provides various 2D charts for data analysis and reporting.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Month names for charts
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def create_monthly_bar_chart(
    monthly_values: List[float],
    title: str = "Monthly Energy Yield",
    y_label: str = "Energy (kWh)",
    color: str = "#FF6B35",
    height: int = 400,
    show_average: bool = True,
) -> go.Figure:
    """
    Create monthly bar chart.

    Args:
        monthly_values: List of 12 monthly values
        title: Chart title
        y_label: Y-axis label
        color: Bar color
        height: Chart height
        show_average: Whether to show average line

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=MONTHS,
        y=monthly_values,
        marker_color=color,
        text=[f"{v:.1f}" for v in monthly_values],
        textposition='outside',
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))

    # Add average line
    if show_average:
        avg = np.mean(monthly_values)
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Avg: {avg:.1f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Month"),
        yaxis=dict(title=y_label),
        height=height,
        showlegend=False,
    )

    return fig


def create_cser_comparison_chart(
    cser_values: Dict[str, float],
    title: str = "CSER Comparison by Climate",
    height: int = 400,
) -> go.Figure:
    """
    Create CSER comparison bar chart.

    Args:
        cser_values: Dictionary of {climate_name: cser_value}
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    profiles = list(cser_values.keys())
    values = list(cser_values.values())

    # Sort by value
    sorted_indices = np.argsort(values)[::-1]
    profiles = [profiles[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Color gradient based on value
    colors = px.colors.sample_colorscale(
        "YlOrRd",
        [v / max(values) for v in values],
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=profiles,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        hovertemplate="%{x}<br>CSER: %{y:.0f} kWh/kWp<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Climate Profile", tickangle=-45),
        yaxis=dict(title="CSER (kWh/kWp)"),
        height=height,
    )

    return fig


def create_loss_breakdown_chart(
    losses: Dict[str, float],
    title: str = "Energy Loss Breakdown",
    height: int = 400,
) -> go.Figure:
    """
    Create loss breakdown pie/bar chart.

    Args:
        losses: Dictionary of {loss_type: percentage}
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    # Filter out zero/nan losses
    filtered = {k: v for k, v in losses.items() if v > 0 and not np.isnan(v)}

    labels = list(filtered.keys())
    values = list(filtered.values())

    # Format labels
    labels = [l.replace('_', ' ').title() for l in labels]

    # Create subplot with pie and bar
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        column_widths=[0.5, 0.5],
    )

    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=px.colors.qualitative.Set2[:len(labels)],
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        showlegend=False,
    )

    return fig


def create_temperature_coefficient_chart(
    temperatures: np.ndarray,
    power_values: np.ndarray,
    pmax_stc: float,
    gamma: float,
    title: str = "Temperature Coefficient Analysis",
    height: int = 400,
) -> go.Figure:
    """
    Create temperature coefficient visualization.

    Args:
        temperatures: Array of temperature values
        power_values: Array of power values at each temperature
        pmax_stc: Power at STC
        gamma: Temperature coefficient (%/°C)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Measured data points
    fig.add_trace(go.Scatter(
        x=temperatures,
        y=power_values,
        mode='markers',
        name='Measured',
        marker=dict(size=10, color='blue'),
        hovertemplate="T: %{x}°C<br>P: %{y:.1f}W<extra></extra>",
    ))

    # Linear fit line
    temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
    p_fit = pmax_stc * (1 + (gamma / 100) * (temp_range - 25))

    fig.add_trace(go.Scatter(
        x=temp_range,
        y=p_fit,
        mode='lines',
        name=f'Linear fit (γ = {gamma:.2f} %/°C)',
        line=dict(color='red', dash='dash'),
    ))

    # Add STC point
    fig.add_trace(go.Scatter(
        x=[25],
        y=[pmax_stc],
        mode='markers',
        name='STC',
        marker=dict(size=12, color='green', symbol='star'),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Cell Temperature (°C)"),
        yaxis=dict(title="Power (W)"),
        height=height,
        legend=dict(x=0.7, y=0.95),
    )

    return fig


def create_efficiency_map(
    irradiance: np.ndarray,
    temperature: np.ndarray,
    efficiency_matrix: np.ndarray,
    title: str = "Module Efficiency Map",
    height: int = 500,
) -> go.Figure:
    """
    Create efficiency heatmap.

    Args:
        irradiance: Irradiance values
        temperature: Temperature values
        efficiency_matrix: 2D efficiency values (%)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        x=temperature,
        y=irradiance,
        z=efficiency_matrix,
        colorscale="RdYlGn",
        colorbar=dict(title="Efficiency (%)"),
        hovertemplate=(
            "T: %{x}°C<br>"
            "G: %{y} W/m²<br>"
            "η: %{z:.1f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Temperature (°C)"),
        yaxis=dict(title="Irradiance (W/m²)"),
        height=height,
    )

    return fig


def create_performance_ratio_chart(
    monthly_pr: List[float],
    monthly_yield: List[float],
    title: str = "Monthly Performance",
    height: int = 400,
) -> go.Figure:
    """
    Create dual-axis chart for PR and yield.

    Args:
        monthly_pr: Monthly performance ratio values (%)
        monthly_yield: Monthly yield values (kWh/kWp)
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Yield bars
    fig.add_trace(
        go.Bar(
            x=MONTHS,
            y=monthly_yield,
            name="Yield (kWh/kWp)",
            marker_color="steelblue",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # PR line
    fig.add_trace(
        go.Scatter(
            x=MONTHS,
            y=monthly_pr,
            name="Performance Ratio (%)",
            mode='lines+markers',
            line=dict(color='darkorange', width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Month"),
        height=height,
        legend=dict(x=0.7, y=1.1, orientation='h'),
    )

    fig.update_yaxes(title_text="Yield (kWh/kWp)", secondary_y=False)
    fig.update_yaxes(title_text="Performance Ratio (%)", secondary_y=True)

    return fig


def create_irradiance_distribution(
    irradiance: np.ndarray,
    title: str = "Irradiance Distribution",
    height: int = 400,
) -> go.Figure:
    """
    Create irradiance frequency distribution.

    Args:
        irradiance: Array of hourly irradiance values
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    # Filter daytime hours
    daytime = irradiance[irradiance > 0]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daytime,
        nbinsx=50,
        marker_color='orange',
        opacity=0.7,
        name='Frequency',
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Irradiance (W/m²)"),
        yaxis=dict(title="Hours"),
        height=height,
        bargap=0.1,
    )

    # Add statistics annotation
    mean_g = np.mean(daytime)
    max_g = np.max(daytime)

    fig.add_annotation(
        x=0.95, y=0.95,
        xref="paper", yref="paper",
        text=f"Mean: {mean_g:.0f} W/m²<br>Max: {max_g:.0f} W/m²",
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
    )

    return fig
