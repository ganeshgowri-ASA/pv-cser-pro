"""
3D visualization plots for PV-CSER Pro.

Provides 3D surface and scatter plots for power matrix visualization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PowerMatrix3DPlot:
    """
    3D visualization for power matrix data.

    Creates interactive 3D surface plots showing power as a function
    of irradiance and temperature.
    """

    # Color schemes
    COLOR_SCHEMES = {
        "solar": "YlOrRd",
        "viridis": "Viridis",
        "plasma": "Plasma",
        "thermal": "Thermal",
        "blues": "Blues",
    }

    def __init__(
        self,
        irradiance: np.ndarray,
        temperature: np.ndarray,
        power_matrix: np.ndarray,
        pmax_stc: Optional[float] = None,
    ):
        """
        Initialize 3D plot generator.

        Args:
            irradiance: Array of irradiance values (W/m²)
            temperature: Array of temperature values (°C)
            power_matrix: 2D array of power values (W)
            pmax_stc: Rated power at STC for normalization
        """
        self.irradiance = irradiance
        self.temperature = temperature
        self.power_matrix = power_matrix
        self.pmax_stc = pmax_stc

    def create_surface_plot(
        self,
        title: str = "Power Matrix - P(G, T)",
        colorscale: str = "solar",
        show_stc: bool = True,
        height: int = 600,
        normalized: bool = False,
    ) -> go.Figure:
        """
        Create interactive 3D surface plot.

        Args:
            title: Plot title
            colorscale: Color scheme name
            show_stc: Whether to show STC point
            height: Plot height in pixels
            normalized: Whether to show normalized values

        Returns:
            Plotly Figure object
        """
        # Prepare data
        g_mesh, t_mesh = np.meshgrid(self.irradiance, self.temperature, indexing='ij')

        z_data = self.power_matrix.copy()
        z_label = "Power (W)"

        if normalized and self.pmax_stc:
            z_data = z_data / self.pmax_stc
            z_label = "Normalized Power"

        # Get colorscale
        cscale = self.COLOR_SCHEMES.get(colorscale, "YlOrRd")

        # Create surface plot
        fig = go.Figure()

        # Add surface
        fig.add_trace(go.Surface(
            x=g_mesh,
            y=t_mesh,
            z=z_data,
            colorscale=cscale,
            colorbar=dict(
                title=z_label,
                titleside="right",
            ),
            hovertemplate=(
                "Irradiance: %{x:.0f} W/m²<br>"
                "Temperature: %{y:.0f} °C<br>"
                "Power: %{z:.1f} W<br>"
                "<extra></extra>"
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    project_z=True,
                    highlightcolor="limegreen",
                    highlightwidth=2,
                )
            ),
        ))

        # Add STC point marker
        if show_stc and 1000 in self.irradiance and 25 in self.temperature:
            idx_g = np.where(self.irradiance == 1000)[0][0]
            idx_t = np.where(self.temperature == 25)[0][0]
            stc_power = z_data[idx_g, idx_t]

            fig.add_trace(go.Scatter3d(
                x=[1000],
                y=[25],
                z=[stc_power],
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                name='STC Point',
                hovertemplate=(
                    "<b>STC</b><br>"
                    "G: 1000 W/m²<br>"
                    "T: 25 °C<br>"
                    f"P: {stc_power:.1f} W<br>"
                    "<extra></extra>"
                ),
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18),
            ),
            scene=dict(
                xaxis=dict(title="Irradiance (W/m²)"),
                yaxis=dict(title="Temperature (°C)"),
                zaxis=dict(title=z_label),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                ),
            ),
            height=height,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    def create_contour_plot(
        self,
        title: str = "Power Contours",
        colorscale: str = "solar",
        height: int = 500,
    ) -> go.Figure:
        """
        Create 2D contour plot.

        Args:
            title: Plot title
            colorscale: Color scheme name
            height: Plot height in pixels

        Returns:
            Plotly Figure object
        """
        cscale = self.COLOR_SCHEMES.get(colorscale, "YlOrRd")

        fig = go.Figure(data=go.Contour(
            x=self.temperature,
            y=self.irradiance,
            z=self.power_matrix,
            colorscale=cscale,
            colorbar=dict(title="Power (W)"),
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white'),
            ),
            hovertemplate=(
                "Temperature: %{x:.0f} °C<br>"
                "Irradiance: %{y:.0f} W/m²<br>"
                "Power: %{z:.1f} W<br>"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(title="Temperature (°C)"),
            yaxis=dict(title="Irradiance (W/m²)"),
            height=height,
        )

        return fig

    def create_heatmap(
        self,
        title: str = "Power Matrix Heatmap",
        colorscale: str = "solar",
        height: int = 400,
        annotate: bool = True,
    ) -> go.Figure:
        """
        Create annotated heatmap.

        Args:
            title: Plot title
            colorscale: Color scheme name
            height: Plot height
            annotate: Whether to show value annotations

        Returns:
            Plotly Figure object
        """
        cscale = self.COLOR_SCHEMES.get(colorscale, "YlOrRd")

        # Create annotation text
        annotations = []
        if annotate:
            for i, g in enumerate(self.irradiance):
                for j, t in enumerate(self.temperature):
                    val = self.power_matrix[i, j]
                    if not np.isnan(val):
                        annotations.append(f"{val:.0f}")
                    else:
                        annotations.append("")

        fig = go.Figure(data=go.Heatmap(
            x=[f"{t}°C" for t in self.temperature],
            y=[f"{g} W/m²" for g in self.irradiance],
            z=self.power_matrix,
            colorscale=cscale,
            colorbar=dict(title="Power (W)"),
            text=np.array(annotations).reshape(self.power_matrix.shape) if annotate else None,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=10),
            hovertemplate=(
                "Temperature: %{x}<br>"
                "Irradiance: %{y}<br>"
                "Power: %{z:.1f} W<br>"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(title="Temperature"),
            yaxis=dict(title="Irradiance"),
            height=height,
        )

        return fig


def create_power_surface(
    irradiance: np.ndarray,
    temperature: np.ndarray,
    power_matrix: np.ndarray,
    title: str = "PV Module Power Surface",
    **kwargs,
) -> go.Figure:
    """
    Convenience function to create power surface plot.

    Args:
        irradiance: Irradiance array
        temperature: Temperature array
        power_matrix: 2D power array
        title: Plot title
        **kwargs: Additional arguments for PowerMatrix3DPlot

    Returns:
        Plotly Figure object
    """
    plotter = PowerMatrix3DPlot(irradiance, temperature, power_matrix)
    return plotter.create_surface_plot(title=title, **kwargs)


def create_dual_surface_comparison(
    data1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    data2: Tuple[np.ndarray, np.ndarray, np.ndarray],
    titles: Tuple[str, str] = ("Module 1", "Module 2"),
    height: int = 500,
) -> go.Figure:
    """
    Create side-by-side 3D surface comparison.

    Args:
        data1: Tuple of (irradiance, temperature, power) for first module
        data2: Tuple of (irradiance, temperature, power) for second module
        titles: Titles for each subplot
        height: Plot height

    Returns:
        Plotly Figure with two subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=titles,
    )

    for idx, (irr, temp, power) in enumerate([data1, data2], 1):
        g_mesh, t_mesh = np.meshgrid(irr, temp, indexing='ij')

        fig.add_trace(
            go.Surface(
                x=g_mesh,
                y=t_mesh,
                z=power,
                colorscale="YlOrRd" if idx == 1 else "Viridis",
                showscale=idx == 2,
            ),
            row=1, col=idx,
        )

    fig.update_layout(
        height=height,
        title="Module Comparison",
    )

    return fig
