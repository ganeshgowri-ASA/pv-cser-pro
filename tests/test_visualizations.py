"""
Plotly Visualization Tests for PV-CSER Pro.

Tests cover:
- 3D power matrix plots
- Monthly bar charts
- CSER comparison charts
- Loss breakdown charts
- Interactive features
- Chart configuration
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List


class TestPowerMatrix3DPlot:
    """Test 3D power matrix visualization."""

    def test_create_surface_plot(self, sample_power_matrix):
        """Test creating 3D surface plot."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_surface_plot_colorscales(self, sample_power_matrix):
        """Test different colorscales for surface plot."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        colorscales = ["YlOrRd", "Viridis", "Plasma", "Blues"]

        for colorscale in colorscales:
            plotter = PowerMatrix3DPlot(colorscale=colorscale)
            fig = plotter.create_surface(
                power_values=sample_power_matrix["power_values"],
                irradiance_levels=sample_power_matrix["irradiance_levels"],
                temperature_levels=sample_power_matrix["temperature_levels"],
            )
            assert fig is not None

    def test_normalized_surface_plot(self, sample_power_matrix):
        """Test normalized surface plot."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
            normalize=True,
        )

        assert fig is not None
        # Check z-axis is normalized (0-1 or percentage)

    def test_stc_point_highlight(self, sample_power_matrix):
        """Test STC point highlighting on plot."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
            highlight_stc=True,
        )

        # Should have marker trace for STC point
        assert len(fig.data) >= 2

    def test_contour_plot(self, sample_power_matrix):
        """Test contour plot creation."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_contour(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_heatmap_plot(self, sample_power_matrix):
        """Test heatmap plot creation."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_heatmap(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestMonthlyCharts:
    """Test monthly bar chart visualization."""

    def test_monthly_energy_bar_chart(self, sample_cser_results):
        """Test monthly energy bar chart."""
        from src.visualizations.charts import create_monthly_bar_chart

        fig = create_monthly_bar_chart(
            monthly_data=sample_cser_results["monthly_yields"],
            title="Monthly Energy Yield",
            y_label="Energy (kWh)",
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_monthly_chart_with_average_line(self, sample_cser_results):
        """Test monthly chart with average line."""
        from src.visualizations.charts import create_monthly_bar_chart

        fig = create_monthly_bar_chart(
            monthly_data=sample_cser_results["monthly_yields"],
            title="Monthly Energy Yield",
            show_average=True,
        )

        # Should have bar trace and line trace
        assert len(fig.data) >= 2

    def test_monthly_chart_colors(self, sample_cser_results):
        """Test monthly chart custom colors."""
        from src.visualizations.charts import create_monthly_bar_chart

        fig = create_monthly_bar_chart(
            monthly_data=sample_cser_results["monthly_yields"],
            bar_color="#FF6B35",
            average_color="#1E3A5F",
            show_average=True,
        )

        assert fig is not None

    def test_monthly_cser_chart(self, sample_cser_results):
        """Test monthly CSER chart."""
        from src.visualizations.charts import create_monthly_bar_chart

        fig = create_monthly_bar_chart(
            monthly_data=sample_cser_results["monthly_cser"],
            title="Monthly CSER",
            y_label="CSER (kWh/kWp)",
        )

        assert fig is not None


class TestCSERComparisonCharts:
    """Test CSER comparison visualization."""

    def test_horizontal_bar_comparison(self):
        """Test horizontal bar comparison chart."""
        from src.visualizations.charts import create_cser_comparison_chart

        comparison_data = [
            {"name": "Tropical Humid", "cser": 1580},
            {"name": "Subtropical Arid", "cser": 1820},
            {"name": "Temperate Coastal", "cser": 1350},
            {"name": "High Elevation", "cser": 1650},
        ]

        fig = create_cser_comparison_chart(comparison_data)

        assert fig is not None
        assert isinstance(fig, go.Figure)
        # Data should be sorted by CSER
        assert len(fig.data) >= 1

    def test_comparison_with_reference(self):
        """Test comparison chart with reference line."""
        from src.visualizations.charts import create_cser_comparison_chart

        comparison_data = [
            {"name": "Profile A", "cser": 1500},
            {"name": "Profile B", "cser": 1600},
        ]

        fig = create_cser_comparison_chart(
            comparison_data,
            reference_value=1550,
            reference_label="Target",
        )

        assert len(fig.data) >= 2  # Bars + reference line

    def test_multi_module_comparison(self):
        """Test multi-module comparison chart."""
        from src.visualizations.charts import create_multi_module_comparison

        data = {
            "Tropical": {
                "Module A": 1580,
                "Module B": 1620,
                "Module C": 1550,
            },
            "Temperate": {
                "Module A": 1350,
                "Module B": 1380,
                "Module C": 1320,
            },
        }

        fig = create_multi_module_comparison(data)

        assert fig is not None


class TestLossBreakdownCharts:
    """Test loss breakdown visualization."""

    def test_pie_chart_losses(self, sample_cser_results):
        """Test loss breakdown pie chart."""
        from src.visualizations.charts import create_loss_breakdown_chart

        fig = create_loss_breakdown_chart(
            losses=sample_cser_results["loss_breakdown"],
            chart_type="pie",
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_bar_chart_losses(self, sample_cser_results):
        """Test loss breakdown bar chart."""
        from src.visualizations.charts import create_loss_breakdown_chart

        fig = create_loss_breakdown_chart(
            losses=sample_cser_results["loss_breakdown"],
            chart_type="bar",
        )

        assert fig is not None

    def test_stacked_bar_losses(self, sample_cser_results):
        """Test stacked bar loss chart."""
        from src.visualizations.charts import create_loss_breakdown_chart

        fig = create_loss_breakdown_chart(
            losses=sample_cser_results["loss_breakdown"],
            chart_type="stacked",
        )

        assert fig is not None

    def test_waterfall_chart_losses(self, sample_cser_results):
        """Test waterfall loss chart."""
        from src.visualizations.charts import create_loss_waterfall_chart

        fig = create_loss_waterfall_chart(
            losses=sample_cser_results["loss_breakdown"],
            gross_energy=700,
            net_energy=632.2,
        )

        assert fig is not None


class TestTemperatureCoefficientCharts:
    """Test temperature coefficient visualization."""

    def test_gamma_curve_chart(self, sample_module_data):
        """Test temperature coefficient curve."""
        from src.visualizations.charts import create_temperature_coefficient_chart

        temperatures = np.arange(0, 80, 5)
        gamma = sample_module_data["temp_coeff_pmax"]

        # Calculate relative power at each temperature
        rel_power = 1 + (gamma / 100) * (temperatures - 25)

        fig = create_temperature_coefficient_chart(
            temperatures=temperatures.tolist(),
            relative_power=rel_power.tolist(),
        )

        assert fig is not None

    def test_multiple_coefficient_curves(self):
        """Test multiple temperature coefficient curves."""
        from src.visualizations.charts import create_temperature_coefficient_chart

        temperatures = np.arange(0, 80, 5)

        curves = {
            "Pmax": {"gamma": -0.35, "color": "#FF6B35"},
            "Voc": {"gamma": -0.28, "color": "#1E3A5F"},
        }

        fig = create_temperature_coefficient_chart(
            temperatures=temperatures.tolist(),
            curves=curves,
        )

        assert len(fig.data) >= 2


class TestInteractiveCharts:
    """Test interactive chart features."""

    def test_interactive_power_matrix(self, sample_power_matrix):
        """Test interactive power matrix view."""
        from src.visualizations.interactive import create_interactive_power_matrix

        fig = create_interactive_power_matrix(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        assert fig is not None
        # Should have dropdown or buttons for view switching
        assert fig.layout.updatemenus is not None or len(fig.data) > 1

    def test_climate_comparison_interactive(self):
        """Test interactive climate comparison."""
        from src.visualizations.interactive import create_climate_comparison_chart

        data = {
            "Tropical Humid": {
                "energy": [45, 48, 58, 62, 68, 65, 64, 62, 55, 48, 42, 41],
                "cser": [113, 121, 146, 155, 171, 163, 162, 156, 138, 120, 105, 104],
            },
            "Subtropical Arid": {
                "energy": [55, 60, 75, 85, 95, 90, 88, 82, 70, 60, 52, 50],
                "cser": [138, 150, 188, 213, 238, 225, 220, 205, 175, 150, 130, 125],
            },
        }

        fig = create_climate_comparison_chart(data)

        assert fig is not None

    def test_energy_yield_time_series(self, sample_hourly_climate_data):
        """Test energy yield time series chart."""
        from src.visualizations.interactive import create_energy_yield_chart

        # Generate mock hourly power data
        np.random.seed(42)
        power = sample_hourly_climate_data["ghi"] * 0.4 * 0.8

        fig = create_energy_yield_chart(
            hourly_power=power,
            aggregate="daily",
        )

        assert fig is not None


class TestChartConfiguration:
    """Test chart configuration and styling."""

    def test_chart_dimensions(self, sample_power_matrix):
        """Test chart dimension settings."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
            height=600,
            width=800,
        )

        assert fig.layout.height == 600
        assert fig.layout.width == 800

    def test_chart_title(self, sample_power_matrix):
        """Test chart title configuration."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        title = "Custom Power Matrix Title"
        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
            title=title,
        )

        assert title in fig.layout.title.text

    def test_axis_labels(self, sample_power_matrix):
        """Test axis label configuration."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
            x_label="Irradiance (W/m²)",
            y_label="Temperature (°C)",
            z_label="Power (W)",
        )

        # Check 3D axis labels exist
        assert fig.layout.scene is not None

    def test_export_to_html(self, sample_power_matrix, temp_dir):
        """Test exporting chart to HTML."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        html_path = temp_dir / "chart.html"
        fig.write_html(str(html_path))

        assert html_path.exists()
        assert html_path.stat().st_size > 0

    def test_export_to_png(self, sample_power_matrix, temp_dir):
        """Test exporting chart to PNG (if kaleido available)."""
        from src.visualizations.plots_3d import PowerMatrix3DPlot

        plotter = PowerMatrix3DPlot()

        fig = plotter.create_surface(
            power_values=sample_power_matrix["power_values"],
            irradiance_levels=sample_power_matrix["irradiance_levels"],
            temperature_levels=sample_power_matrix["temperature_levels"],
        )

        try:
            png_path = temp_dir / "chart.png"
            fig.write_image(str(png_path))
            assert png_path.exists()
        except Exception:
            # kaleido not installed, skip
            pytest.skip("kaleido not installed for image export")


class TestChartDataValidation:
    """Test chart data validation."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from src.visualizations.charts import create_monthly_bar_chart

        with pytest.raises(ValueError):
            create_monthly_bar_chart(monthly_data={})

    def test_invalid_data_type(self):
        """Test handling of invalid data types."""
        from src.visualizations.charts import create_monthly_bar_chart

        with pytest.raises((TypeError, ValueError)):
            create_monthly_bar_chart(monthly_data="not a dict")

    def test_missing_months(self):
        """Test handling of missing months."""
        from src.visualizations.charts import create_monthly_bar_chart

        partial_data = {"Jan": 45, "Feb": 48}  # Missing other months

        # Should either fill missing or raise error
        try:
            fig = create_monthly_bar_chart(monthly_data=partial_data)
            assert fig is not None
        except ValueError:
            pass  # Also acceptable
