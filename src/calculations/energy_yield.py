"""
Energy yield calculations for PV systems.

Provides comprehensive energy yield calculations including
hourly simulations, annual summaries, and loss analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .temperature_models import TemperatureModel, ThermalModelType


@dataclass
class YieldResult:
    """Results from energy yield calculations."""

    # Energy totals
    annual_energy_dc: float          # kWh DC
    annual_energy_ac: float          # kWh AC
    specific_yield: float            # kWh/kWp

    # Time series
    hourly_power: np.ndarray         # W
    hourly_energy: np.ndarray        # Wh

    # Monthly summaries
    monthly_energy: List[float]      # kWh
    monthly_irradiation: List[float] # kWh/m²

    # Performance metrics
    performance_ratio: float         # %
    capacity_factor: float           # %

    # Loss breakdown
    losses: Dict[str, float] = field(default_factory=dict)

    # Metadata
    system_info: Dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert hourly data to DataFrame."""
        hours = pd.date_range(
            start='2024-01-01',
            periods=len(self.hourly_power),
            freq='H'
        )
        return pd.DataFrame({
            'power_w': self.hourly_power,
            'energy_wh': self.hourly_energy,
        }, index=hours)


class EnergyYieldCalculator:
    """
    Comprehensive energy yield calculator.

    Calculates DC and AC energy yields with detailed loss modeling.
    """

    # Month hours (non-leap year)
    MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    MONTH_NAMES = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    def __init__(
        self,
        power_model: callable,
        pmax_stc: float,
        module_area: float,
        temp_coeff_pmax: float = -0.35,
        nmot: float = 45.0,
        inverter_efficiency: float = 0.96,
    ):
        """
        Initialize energy yield calculator.

        Args:
            power_model: Function(G, T) -> P that returns power in W
            pmax_stc: Maximum power at STC (W)
            module_area: Module area (m²)
            temp_coeff_pmax: Temperature coefficient for Pmax (%/°C)
            nmot: Nominal Module Operating Temperature (°C)
            inverter_efficiency: Inverter efficiency (0-1)
        """
        self.power_model = power_model
        self.pmax_stc = pmax_stc
        self.module_area = module_area
        self.temp_coeff_pmax = temp_coeff_pmax
        self.nmot = nmot
        self.inverter_efficiency = inverter_efficiency

        # Create temperature model
        self.temp_model = TemperatureModel.from_noct(nmot)

    def calculate_yield(
        self,
        irradiance: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        apply_losses: bool = True,
    ) -> YieldResult:
        """
        Calculate energy yield from hourly data.

        Args:
            irradiance: Hourly POA irradiance (W/m²)
            ambient_temp: Hourly ambient temperature (°C)
            wind_speed: Hourly wind speed (m/s)
            apply_losses: Whether to apply additional losses

        Returns:
            YieldResult with all calculated values
        """
        n_hours = len(irradiance)

        if wind_speed is None:
            wind_speed = np.ones(n_hours)

        # Initialize output arrays
        hourly_power = np.zeros(n_hours)
        cell_temps = np.zeros(n_hours)

        # Calculate hourly power
        for i in range(n_hours):
            if irradiance[i] > 0:
                # Calculate cell temperature
                cell_temps[i] = self.temp_model.calculate(
                    ambient_temp[i],
                    irradiance[i],
                    wind_speed[i],
                )

                # Calculate power from model
                hourly_power[i] = self.power_model(irradiance[i], cell_temps[i])

        # Apply additional losses if requested
        loss_factors = {}
        if apply_losses:
            # Soiling loss (2%)
            hourly_power *= 0.98
            loss_factors['soiling'] = 2.0

            # Mismatch loss (2%)
            hourly_power *= 0.98
            loss_factors['mismatch'] = 2.0

            # Wiring loss (1.5%)
            hourly_power *= 0.985
            loss_factors['wiring'] = 1.5

            # Reflection loss (3%)
            hourly_power *= 0.97
            loss_factors['reflection'] = 3.0

        # Clip to valid range
        hourly_power = np.clip(hourly_power, 0, self.pmax_stc * 1.1)

        # Calculate DC energy
        hourly_energy_dc = hourly_power  # Wh (assuming 1-hour intervals)
        annual_energy_dc = np.sum(hourly_energy_dc) / 1000  # kWh

        # Calculate AC energy
        hourly_energy_ac = hourly_energy_dc * self.inverter_efficiency
        annual_energy_ac = np.sum(hourly_energy_ac) / 1000  # kWh

        # Specific yield (kWh/kWp)
        specific_yield = annual_energy_dc / (self.pmax_stc / 1000)

        # Monthly breakdown
        monthly_energy = []
        monthly_irradiation = []
        hour_idx = 0

        for month_hours in self.MONTH_HOURS:
            end_idx = hour_idx + month_hours
            month_power = hourly_power[hour_idx:end_idx]
            month_irr = irradiance[hour_idx:end_idx]

            monthly_energy.append(np.sum(month_power) / 1000)  # kWh
            monthly_irradiation.append(np.sum(month_irr) / 1000)  # kWh/m²

            hour_idx = end_idx

        # Performance ratio
        total_irradiation = np.sum(irradiance) / 1000  # kWh/m²
        ideal_energy = total_irradiation * self.module_area * 0.20  # Assume 20% efficiency
        pr = (annual_energy_dc / ideal_energy) * 100 if ideal_energy > 0 else 0

        # More accurate PR calculation
        ideal_at_stc = np.sum(irradiance) * self.pmax_stc / 1000 / 1000  # kWh
        pr = (annual_energy_dc / ideal_at_stc) * 100 if ideal_at_stc > 0 else 0

        # Capacity factor
        max_possible = self.pmax_stc * 8760 / 1000  # kWh
        cf = (annual_energy_dc / max_possible) * 100

        # Loss analysis
        losses = self._calculate_losses(
            irradiance, ambient_temp, cell_temps, hourly_power, loss_factors
        )

        return YieldResult(
            annual_energy_dc=annual_energy_dc,
            annual_energy_ac=annual_energy_ac,
            specific_yield=specific_yield,
            hourly_power=hourly_power,
            hourly_energy=hourly_energy_dc,
            monthly_energy=monthly_energy,
            monthly_irradiation=monthly_irradiation,
            performance_ratio=pr,
            capacity_factor=cf,
            losses=losses,
            system_info={
                'pmax_stc': self.pmax_stc,
                'module_area': self.module_area,
                'nmot': self.nmot,
                'temp_coeff': self.temp_coeff_pmax,
            },
        )

    def _calculate_losses(
        self,
        irradiance: np.ndarray,
        ambient_temp: np.ndarray,
        cell_temp: np.ndarray,
        actual_power: np.ndarray,
        applied_loss_factors: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate detailed loss breakdown."""

        # Ideal energy at STC conditions
        ideal_power = self.pmax_stc * (irradiance / 1000.0)
        ideal_energy = np.sum(ideal_power) / 1000  # kWh

        # Actual energy
        actual_energy = np.sum(actual_power) / 1000  # kWh

        # Total loss
        total_loss = ((ideal_energy - actual_energy) / ideal_energy * 100
                      if ideal_energy > 0 else 0)

        # Temperature loss
        temp_losses = np.zeros_like(irradiance)
        daytime = irradiance > 0
        temp_losses[daytime] = (cell_temp[daytime] - 25) * abs(self.temp_coeff_pmax)
        avg_temp_loss = np.mean(temp_losses[daytime]) if np.any(daytime) else 0

        # Irradiance loss (non-linearity at low light)
        low_light = (irradiance > 0) & (irradiance < 400)
        irr_loss = 2.0 if np.any(low_light) else 0.5  # Estimate

        losses = {
            'total': total_loss,
            'temperature': avg_temp_loss,
            'low_irradiance': irr_loss,
            'inverter': (1 - self.inverter_efficiency) * 100,
            **applied_loss_factors,
        }

        return losses

    def calculate_monthly_statistics(
        self,
        yield_result: YieldResult,
    ) -> pd.DataFrame:
        """
        Generate monthly statistics summary.

        Args:
            yield_result: YieldResult from calculate_yield

        Returns:
            DataFrame with monthly statistics
        """
        data = {
            'Month': self.MONTH_NAMES,
            'Energy (kWh)': yield_result.monthly_energy,
            'Irradiation (kWh/m²)': yield_result.monthly_irradiation,
            'Yield (kWh/kWp)': [
                e / (self.pmax_stc / 1000) for e in yield_result.monthly_energy
            ],
        }

        df = pd.DataFrame(data)

        # Add totals row
        totals = pd.DataFrame([{
            'Month': 'Total',
            'Energy (kWh)': yield_result.annual_energy_dc,
            'Irradiation (kWh/m²)': sum(yield_result.monthly_irradiation),
            'Yield (kWh/kWp)': yield_result.specific_yield,
        }])

        return pd.concat([df, totals], ignore_index=True)

    @staticmethod
    def create_simple_power_model(
        pmax_stc: float,
        temp_coeff: float = -0.35,
    ) -> callable:
        """
        Create a simple power model function.

        Args:
            pmax_stc: Maximum power at STC (W)
            temp_coeff: Temperature coefficient (%/°C)

        Returns:
            Power model function(G, T) -> P
        """
        def power_model(g: float, t: float) -> float:
            # Linear irradiance scaling
            p_g = pmax_stc * (g / 1000.0)
            # Temperature correction
            temp_factor = 1 + (temp_coeff / 100) * (t - 25)
            return p_g * temp_factor

        return power_model
