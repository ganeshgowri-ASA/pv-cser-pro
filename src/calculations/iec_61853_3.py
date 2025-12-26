"""
IEC 61853-3: Energy rating methodology.

Implements energy rating calculations according to IEC 61853-3,
including Climate Specific Energy Rating (CSER) calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate


@dataclass
class EnergyRatingResult:
    """Results from IEC 61853-3 energy rating calculations."""

    # Annual energy yields
    annual_energy: float              # kWh
    annual_energy_per_kw: float       # kWh/kWp (CSER)

    # Monthly breakdown
    monthly_energy: List[float]       # kWh per month

    # Performance metrics
    performance_ratio: float          # %
    capacity_factor: float            # %

    # Loss breakdown
    losses: Dict[str, float] = field(default_factory=dict)

    # Metadata
    climate_profile: str = ""
    calculation_method: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "annual_energy": self.annual_energy,
            "cser": self.annual_energy_per_kw,
            "monthly_energy": self.monthly_energy,
            "performance_ratio": self.performance_ratio,
            "capacity_factor": self.capacity_factor,
            "losses": self.losses,
            "climate_profile": self.climate_profile,
            "calculation_method": self.calculation_method,
        }


class IEC61853Part3:
    """
    IEC 61853-3 Energy Rating Calculator.

    Implements the energy rating methodology including:
    - Hourly power calculations
    - Annual energy summation
    - Loss calculations
    - CSER determination
    """

    # Standard constants
    HOURS_PER_YEAR = 8760
    G_STC = 1000.0  # W/m²
    T_STC = 25.0    # °C

    def __init__(
        self,
        power_model: callable,
        pmax_stc: float,
        module_area: float,
        temp_coeff_pmax: float = -0.35,
    ):
        """
        Initialize energy rating calculator.

        Args:
            power_model: Function(G, T) -> P that returns power
            pmax_stc: Maximum power at STC (W)
            module_area: Module area (m²)
            temp_coeff_pmax: Temperature coefficient for Pmax (%/°C)
        """
        self.power_model = power_model
        self.pmax_stc = pmax_stc
        self.module_area = module_area
        self.temp_coeff_pmax = temp_coeff_pmax

    def calculate_cell_temperature(
        self,
        ambient_temp: float,
        irradiance: float,
        wind_speed: float = 1.0,
        nmot: float = 45.0,
    ) -> float:
        """
        Calculate cell temperature using NMOT model.

        T_cell = T_amb + G * (NMOT - 20) / 800 * (9.5 / (5.7 + 3.8 * v))

        Args:
            ambient_temp: Ambient temperature (°C)
            irradiance: Global irradiance (W/m²)
            wind_speed: Wind speed (m/s)
            nmot: Nominal Module Operating Temperature (°C)

        Returns:
            Cell temperature in °C
        """
        if irradiance <= 0:
            return ambient_temp

        # NMOT-based model with wind correction
        # Ross coefficient approximation
        delta_t_nmot = nmot - 20.0
        wind_factor = 9.5 / (5.7 + 3.8 * wind_speed)

        delta_t = irradiance * (delta_t_nmot / 800.0) * wind_factor

        return ambient_temp + delta_t

    def calculate_hourly_power(
        self,
        irradiance: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        nmot: float = 45.0,
    ) -> np.ndarray:
        """
        Calculate hourly power output.

        Args:
            irradiance: Array of hourly irradiance values (W/m²)
            ambient_temp: Array of hourly ambient temperatures (°C)
            wind_speed: Array of hourly wind speeds (m/s)
            nmot: Nominal Module Operating Temperature (°C)

        Returns:
            Array of hourly power values (W)
        """
        n_hours = len(irradiance)

        if wind_speed is None:
            wind_speed = np.ones(n_hours) * 1.0  # Default 1 m/s

        power = np.zeros(n_hours)

        for i in range(n_hours):
            if irradiance[i] > 0:
                # Calculate cell temperature
                t_cell = self.calculate_cell_temperature(
                    ambient_temp[i],
                    irradiance[i],
                    wind_speed[i],
                    nmot,
                )

                # Calculate power
                power[i] = self.power_model(irradiance[i], t_cell)

        # Clip to valid range
        power = np.clip(power, 0, self.pmax_stc * 1.2)

        return power

    def calculate_annual_energy(
        self,
        irradiance: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        nmot: float = 45.0,
    ) -> EnergyRatingResult:
        """
        Calculate annual energy yield.

        Args:
            irradiance: Hourly irradiance (W/m², 8760 values)
            ambient_temp: Hourly ambient temperature (°C, 8760 values)
            wind_speed: Hourly wind speed (m/s, 8760 values)
            nmot: Nominal Module Operating Temperature (°C)

        Returns:
            EnergyRatingResult with all calculated values
        """
        # Calculate hourly power
        hourly_power = self.calculate_hourly_power(
            irradiance, ambient_temp, wind_speed, nmot
        )

        # Annual energy (Wh -> kWh)
        annual_energy = np.sum(hourly_power) / 1000.0

        # CSER (kWh/kWp)
        cser = annual_energy / (self.pmax_stc / 1000.0)

        # Monthly breakdown
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        monthly_energy = []
        start_idx = 0
        for hours in hours_per_month:
            end_idx = start_idx + hours
            month_power = hourly_power[start_idx:end_idx]
            monthly_energy.append(np.sum(month_power) / 1000.0)
            start_idx = end_idx

        # Calculate performance ratio
        # PR = E_actual / E_ideal
        total_irradiation = np.sum(irradiance)  # Wh/m²
        e_ideal = (total_irradiation / 1000.0) * (self.pmax_stc / 1000.0)  # kWh
        pr = (annual_energy / e_ideal) * 100 if e_ideal > 0 else 0

        # Capacity factor
        max_possible = (self.pmax_stc / 1000.0) * 8760  # kWh
        cf = (annual_energy / max_possible) * 100 if max_possible > 0 else 0

        # Calculate losses
        losses = self._calculate_losses(
            irradiance, ambient_temp, hourly_power, nmot
        )

        return EnergyRatingResult(
            annual_energy=annual_energy,
            annual_energy_per_kw=cser,
            monthly_energy=monthly_energy,
            performance_ratio=pr,
            capacity_factor=cf,
            losses=losses,
            calculation_method="IEC 61853-3",
        )

    def _calculate_losses(
        self,
        irradiance: np.ndarray,
        ambient_temp: np.ndarray,
        actual_power: np.ndarray,
        nmot: float,
    ) -> Dict[str, float]:
        """
        Calculate detailed loss breakdown.

        Returns:
            Dictionary with loss percentages
        """
        # Ideal power at STC
        ideal_power = np.where(
            irradiance > 0,
            self.pmax_stc * (irradiance / 1000.0),
            0,
        )
        ideal_energy = np.sum(ideal_power) / 1000.0  # kWh

        # Actual energy
        actual_energy = np.sum(actual_power) / 1000.0  # kWh

        # Total loss
        total_loss = (1 - actual_energy / ideal_energy) * 100 if ideal_energy > 0 else 0

        # Estimate temperature loss
        temp_loss_factor = 0
        for i, g in enumerate(irradiance):
            if g > 0:
                t_cell = self.calculate_cell_temperature(ambient_temp[i], g, 1.0, nmot)
                temp_loss_factor += (t_cell - 25) * abs(self.temp_coeff_pmax) / 100

        temp_loss = temp_loss_factor / len(irradiance[irradiance > 0]) * 100 if np.sum(irradiance > 0) > 0 else 0

        # Low irradiance loss (efficiency drops at low G)
        low_irr_loss = total_loss - temp_loss
        low_irr_loss = max(0, low_irr_loss)

        return {
            "total_loss": total_loss,
            "temperature_loss": temp_loss,
            "low_irradiance_loss": low_irr_loss,
            "reflection_loss": 3.0,  # Typical estimate
            "spectral_loss": 1.0,    # Typical estimate
            "soiling_loss": 2.0,     # Typical estimate
        }

    def calculate_cser_comparison(
        self,
        climate_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
        nmot: float = 45.0,
    ) -> Dict[str, float]:
        """
        Calculate CSER for multiple climate profiles.

        Args:
            climate_profiles: Dictionary of {name: (irradiance, temperature)}
            nmot: Nominal Module Operating Temperature

        Returns:
            Dictionary of {profile_name: cser_value}
        """
        results = {}

        for name, (irr, temp) in climate_profiles.items():
            energy_result = self.calculate_annual_energy(irr, temp, nmot=nmot)
            results[name] = energy_result.annual_energy_per_kw

        return results

    def calculate_power_bins(
        self,
        irradiance_bins: np.ndarray,
        temperature_bins: np.ndarray,
        bin_hours: np.ndarray,
    ) -> float:
        """
        Calculate energy using binned climate data (IEC 61853-3 method).

        Args:
            irradiance_bins: Array of irradiance bin centers (W/m²)
            temperature_bins: Array of temperature bin centers (°C)
            bin_hours: 2D array of hours in each (G, T) bin

        Returns:
            Annual energy in kWh
        """
        total_energy = 0.0

        for i, g in enumerate(irradiance_bins):
            for j, t in enumerate(temperature_bins):
                hours = bin_hours[i, j]
                if hours > 0 and g > 0:
                    power = self.power_model(g, t)
                    total_energy += power * hours

        return total_energy / 1000.0  # Wh -> kWh
