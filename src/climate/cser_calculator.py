"""
CSER (Climate Specific Energy Rating) Calculator.

Implements the Climate Specific Energy Rating methodology
according to IEC 61853-4 standards.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .climate_profiles import ClimateProfile, ClimateProfileManager


@dataclass
class CSERResult:
    """Results from CSER calculation."""

    # Main CSER value
    cser: float                      # kWh/kWp

    # Breakdown
    annual_energy: float             # kWh (for rated module)
    monthly_energy: List[float]      # kWh per month
    monthly_cser: List[float]        # kWh/kWp per month

    # Performance metrics
    performance_ratio: float         # %
    temperature_loss: float          # %
    low_irradiance_loss: float       # %

    # Climate info
    climate_profile: str
    annual_irradiation: float        # kWh/m²
    avg_temperature: float           # °C

    # Comparison to STC
    relative_to_stc: float           # % (CSER as percentage of ideal)

    # Metadata
    module_pmax: float              # W
    calculation_details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cser": self.cser,
            "annual_energy": self.annual_energy,
            "monthly_energy": self.monthly_energy,
            "monthly_cser": self.monthly_cser,
            "performance_ratio": self.performance_ratio,
            "temperature_loss": self.temperature_loss,
            "low_irradiance_loss": self.low_irradiance_loss,
            "climate_profile": self.climate_profile,
            "annual_irradiation": self.annual_irradiation,
            "avg_temperature": self.avg_temperature,
            "relative_to_stc": self.relative_to_stc,
            "module_pmax": self.module_pmax,
        }


class CSERCalculator:
    """
    Climate Specific Energy Rating Calculator.

    Calculates CSER values for PV modules according to IEC 61853-4,
    considering climate-specific operating conditions.
    """

    HOURS_PER_YEAR = 8760
    HOURS_PER_MONTH = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    G_STC = 1000.0  # W/m²
    T_STC = 25.0    # °C

    def __init__(
        self,
        power_model: Callable[[float, float], float],
        pmax_stc: float,
        temp_coeff_pmax: float = -0.35,
        nmot: float = 45.0,
    ):
        """
        Initialize CSER calculator.

        Args:
            power_model: Function(G, T) -> P returning power in W
            pmax_stc: Rated power at STC (W)
            temp_coeff_pmax: Power temperature coefficient (%/°C)
            nmot: Nominal Module Operating Temperature (°C)
        """
        self.power_model = power_model
        self.pmax_stc = pmax_stc
        self.temp_coeff_pmax = temp_coeff_pmax
        self.nmot = nmot

    def calculate_cell_temperature(
        self,
        ambient_temp: float,
        irradiance: float,
        wind_speed: float = 1.0,
    ) -> float:
        """
        Calculate cell temperature from ambient conditions.

        Args:
            ambient_temp: Ambient temperature (°C)
            irradiance: Irradiance (W/m²)
            wind_speed: Wind speed (m/s)

        Returns:
            Cell temperature (°C)
        """
        if irradiance <= 0:
            return ambient_temp

        # NMOT-based model with wind correction
        wind_factor = 9.5 / (5.7 + 3.8 * wind_speed)
        delta_t = irradiance * ((self.nmot - 20.0) / 800.0) * wind_factor

        return ambient_temp + delta_t

    def calculate_cser(
        self,
        climate_profile: ClimateProfile,
    ) -> CSERResult:
        """
        Calculate CSER for a given climate profile.

        Args:
            climate_profile: Climate profile with hourly data

        Returns:
            CSERResult with all calculated values
        """
        ghi = climate_profile.ghi
        ambient_temp = climate_profile.ambient_temp
        wind_speed = climate_profile.wind_speed

        # Calculate hourly power
        hourly_power = np.zeros(self.HOURS_PER_YEAR)
        cell_temps = np.zeros(self.HOURS_PER_YEAR)
        temp_losses = np.zeros(self.HOURS_PER_YEAR)

        for i in range(self.HOURS_PER_YEAR):
            if ghi[i] > 0:
                # Calculate cell temperature
                cell_temps[i] = self.calculate_cell_temperature(
                    ambient_temp[i], ghi[i], wind_speed[i]
                )

                # Calculate actual power from model
                hourly_power[i] = self.power_model(ghi[i], cell_temps[i])

                # Calculate temperature loss
                temp_factor = 1 + (self.temp_coeff_pmax / 100) * (cell_temps[i] - 25)
                temp_losses[i] = (1 - temp_factor) * 100 if temp_factor < 1 else 0

        # Clip to valid range
        hourly_power = np.clip(hourly_power, 0, self.pmax_stc * 1.1)

        # Calculate annual energy
        annual_energy = np.sum(hourly_power) / 1000  # kWh

        # Calculate CSER (kWh/kWp)
        cser = annual_energy / (self.pmax_stc / 1000)

        # Monthly breakdown
        monthly_energy = []
        monthly_cser = []
        idx = 0
        for hours in self.HOURS_PER_MONTH:
            month_energy = np.sum(hourly_power[idx:idx+hours]) / 1000
            monthly_energy.append(month_energy)
            monthly_cser.append(month_energy / (self.pmax_stc / 1000))
            idx += hours

        # Performance ratio
        annual_irradiation = np.sum(ghi) / 1000  # kWh/m²
        ideal_energy = annual_irradiation * (self.pmax_stc / 1000)  # kWh
        performance_ratio = (annual_energy / ideal_energy * 100
                             if ideal_energy > 0 else 0)

        # Average losses
        daytime_mask = ghi > 0
        avg_temp_loss = np.mean(temp_losses[daytime_mask]) if np.any(daytime_mask) else 0

        # Low irradiance loss (estimated)
        low_irr_mask = (ghi > 0) & (ghi < 400)
        low_irr_loss = (np.sum(low_irr_mask) / np.sum(daytime_mask) * 2
                        if np.any(daytime_mask) else 0)

        # CSER as percentage of STC-equivalent
        stc_equivalent = annual_irradiation  # kWh/kWp if no losses
        relative_to_stc = (cser / stc_equivalent * 100) if stc_equivalent > 0 else 0

        return CSERResult(
            cser=cser,
            annual_energy=annual_energy,
            monthly_energy=monthly_energy,
            monthly_cser=monthly_cser,
            performance_ratio=performance_ratio,
            temperature_loss=avg_temp_loss,
            low_irradiance_loss=low_irr_loss,
            climate_profile=climate_profile.name,
            annual_irradiation=annual_irradiation,
            avg_temperature=climate_profile.avg_temp,
            relative_to_stc=relative_to_stc,
            module_pmax=self.pmax_stc,
            calculation_details={
                "nmot": self.nmot,
                "temp_coeff": self.temp_coeff_pmax,
                "hours_simulated": int(np.sum(daytime_mask)),
            },
        )

    def calculate_all_profiles(
        self,
        profile_manager: Optional[ClimateProfileManager] = None,
    ) -> Dict[str, CSERResult]:
        """
        Calculate CSER for all standard climate profiles.

        Args:
            profile_manager: Optional profile manager (creates new if None)

        Returns:
            Dictionary of {profile_name: CSERResult}
        """
        if profile_manager is None:
            profile_manager = ClimateProfileManager()

        results = {}
        for name in profile_manager.get_standard_profiles():
            profile = profile_manager.get_profile(name)
            if profile:
                results[name] = self.calculate_cser(profile)

        return results

    def compare_modules(
        self,
        modules: Dict[str, Dict[str, Any]],
        climate_profile: ClimateProfile,
    ) -> Dict[str, CSERResult]:
        """
        Compare CSER for multiple modules under same climate.

        Args:
            modules: Dictionary of {name: {pmax_stc, temp_coeff_pmax, nmot}}
            climate_profile: Climate profile to use

        Returns:
            Dictionary of {module_name: CSERResult}
        """
        results = {}

        for name, params in modules.items():
            # Create power model for this module
            pmax = params.get("pmax_stc", 400)
            temp_coeff = params.get("temp_coeff_pmax", -0.35)

            def make_power_model(p, tc):
                def model(g, t):
                    p_g = p * (g / 1000.0)
                    temp_factor = 1 + (tc / 100) * (t - 25)
                    return p_g * temp_factor
                return model

            power_model = make_power_model(pmax, temp_coeff)

            # Calculate CSER
            calculator = CSERCalculator(
                power_model=power_model,
                pmax_stc=pmax,
                temp_coeff_pmax=temp_coeff,
                nmot=params.get("nmot", 45),
            )

            results[name] = calculator.calculate_cser(climate_profile)

        return results

    def sensitivity_analysis(
        self,
        climate_profile: ClimateProfile,
        parameter: str,
        values: List[float],
    ) -> Dict[float, float]:
        """
        Perform sensitivity analysis on a parameter.

        Args:
            climate_profile: Climate profile
            parameter: Parameter to vary ('temp_coeff', 'nmot')
            values: List of parameter values to test

        Returns:
            Dictionary of {parameter_value: cser}
        """
        results = {}
        original_value = getattr(self, parameter.replace('temp_coeff', 'temp_coeff_pmax'))

        for value in values:
            if parameter == 'temp_coeff':
                self.temp_coeff_pmax = value
            elif parameter == 'nmot':
                self.nmot = value

            result = self.calculate_cser(climate_profile)
            results[value] = result.cser

        # Restore original
        setattr(self, parameter.replace('temp_coeff', 'temp_coeff_pmax'), original_value)

        return results

    @staticmethod
    def create_simple_power_model(
        pmax_stc: float,
        temp_coeff: float = -0.35,
    ) -> Callable[[float, float], float]:
        """
        Create a simple power model function.

        Args:
            pmax_stc: Maximum power at STC (W)
            temp_coeff: Temperature coefficient (%/°C)

        Returns:
            Power model function(G, T) -> P
        """
        def power_model(g: float, t: float) -> float:
            if g <= 0:
                return 0.0
            p_g = pmax_stc * (g / 1000.0)
            temp_factor = 1 + (temp_coeff / 100) * (t - 25)
            return p_g * temp_factor

        return power_model
