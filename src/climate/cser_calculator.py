"""
Climate Specific Energy Rating (CSER) Calculator.

Implements the CSER calculation methodology per IEC 61853-3/4 standards.

CSER quantifies how a PV module's energy output in a specific climate
compares to its output at reference (STC) temperature conditions:

    CSER = E_actual / E_reference

Where:
    - E_actual: Energy yield with climate-specific cell temperatures
    - E_reference: Energy yield at constant 25C cell temperature

A CSER < 1.0 indicates thermal losses in hot climates.
A CSER > 1.0 indicates gains in cold climates.

References:
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-4:2018 - Standard reference climatic profiles

Example:
    >>> from src.climate.cser_calculator import CSERCalculator, calculate_cser
    >>> from src.climate.climate_profiles import ClimateType
    >>>
    >>> calculator = CSERCalculator(power_matrix, pmax_stc=400.0)
    >>> result = calculator.calculate_cser(ClimateType.SUBTROPICAL_ARID)
    >>> print(f"CSER: {result.cser:.4f}")
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..calculations.energy_yield import EnergyYieldCalculator, EnergyYieldResult
from ..utils.constants import STC_IRRADIANCE, STC_TEMPERATURE
from ..utils.interpolation import PowerMatrixInterpolator, PowerMatrixSpec, create_power_matrix
from .climate_profiles import (
    ClimateProfile,
    ClimateType,
    get_climate_profile,
    list_available_profiles,
)


@dataclass(frozen=True)
class CSERResult:
    """
    Result container for CSER calculations.

    Attributes:
        cser: Climate Specific Energy Rating [-]
        e_actual: Actual energy yield with temperature effects [kWh]
        e_reference: Reference energy yield at 25C [kWh]
        climate_type: Climate profile used for calculation
        climate_name: Human-readable climate name
        annual_ghi: Annual global horizontal irradiation [kWh/m^2]
        avg_operating_temp: Average cell temperature during operation [C]
        temperature_loss: Relative loss due to temperature [%]
        performance_ratio: Overall performance ratio [-]
        specific_yield: Energy per rated power [kWh/kWp]

    References:
        IEC 61853-3:2018, Section 8 - CSER output requirements
    """

    cser: float
    e_actual: float
    e_reference: float
    climate_type: ClimateType
    climate_name: str
    annual_ghi: float
    avg_operating_temp: float
    temperature_loss: float
    performance_ratio: float
    specific_yield: float

    def __repr__(self) -> str:
        return (
            f"CSERResult(cser={self.cser:.4f}, "
            f"climate='{self.climate_name}', "
            f"E_actual={self.e_actual:.1f} kWh, "
            f"E_ref={self.e_reference:.1f} kWh, "
            f"temp_loss={self.temperature_loss:.1f}%)"
        )

    @property
    def temperature_coefficient_effective(self) -> float:
        """
        Calculate effective temperature coefficient from CSER.

        Returns the equivalent gamma that would produce this CSER
        given the average operating temperature.

        Returns:
            Effective gamma_pmax [1/K]

        References:
            IEC 61853-1:2011, Section 8.2 - Temperature coefficient
        """
        if self.avg_operating_temp == STC_TEMPERATURE:
            return 0.0
        delta_t = self.avg_operating_temp - STC_TEMPERATURE
        if delta_t == 0:
            return 0.0
        return (self.cser - 1.0) / delta_t


@dataclass
class CSERComparison:
    """
    Comparison of CSER across multiple climate profiles.

    Attributes:
        results: Dictionary mapping ClimateType to CSERResult
        best_climate: Climate with highest CSER
        worst_climate: Climate with lowest CSER
        cser_range: (min_cser, max_cser) tuple
        avg_cser: Average CSER across all climates

    References:
        IEC 61853-4:2018, Annex D - Comparative analysis
    """

    results: Dict[ClimateType, CSERResult] = field(default_factory=dict)

    @property
    def best_climate(self) -> ClimateType:
        """Return climate type with highest CSER."""
        return max(self.results, key=lambda c: self.results[c].cser)

    @property
    def worst_climate(self) -> ClimateType:
        """Return climate type with lowest CSER."""
        return min(self.results, key=lambda c: self.results[c].cser)

    @property
    def cser_range(self) -> Tuple[float, float]:
        """Return (min, max) CSER values."""
        cser_values = [r.cser for r in self.results.values()]
        return (min(cser_values), max(cser_values))

    @property
    def avg_cser(self) -> float:
        """Return average CSER across all climates."""
        return float(np.mean([r.cser for r in self.results.values()]))

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns:
            DataFrame with CSER results for each climate

        Requires:
            pandas must be installed
        """
        import pandas as pd

        data = []
        for climate_type, result in self.results.items():
            data.append({
                "Climate": result.climate_name,
                "CSER": result.cser,
                "E_actual (kWh)": result.e_actual,
                "E_reference (kWh)": result.e_reference,
                "Annual GHI (kWh/m2)": result.annual_ghi,
                "Avg Temp (C)": result.avg_operating_temp,
                "Temp Loss (%)": result.temperature_loss,
                "PR": result.performance_ratio,
                "Specific Yield (kWh/kWp)": result.specific_yield,
            })
        return pd.DataFrame(data)


class CSERCalculator:
    """
    IEC 61853-3/4 compliant CSER calculator.

    Calculates the Climate Specific Energy Rating by comparing
    actual energy yield (with temperature effects) to reference
    energy yield (at constant STC temperature).

    Attributes:
        energy_calculator: EnergyYieldCalculator instance
        pmax_stc: Rated power at STC [W]
        gamma_pmax: Power temperature coefficient [1/K]
        nmot: Nominal Module Operating Temperature [C]

    References:
        IEC 61853-3:2018 - Energy rating methodology
        IEC 61853-4:2018 - Standard climate profiles

    Example:
        >>> calc = CSERCalculator(
        ...     irradiance=[200, 400, 600, 800, 1000],
        ...     temperature=[15, 25, 50, 75],
        ...     power_matrix=power_data,
        ...     pmax_stc=400.0
        ... )
        >>> result = calc.calculate_cser(ClimateType.SUBTROPICAL_ARID)
        >>> print(f"CSER: {result.cser:.4f}")
    """

    def __init__(
        self,
        irradiance: ArrayLike,
        temperature: ArrayLike,
        power_matrix: ArrayLike,
        pmax_stc: float,
        nmot: float = 45.0,
        gamma_pmax: float = -0.004,
    ) -> None:
        """
        Initialize the CSER calculator.

        Args:
            irradiance: 1D array of irradiance levels [W/m^2]
            temperature: 1D array of cell temperature levels [C]
            power_matrix: 2D array of power values [W]
            pmax_stc: Rated power at STC [W]
            nmot: Nominal Module Operating Temperature [C]
            gamma_pmax: Power temperature coefficient [1/K]

        References:
            IEC 61853-1:2011, Section 7 - Input data requirements
        """
        self._irradiance = np.asarray(irradiance, dtype=np.float64)
        self._temperature = np.asarray(temperature, dtype=np.float64)
        self._power_matrix = np.asarray(power_matrix, dtype=np.float64)
        self._pmax_stc = pmax_stc
        self._nmot = nmot
        self._gamma_pmax = gamma_pmax

        self._energy_calculator = EnergyYieldCalculator(
            irradiance=self._irradiance,
            temperature=self._temperature,
            power_matrix=self._power_matrix,
            pmax_stc=pmax_stc,
            nmot=nmot,
            gamma_pmax=gamma_pmax,
        )

    @classmethod
    def from_power_matrix_spec(
        cls,
        spec: PowerMatrixSpec,
        pmax_stc: Optional[float] = None,
        nmot: float = 45.0,
        gamma_pmax: float = -0.004,
    ) -> "CSERCalculator":
        """
        Create calculator from PowerMatrixSpec.

        Args:
            spec: PowerMatrixSpec instance
            pmax_stc: Override STC power (default: interpolate)
            nmot: Nominal Module Operating Temperature [C]
            gamma_pmax: Power temperature coefficient [1/K]

        Returns:
            CSERCalculator instance

        Example:
            >>> spec = create_power_matrix(pmax_stc=400.0)
            >>> calc = CSERCalculator.from_power_matrix_spec(spec)
        """
        if pmax_stc is None:
            interp = PowerMatrixInterpolator(
                spec.irradiance_levels,
                spec.temperature_levels,
                spec.power_values,
            )
            pmax_stc = interp.power_at_stc

        return cls(
            irradiance=spec.irradiance_levels,
            temperature=spec.temperature_levels,
            power_matrix=spec.power_values,
            pmax_stc=pmax_stc,
            nmot=nmot,
            gamma_pmax=gamma_pmax,
        )

    @classmethod
    def from_stc_parameters(
        cls,
        pmax_stc: float,
        gamma_pmax: float = -0.004,
        nmot: float = 45.0,
        efficiency_ratio: float = 0.95,
    ) -> "CSERCalculator":
        """
        Create calculator from basic STC parameters.

        Generates synthetic power matrix from STC power and
        temperature coefficient for quick CSER estimation.

        Args:
            pmax_stc: Rated power at STC [W]
            gamma_pmax: Power temperature coefficient [1/K]
            nmot: Nominal Module Operating Temperature [C]
            efficiency_ratio: Low-light efficiency ratio

        Returns:
            CSERCalculator instance

        References:
            IEC 61853-1:2011, Section 8 - Simplified model

        Example:
            >>> calc = CSERCalculator.from_stc_parameters(
            ...     pmax_stc=400.0,
            ...     gamma_pmax=-0.0038
            ... )
        """
        spec = create_power_matrix(
            pmax_stc=pmax_stc,
            gamma_pmax=gamma_pmax,
            efficiency_ratio=efficiency_ratio,
        )
        return cls.from_power_matrix_spec(
            spec,
            pmax_stc=pmax_stc,
            nmot=nmot,
            gamma_pmax=gamma_pmax,
        )

    @functools.lru_cache(maxsize=16)
    def calculate_cser(
        self,
        climate: Union[ClimateType, ClimateProfile],
    ) -> CSERResult:
        """
        Calculate CSER for a specific climate.

        Computes the Climate Specific Energy Rating as the ratio
        of actual energy yield to reference energy yield.

        Args:
            climate: ClimateType enum or ClimateProfile instance

        Returns:
            CSERResult with detailed calculation outputs

        References:
            IEC 61853-3:2018, Equation 5 - CSER definition:
            CSER = E_actual / E_reference

        Example:
            >>> result = calc.calculate_cser(ClimateType.SUBTROPICAL_ARID)
            >>> print(f"CSER: {result.cser:.4f}")
            >>> print(f"Temperature loss: {result.temperature_loss:.1f}%")
        """
        # Get climate profile
        if isinstance(climate, ClimateType):
            profile = get_climate_profile(climate)
        else:
            profile = climate

        # Calculate actual energy (with temperature effects)
        yield_result: EnergyYieldResult = self._energy_calculator.calculate_annual_energy(
            hourly_ghi=profile.hourly_ghi,
            hourly_ambient_temp=profile.hourly_ambient_temp,
            hourly_wind_speed=profile.hourly_wind_speed,
        )
        e_actual = yield_result.annual_energy

        # Calculate reference energy (at constant 25C)
        e_reference = self._energy_calculator.calculate_reference_energy(
            hourly_ghi=profile.hourly_ghi,
        )

        # CSER calculation
        if e_reference > 0:
            cser = e_actual / e_reference
        else:
            cser = 1.0

        # Calculate average operating temperature
        mask = profile.hourly_ghi > 0
        avg_irradiance = float(np.mean(profile.hourly_ghi[mask])) if np.any(mask) else 0
        avg_ambient = float(np.mean(profile.hourly_ambient_temp[mask])) if np.any(mask) else 25.0
        avg_wind = float(np.mean(profile.hourly_wind_speed[mask])) if np.any(mask) else 1.0

        avg_cell_temp = self._energy_calculator.calculate_cell_temperature(
            avg_irradiance,
            avg_ambient,
            avg_wind,
        )

        # Temperature loss percentage
        temperature_loss = (1.0 - cser) * 100.0

        return CSERResult(
            cser=cser,
            e_actual=e_actual,
            e_reference=e_reference,
            climate_type=profile.climate_type,
            climate_name=profile.name,
            annual_ghi=profile.annual_global_irradiation,
            avg_operating_temp=avg_cell_temp,
            temperature_loss=temperature_loss,
            performance_ratio=yield_result.performance_ratio,
            specific_yield=yield_result.specific_yield,
        )

    def calculate_all_climates(self) -> CSERComparison:
        """
        Calculate CSER for all IEC 61853-4 standard climates.

        Returns:
            CSERComparison with results for all 6 standard climates

        References:
            IEC 61853-4:2018, Table 1 - Standard climate profiles

        Example:
            >>> comparison = calc.calculate_all_climates()
            >>> print(f"Best: {comparison.best_climate.value}")
            >>> print(f"Worst: {comparison.worst_climate.value}")
            >>> print(f"Range: {comparison.cser_range}")
        """
        comparison = CSERComparison()

        for climate_type in ClimateType:
            result = self.calculate_cser(climate_type)
            comparison.results[climate_type] = result

        return comparison

    def sensitivity_analysis(
        self,
        climate: Union[ClimateType, ClimateProfile],
        gamma_range: Optional[Tuple[float, float]] = None,
        nmot_range: Optional[Tuple[float, float]] = None,
        n_points: int = 10,
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Perform sensitivity analysis on CSER.

        Analyzes how CSER varies with temperature coefficient
        and/or NMOT parameters.

        Args:
            climate: Climate for analysis
            gamma_range: (min, max) gamma_pmax values [1/K]
            nmot_range: (min, max) NMOT values [C]
            n_points: Number of points in each range

        Returns:
            Dictionary with sensitivity data arrays

        References:
            IEC 61853-3:2018, Annex E - Sensitivity analysis

        Example:
            >>> sensitivity = calc.sensitivity_analysis(
            ...     ClimateType.SUBTROPICAL_ARID,
            ...     gamma_range=(-0.005, -0.002)
            ... )
        """
        if gamma_range is None:
            gamma_range = (-0.005, -0.002)
        if nmot_range is None:
            nmot_range = (40.0, 55.0)

        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
        nmot_values = np.linspace(nmot_range[0], nmot_range[1], n_points)

        # Get profile
        if isinstance(climate, ClimateType):
            profile = get_climate_profile(climate)
        else:
            profile = climate

        # Sensitivity to gamma
        cser_vs_gamma = np.zeros(n_points)
        for i, gamma in enumerate(gamma_values):
            temp_calc = CSERCalculator(
                irradiance=self._irradiance,
                temperature=self._temperature,
                power_matrix=self._power_matrix,
                pmax_stc=self._pmax_stc,
                nmot=self._nmot,
                gamma_pmax=gamma,
            )
            result = temp_calc.calculate_cser(profile)
            cser_vs_gamma[i] = result.cser

        # Sensitivity to NMOT
        cser_vs_nmot = np.zeros(n_points)
        for i, nmot in enumerate(nmot_values):
            temp_calc = CSERCalculator(
                irradiance=self._irradiance,
                temperature=self._temperature,
                power_matrix=self._power_matrix,
                pmax_stc=self._pmax_stc,
                nmot=nmot,
                gamma_pmax=self._gamma_pmax,
            )
            result = temp_calc.calculate_cser(profile)
            cser_vs_nmot[i] = result.cser

        return {
            "gamma_values": gamma_values,
            "cser_vs_gamma": cser_vs_gamma,
            "nmot_values": nmot_values,
            "cser_vs_nmot": cser_vs_nmot,
        }

    @property
    def pmax_stc(self) -> float:
        """Return rated STC power [W]."""
        return self._pmax_stc

    @property
    def gamma_pmax(self) -> float:
        """Return power temperature coefficient [1/K]."""
        return self._gamma_pmax

    @property
    def nmot(self) -> float:
        """Return Nominal Module Operating Temperature [C]."""
        return self._nmot


def calculate_cser(
    power_matrix_spec: PowerMatrixSpec,
    climate: Union[ClimateType, ClimateProfile],
    pmax_stc: Optional[float] = None,
    nmot: float = 45.0,
    gamma_pmax: float = -0.004,
) -> CSERResult:
    """
    Calculate CSER for a PV module in a specific climate.

    Convenience function for single CSER calculation.

    CSER = E_actual / E_reference

    Where:
        - E_actual: Annual energy with climate-specific temperatures
        - E_reference: Annual energy at constant 25C

    Args:
        power_matrix_spec: PowerMatrixSpec with module data
        climate: ClimateType or ClimateProfile
        pmax_stc: Rated STC power [W] (default: from matrix)
        nmot: Nominal Module Operating Temperature [C]
        gamma_pmax: Power temperature coefficient [1/K]

    Returns:
        CSERResult with detailed calculation outputs

    References:
        IEC 61853-3:2018 - Energy rating of PV modules
        IEC 61853-4:2018 - Standard reference climatic profiles

    Example:
        >>> from src.utils.interpolation import create_power_matrix
        >>> spec = create_power_matrix(pmax_stc=400.0, gamma_pmax=-0.0038)
        >>> result = calculate_cser(spec, ClimateType.SUBTROPICAL_ARID)
        >>> print(f"CSER: {result.cser:.4f}")
        CSER: 0.9523
    """
    calculator = CSERCalculator.from_power_matrix_spec(
        spec=power_matrix_spec,
        pmax_stc=pmax_stc,
        nmot=nmot,
        gamma_pmax=gamma_pmax,
    )
    return calculator.calculate_cser(climate)


def calculate_cser_from_stc(
    pmax_stc: float,
    climate: Union[ClimateType, ClimateProfile],
    gamma_pmax: float = -0.004,
    nmot: float = 45.0,
) -> CSERResult:
    """
    Calculate CSER from basic STC parameters.

    Simplified interface for CSER estimation when only STC
    power and temperature coefficient are known.

    Args:
        pmax_stc: Rated power at STC [W]
        climate: ClimateType or ClimateProfile
        gamma_pmax: Power temperature coefficient [1/K]
        nmot: Nominal Module Operating Temperature [C]

    Returns:
        CSERResult with calculation outputs

    References:
        IEC 61853-3:2018, Annex F - Simplified CSER estimation

    Example:
        >>> result = calculate_cser_from_stc(
        ...     pmax_stc=400.0,
        ...     climate=ClimateType.HIGH_ALTITUDE,
        ...     gamma_pmax=-0.0032
        ... )
        >>> print(f"CSER: {result.cser:.4f}")
    """
    calculator = CSERCalculator.from_stc_parameters(
        pmax_stc=pmax_stc,
        gamma_pmax=gamma_pmax,
        nmot=nmot,
    )
    return calculator.calculate_cser(climate)
