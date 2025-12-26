"""
Energy Yield Calculations for IEC 61853-3 Energy Rating.

Implements the energy yield calculation methodology for computing
E_actual (climate-specific energy) and E_reference (STC-normalized energy)
required for CSER determination.

The core calculation follows IEC 61853-3:
    E = sum(P(G, T) * dt)

Where:
    - P(G, T) is power interpolated from the power matrix at irradiance G
      and module temperature T
    - dt is the time step (typically 1 hour)

References:
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-1:2011 - Power matrix definition

Example:
    >>> from src.calculations.energy_yield import EnergyYieldCalculator
    >>> from src.climate.climate_profiles import get_climate_profile, ClimateType
    >>>
    >>> calculator = EnergyYieldCalculator(power_matrix, pmax_stc=400.0)
    >>> profile = get_climate_profile(ClimateType.SUBTROPICAL_ARID)
    >>> e_actual = calculator.calculate_annual_energy(profile)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.constants import (
    HOURS_PER_YEAR,
    MAX_CELL_TEMPERATURE,
    MAX_IRRADIANCE,
    MIN_CELL_TEMPERATURE,
    MIN_IRRADIANCE_THRESHOLD,
    NMOT_AMBIENT_TEMPERATURE,
    NMOT_IRRADIANCE,
    NMOT_WIND_SPEED,
    STC_IRRADIANCE,
    STC_TEMPERATURE,
)
from ..utils.interpolation import PowerMatrixInterpolator, PowerMatrixSpec


@dataclass(frozen=True)
class EnergyYieldResult:
    """
    Result container for energy yield calculations.

    Attributes:
        annual_energy: Total annual energy yield [kWh]
        monthly_energy: Monthly energy yields [kWh], shape (12,)
        specific_yield: Energy per rated power [kWh/kWp]
        performance_ratio: Actual vs theoretical performance [-]
        hours_of_operation: Hours with positive power output [h]
        peak_power_output: Maximum hourly power [W]
        capacity_factor: Average power / rated power [-]

    References:
        IEC 61853-3:2018, Section 7 - Energy yield output parameters
    """

    annual_energy: float
    monthly_energy: NDArray[np.floating]
    specific_yield: float
    performance_ratio: float
    hours_of_operation: int
    peak_power_output: float
    capacity_factor: float

    def __repr__(self) -> str:
        return (
            f"EnergyYieldResult(annual_energy={self.annual_energy:.1f} kWh, "
            f"specific_yield={self.specific_yield:.1f} kWh/kWp, "
            f"PR={self.performance_ratio:.3f})"
        )


class EnergyYieldCalculator:
    """
    IEC 61853-3 compliant energy yield calculator.

    Calculates annual energy yield using hourly climate data and
    interpolated power from the module's power matrix.

    Attributes:
        power_interpolator: PowerMatrixInterpolator instance
        pmax_stc: Rated power at STC [W]
        nmot: Nominal Module Operating Temperature [C]
        gamma_pmax: Power temperature coefficient [1/K]

    References:
        IEC 61853-3:2018, Section 6 - Energy rating calculation procedure

    Example:
        >>> calc = EnergyYieldCalculator(
        ...     irradiance=[200, 400, 600, 800, 1000],
        ...     temperature=[15, 25, 50, 75],
        ...     power_matrix=power_data,
        ...     pmax_stc=400.0
        ... )
        >>> result = calc.calculate_annual_energy(climate_profile)
    """

    def __init__(
        self,
        irradiance: ArrayLike,
        temperature: ArrayLike,
        power_matrix: ArrayLike,
        pmax_stc: float,
        nmot: float = 45.0,
        gamma_pmax: float = -0.004,
        u_const: float = 29.0,
        u_wind: float = 0.0,
    ) -> None:
        """
        Initialize the energy yield calculator.

        Args:
            irradiance: 1D array of irradiance levels [W/m^2]
            temperature: 1D array of cell temperature levels [C]
            power_matrix: 2D array of power values [W], shape (n_irr, n_temp)
            pmax_stc: Rated power at STC (1000 W/m^2, 25C) [W]
            nmot: Nominal Module Operating Temperature [C]
            gamma_pmax: Power temperature coefficient [1/K]
            u_const: Constant heat transfer coefficient [W/m^2K]
            u_wind: Wind-dependent heat transfer coefficient [W/m^2K/(m/s)]

        References:
            IEC 61853-2:2016, Section 7.3 - Temperature model parameters
        """
        self._interpolator = PowerMatrixInterpolator(
            irradiance=irradiance,
            temperature=temperature,
            power=power_matrix,
            method="linear",
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )
        self._pmax_stc = pmax_stc
        self._nmot = nmot
        self._gamma_pmax = gamma_pmax
        self._u_const = u_const
        self._u_wind = u_wind

        # Cache STC power from matrix
        self._p_stc_measured = self._interpolator.power_at_stc

    @classmethod
    def from_power_matrix_spec(
        cls,
        spec: PowerMatrixSpec,
        pmax_stc: Optional[float] = None,
        **kwargs,
    ) -> "EnergyYieldCalculator":
        """
        Create calculator from PowerMatrixSpec.

        Args:
            spec: PowerMatrixSpec instance
            pmax_stc: Override STC power (default: interpolate from matrix)
            **kwargs: Additional arguments passed to __init__

        Returns:
            EnergyYieldCalculator instance
        """
        interp = PowerMatrixInterpolator(
            spec.irradiance_levels,
            spec.temperature_levels,
            spec.power_values,
        )
        if pmax_stc is None:
            pmax_stc = interp.power_at_stc

        return cls(
            irradiance=spec.irradiance_levels,
            temperature=spec.temperature_levels,
            power_matrix=spec.power_values,
            pmax_stc=pmax_stc,
            **kwargs,
        )

    @functools.lru_cache(maxsize=32)
    def calculate_cell_temperature(
        self,
        irradiance: float,
        ambient_temp: float,
        wind_speed: float = 1.0,
    ) -> float:
        """
        Calculate module cell temperature using IEC 61853-2 model.

        Uses the steady-state thermal model:
            T_cell = T_amb + G * (NMOT - 20) / 800 * (1 - eta)

        Or the Faiman model if wind coefficients are provided.

        Args:
            irradiance: Plane-of-array irradiance [W/m^2]
            ambient_temp: Ambient temperature [C]
            wind_speed: Wind speed at module height [m/s]

        Returns:
            Estimated cell temperature [C]

        References:
            IEC 61853-2:2016, Section 7 - Cell temperature determination
        """
        if irradiance <= 0:
            return ambient_temp

        # Simplified NMOT model (IEC 61853-2, Eq. 3)
        # T_cell = T_amb + (NMOT - 20) * G / 800
        delta_t_nmot = self._nmot - NMOT_AMBIENT_TEMPERATURE
        t_cell = ambient_temp + delta_t_nmot * (irradiance / NMOT_IRRADIANCE)

        # Apply wind correction if wind coefficient is set
        if self._u_wind > 0 and wind_speed > NMOT_WIND_SPEED:
            wind_factor = 1.0 - 0.05 * (wind_speed - NMOT_WIND_SPEED)
            wind_factor = max(0.7, min(1.0, wind_factor))
            t_cell = ambient_temp + (t_cell - ambient_temp) * wind_factor

        # Clamp to valid range
        return float(np.clip(t_cell, MIN_CELL_TEMPERATURE, MAX_CELL_TEMPERATURE))

    def calculate_hourly_power(
        self,
        irradiance: float,
        cell_temperature: float,
    ) -> float:
        """
        Calculate instantaneous power at given conditions.

        Interpolates power from the power matrix at the specified
        irradiance and cell temperature.

        Args:
            irradiance: Plane-of-array irradiance [W/m^2]
            cell_temperature: Module cell temperature [C]

        Returns:
            DC power output [W]

        References:
            IEC 61853-3:2018, Section 6.2 - Power interpolation
        """
        if irradiance <= MIN_IRRADIANCE_THRESHOLD:
            return 0.0

        # Clamp inputs to valid ranges
        irradiance = float(np.clip(irradiance, 0, MAX_IRRADIANCE))
        cell_temperature = float(
            np.clip(cell_temperature, MIN_CELL_TEMPERATURE, MAX_CELL_TEMPERATURE)
        )

        power = self._interpolator(irradiance, cell_temperature)
        return max(0.0, float(power))

    def calculate_annual_energy(
        self,
        hourly_ghi: ArrayLike,
        hourly_ambient_temp: ArrayLike,
        hourly_wind_speed: Optional[ArrayLike] = None,
    ) -> EnergyYieldResult:
        """
        Calculate annual energy yield from hourly climate data.

        Computes energy yield by summing hourly power outputs calculated
        from the power matrix using interpolated values at each hour's
        irradiance and temperature conditions.

        Args:
            hourly_ghi: 8760 hourly GHI values [W/m^2]
            hourly_ambient_temp: 8760 hourly ambient temperatures [C]
            hourly_wind_speed: 8760 hourly wind speeds [m/s], default 1.0

        Returns:
            EnergyYieldResult with annual and monthly energy statistics

        References:
            IEC 61853-3:2018, Section 6.4 - Annual energy calculation

        Example:
            >>> result = calc.calculate_annual_energy(
            ...     hourly_ghi=profile.hourly_ghi,
            ...     hourly_ambient_temp=profile.hourly_ambient_temp,
            ...     hourly_wind_speed=profile.hourly_wind_speed
            ... )
            >>> print(f"Annual yield: {result.annual_energy:.1f} kWh")
        """
        ghi = np.asarray(hourly_ghi, dtype=np.float64)
        temp = np.asarray(hourly_ambient_temp, dtype=np.float64)

        if hourly_wind_speed is None:
            wind = np.ones(HOURS_PER_YEAR, dtype=np.float64)
        else:
            wind = np.asarray(hourly_wind_speed, dtype=np.float64)

        if len(ghi) != HOURS_PER_YEAR:
            raise ValueError(f"Expected {HOURS_PER_YEAR} hourly values, got {len(ghi)}")

        # Calculate hourly power output
        hourly_power = np.zeros(HOURS_PER_YEAR, dtype=np.float64)

        for hour in range(HOURS_PER_YEAR):
            if ghi[hour] > MIN_IRRADIANCE_THRESHOLD:
                t_cell = self.calculate_cell_temperature(
                    float(ghi[hour]),
                    float(temp[hour]),
                    float(wind[hour]),
                )
                hourly_power[hour] = self.calculate_hourly_power(
                    float(ghi[hour]),
                    t_cell,
                )

        # Annual energy in kWh (power in W, 1 hour time step)
        annual_energy = np.sum(hourly_power) / 1000.0

        # Monthly breakdown
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        monthly_energy = np.zeros(12, dtype=np.float64)
        hour_idx = 0
        for month, days in enumerate(days_per_month):
            hours_in_month = days * 24
            monthly_energy[month] = (
                np.sum(hourly_power[hour_idx : hour_idx + hours_in_month]) / 1000.0
            )
            hour_idx += hours_in_month

        # Statistics
        hours_of_operation = int(np.sum(hourly_power > 0))
        peak_power = float(np.max(hourly_power))

        # Specific yield [kWh/kWp]
        specific_yield = annual_energy / (self._pmax_stc / 1000.0)

        # Performance ratio (simplified)
        # PR = E_actual / (G_total * Pmax / G_stc)
        annual_ghi_kwh = np.sum(ghi) / 1000.0
        theoretical_energy = annual_ghi_kwh * (self._pmax_stc / STC_IRRADIANCE) / 1000.0
        performance_ratio = (
            annual_energy / theoretical_energy if theoretical_energy > 0 else 0.0
        )

        # Capacity factor
        capacity_factor = annual_energy / (self._pmax_stc / 1000.0 * HOURS_PER_YEAR)

        return EnergyYieldResult(
            annual_energy=annual_energy,
            monthly_energy=monthly_energy,
            specific_yield=specific_yield,
            performance_ratio=performance_ratio,
            hours_of_operation=hours_of_operation,
            peak_power_output=peak_power,
            capacity_factor=capacity_factor,
        )

    def calculate_reference_energy(
        self,
        hourly_ghi: ArrayLike,
    ) -> float:
        """
        Calculate reference energy at STC temperature.

        Computes the theoretical energy yield assuming constant cell
        temperature of 25C (STC), used as denominator in CSER.

        E_ref = sum(P(G, 25C) * dt)

        Args:
            hourly_ghi: 8760 hourly GHI values [W/m^2]

        Returns:
            Reference energy yield [kWh]

        References:
            IEC 61853-3:2018, Section 6.5 - Reference energy definition
        """
        ghi = np.asarray(hourly_ghi, dtype=np.float64)

        if len(ghi) != HOURS_PER_YEAR:
            raise ValueError(f"Expected {HOURS_PER_YEAR} hourly values, got {len(ghi)}")

        # Calculate power at each irradiance level, fixed T=25C
        hourly_power = np.zeros(HOURS_PER_YEAR, dtype=np.float64)

        for hour in range(HOURS_PER_YEAR):
            if ghi[hour] > MIN_IRRADIANCE_THRESHOLD:
                hourly_power[hour] = self.calculate_hourly_power(
                    float(ghi[hour]),
                    STC_TEMPERATURE,  # Fixed at 25C
                )

        return float(np.sum(hourly_power) / 1000.0)


def calculate_actual_energy(
    power_matrix_spec: PowerMatrixSpec,
    hourly_ghi: ArrayLike,
    hourly_ambient_temp: ArrayLike,
    hourly_wind_speed: Optional[ArrayLike] = None,
    nmot: float = 45.0,
) -> float:
    """
    Calculate actual energy yield for CSER numerator.

    Convenience function for calculating E_actual with climate-specific
    cell temperatures.

    Args:
        power_matrix_spec: PowerMatrixSpec with module power data
        hourly_ghi: 8760 hourly GHI values [W/m^2]
        hourly_ambient_temp: 8760 hourly ambient temperatures [C]
        hourly_wind_speed: 8760 hourly wind speeds [m/s]
        nmot: Nominal Module Operating Temperature [C]

    Returns:
        Actual annual energy yield [kWh]

    References:
        IEC 61853-3:2018, Equation 1 - E_actual definition

    Example:
        >>> e_actual = calculate_actual_energy(
        ...     power_matrix_spec,
        ...     profile.hourly_ghi,
        ...     profile.hourly_ambient_temp
        ... )
    """
    calculator = EnergyYieldCalculator.from_power_matrix_spec(
        power_matrix_spec,
        nmot=nmot,
    )
    result = calculator.calculate_annual_energy(
        hourly_ghi,
        hourly_ambient_temp,
        hourly_wind_speed,
    )
    return result.annual_energy


def calculate_reference_energy(
    power_matrix_spec: PowerMatrixSpec,
    hourly_ghi: ArrayLike,
) -> float:
    """
    Calculate reference energy yield for CSER denominator.

    Convenience function for calculating E_reference at constant
    STC temperature (25C).

    Args:
        power_matrix_spec: PowerMatrixSpec with module power data
        hourly_ghi: 8760 hourly GHI values [W/m^2]

    Returns:
        Reference annual energy yield [kWh]

    References:
        IEC 61853-3:2018, Equation 2 - E_reference definition

    Example:
        >>> e_ref = calculate_reference_energy(
        ...     power_matrix_spec,
        ...     profile.hourly_ghi
        ... )
    """
    calculator = EnergyYieldCalculator.from_power_matrix_spec(power_matrix_spec)
    return calculator.calculate_reference_energy(hourly_ghi)


def calculate_hourly_power(
    irradiance: float,
    cell_temperature: float,
    power_matrix_spec: PowerMatrixSpec,
) -> float:
    """
    Calculate instantaneous power at given operating conditions.

    Standalone function for single-point power calculation.

    Args:
        irradiance: Plane-of-array irradiance [W/m^2]
        cell_temperature: Module cell temperature [C]
        power_matrix_spec: PowerMatrixSpec with module power data

    Returns:
        DC power output [W]

    References:
        IEC 61853-1:2011, Section 7 - Power rating at conditions

    Example:
        >>> power = calculate_hourly_power(800.0, 45.0, power_matrix_spec)
    """
    interpolator = PowerMatrixInterpolator(
        power_matrix_spec.irradiance_levels,
        power_matrix_spec.temperature_levels,
        power_matrix_spec.power_values,
    )
    return max(0.0, float(interpolator(irradiance, cell_temperature)))
