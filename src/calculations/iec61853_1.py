"""
IEC 61853-1 Core Calculations Module.

This module implements the core calculation methods for PV module performance
characterization according to IEC 61853-1 standard.

Includes:
- Temperature coefficient calculations (alpha, beta, gamma)
- Interpolation methods (linear, logarithmic, polynomial)
- Power matrix operations
- CSER rating calculations

Reference:
    IEC 61853-1:2011 - Irradiance and temperature performance measurements
    IEC 61853-3:2018 - Energy rating of PV modules
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator
from scipy import interpolate
from scipy.optimize import curve_fit

from src.utils.constants import (
    IRRADIANCES,
    TEMPERATURES,
    IRRADIANCE_MIN,
    IRRADIANCE_MAX,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    THERMAL_VOLTAGE_25C,
    IDEALITY_FACTOR_TYPICAL,
    STC,
)


# =============================================================================
# Pydantic Models for Input Validation
# =============================================================================

class PowerMatrixInput(BaseModel):
    """
    Validated power matrix input data.

    Attributes:
        irradiance: Irradiance value in W/m²
        temperature: Cell temperature in °C
        isc: Short-circuit current in A
        voc: Open-circuit voltage in V
        pmax: Maximum power in W
        vmp: Voltage at maximum power in V
        imp: Current at maximum power in A
        fill_factor: Fill factor (0-1)
    """
    irradiance: float = Field(..., ge=IRRADIANCE_MIN, le=IRRADIANCE_MAX)
    temperature: float = Field(..., ge=TEMPERATURE_MIN, le=TEMPERATURE_MAX)
    isc: float = Field(..., ge=0)
    voc: float = Field(..., ge=0)
    pmax: float = Field(..., ge=0)
    vmp: float = Field(..., ge=0)
    imp: float = Field(..., ge=0)
    fill_factor: Optional[float] = Field(None, ge=0, le=1)

    @field_validator('fill_factor', mode='before')
    @classmethod
    def calculate_fill_factor(cls, v, info):
        """Calculate fill factor if not provided."""
        if v is None:
            data = info.data
            if data.get('isc') and data.get('voc') and data.get('pmax'):
                isc = data['isc']
                voc = data['voc']
                pmax = data['pmax']
                if isc > 0 and voc > 0:
                    return pmax / (isc * voc)
        return v


class ModuleParameters(BaseModel):
    """
    PV module parameters at STC.

    Attributes:
        pmax_stc: Maximum power at STC in W
        voc_stc: Open-circuit voltage at STC in V
        isc_stc: Short-circuit current at STC in A
        vmp_stc: Voltage at maximum power at STC in V
        imp_stc: Current at maximum power at STC in A
        alpha_isc: Temperature coefficient of Isc in %/°C
        beta_voc: Temperature coefficient of Voc in %/°C
        gamma_pmax: Temperature coefficient of Pmax in %/°C
        cells_in_series: Number of cells in series
        area: Module area in m²
    """
    pmax_stc: float = Field(..., gt=0)
    voc_stc: float = Field(..., gt=0)
    isc_stc: float = Field(..., gt=0)
    vmp_stc: float = Field(..., gt=0)
    imp_stc: float = Field(..., gt=0)
    alpha_isc: float = Field(..., ge=-1, le=1)
    beta_voc: float = Field(..., ge=-1, le=1)
    gamma_pmax: float = Field(..., ge=-1, le=1)
    cells_in_series: int = Field(60, gt=0)
    area: float = Field(..., gt=0)


@dataclass
class TemperatureCoefficients:
    """
    Temperature coefficients calculated from power matrix.

    Attributes:
        alpha_isc: Isc temperature coefficient (%/°C)
        beta_voc: Voc temperature coefficient (%/°C)
        gamma_pmax: Pmax temperature coefficient (%/°C)
        alpha_abs: Absolute Isc coefficient (A/°C)
        beta_abs: Absolute Voc coefficient (V/°C)
        gamma_abs: Absolute Pmax coefficient (W/°C)
    """
    alpha_isc: float
    beta_voc: float
    gamma_pmax: float
    alpha_abs: float
    beta_abs: float
    gamma_abs: float


@dataclass
class InterpolationResult:
    """
    Result of power interpolation at specific conditions.

    Attributes:
        irradiance: Target irradiance (W/m²)
        temperature: Target temperature (°C)
        pmax: Interpolated maximum power (W)
        isc: Interpolated short-circuit current (A)
        voc: Interpolated open-circuit voltage (V)
        efficiency: Calculated efficiency (%)
        method: Interpolation method used
    """
    irradiance: float
    temperature: float
    pmax: float
    isc: float
    voc: float
    efficiency: float
    method: str


# =============================================================================
# Temperature Coefficient Calculations
# =============================================================================

def calculate_temperature_coefficients(
    power_matrix: NDArray[np.float64],
    isc_matrix: NDArray[np.float64],
    voc_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    reference_irradiance: float = 1000.0,
    reference_temperature: float = 25.0
) -> TemperatureCoefficients:
    """
    Calculate temperature coefficients from power matrix data.

    Per IEC 61853-1 Clause 9, temperature coefficients are determined
    by linear regression of the parameter vs temperature at constant
    irradiance (typically 1000 W/m²).

    Args:
        power_matrix: Pmax values array of shape (n_irradiances, n_temperatures)
        isc_matrix: Isc values array of shape (n_irradiances, n_temperatures)
        voc_matrix: Voc values array of shape (n_irradiances, n_temperatures)
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        reference_irradiance: Reference irradiance for coefficient calculation
        reference_temperature: Reference temperature (typically 25°C)

    Returns:
        TemperatureCoefficients dataclass with all coefficients

    Reference:
        IEC 61853-1:2011 Clause 9
    """
    # Find index closest to reference irradiance (1000 W/m²)
    g_idx = np.argmin(np.abs(np.array(irradiances) - reference_irradiance))
    t_ref_idx = np.argmin(np.abs(np.array(temperatures) - reference_temperature))

    temps = np.array(temperatures, dtype=np.float64)

    # Get values at reference irradiance
    pmax_at_g = power_matrix[g_idx, :]
    isc_at_g = isc_matrix[g_idx, :]
    voc_at_g = voc_matrix[g_idx, :]

    # Reference values at STC
    pmax_ref = pmax_at_g[t_ref_idx]
    isc_ref = isc_at_g[t_ref_idx]
    voc_ref = voc_at_g[t_ref_idx]

    # Linear regression for each parameter
    # Isc coefficient (alpha)
    valid_isc = ~np.isnan(isc_at_g)
    if np.sum(valid_isc) >= 2:
        slope_isc, _ = np.polyfit(temps[valid_isc], isc_at_g[valid_isc], 1)
        alpha_abs = slope_isc
        alpha_isc = (slope_isc / isc_ref) * 100 if isc_ref > 0 else 0.0
    else:
        alpha_abs = 0.0
        alpha_isc = 0.0

    # Voc coefficient (beta)
    valid_voc = ~np.isnan(voc_at_g)
    if np.sum(valid_voc) >= 2:
        slope_voc, _ = np.polyfit(temps[valid_voc], voc_at_g[valid_voc], 1)
        beta_abs = slope_voc
        beta_voc = (slope_voc / voc_ref) * 100 if voc_ref > 0 else 0.0
    else:
        beta_abs = 0.0
        beta_voc = 0.0

    # Pmax coefficient (gamma)
    valid_pmax = ~np.isnan(pmax_at_g)
    if np.sum(valid_pmax) >= 2:
        slope_pmax, _ = np.polyfit(temps[valid_pmax], pmax_at_g[valid_pmax], 1)
        gamma_abs = slope_pmax
        gamma_pmax = (slope_pmax / pmax_ref) * 100 if pmax_ref > 0 else 0.0
    else:
        gamma_abs = 0.0
        gamma_pmax = 0.0

    return TemperatureCoefficients(
        alpha_isc=alpha_isc,
        beta_voc=beta_voc,
        gamma_pmax=gamma_pmax,
        alpha_abs=alpha_abs,
        beta_abs=beta_abs,
        gamma_abs=gamma_abs
    )


# =============================================================================
# Interpolation Functions
# =============================================================================

def interpolate_isc_temperature(
    isc_ref: float,
    temp_ref: float,
    temp_target: float,
    alpha: float
) -> float:
    """
    Interpolate short-circuit current for temperature using linear relationship.

    Per IEC 61853-1 Clause 7.2, Isc varies linearly with temperature:
    Isc(T) = Isc_ref * [1 + alpha * (T - T_ref)]

    Args:
        isc_ref: Reference Isc at temp_ref in Amperes
        temp_ref: Reference temperature in °C
        temp_target: Target temperature in °C
        alpha: Temperature coefficient in %/°C

    Returns:
        Interpolated Isc in Amperes

    Reference:
        IEC 61853-1:2011 Clause 7.2
    """
    alpha_decimal = alpha / 100.0  # Convert from %/°C to 1/°C
    return isc_ref * (1 + alpha_decimal * (temp_target - temp_ref))


def interpolate_isc_irradiance(
    isc_ref: float,
    g_ref: float,
    g_target: float
) -> float:
    """
    Interpolate short-circuit current for irradiance using linear relationship.

    Per IEC 61853-1 Clause 7.1, Isc varies linearly with irradiance:
    Isc(G) = Isc_ref * (G / G_ref)

    Args:
        isc_ref: Reference Isc at g_ref in Amperes
        g_ref: Reference irradiance in W/m²
        g_target: Target irradiance in W/m²

    Returns:
        Interpolated Isc in Amperes

    Raises:
        ValueError: If g_ref <= 0

    Reference:
        IEC 61853-1:2011 Clause 7.1
    """
    if g_ref <= 0:
        raise ValueError("Reference irradiance must be positive")
    if g_target < 0:
        raise ValueError("Target irradiance cannot be negative")

    return isc_ref * (g_target / g_ref)


def interpolate_voc_irradiance(
    voc_ref: float,
    g_ref: float,
    g_target: float,
    n_cells: int = 60,
    temp_kelvin: float = 298.15,
    ideality_factor: float = IDEALITY_FACTOR_TYPICAL
) -> float:
    """
    Interpolate open-circuit voltage for irradiance using logarithmic relationship.

    Per IEC 61853-1 Clause 7.3, Voc varies logarithmically with irradiance:
    Voc(G) = Voc_ref + n * N_s * (kT/q) * ln(G / G_ref)

    Where:
        n = ideality factor
        N_s = number of cells in series
        k = Boltzmann constant
        T = temperature in Kelvin
        q = elementary charge

    Args:
        voc_ref: Reference Voc at g_ref in Volts
        g_ref: Reference irradiance in W/m²
        g_target: Target irradiance in W/m²
        n_cells: Number of cells in series
        temp_kelvin: Temperature in Kelvin
        ideality_factor: Diode ideality factor (typically 1.0-2.0)

    Returns:
        Interpolated Voc in Volts

    Raises:
        ValueError: If irradiance values are invalid

    Reference:
        IEC 61853-1:2011 Clause 7.3
    """
    if g_ref <= 0:
        raise ValueError("Reference irradiance must be positive")
    if g_target <= 0:
        raise ValueError("Target irradiance must be positive for Voc calculation")

    # Thermal voltage at given temperature
    v_thermal = THERMAL_VOLTAGE_25C * (temp_kelvin / 298.15)

    # Logarithmic correction
    delta_voc = ideality_factor * n_cells * v_thermal * np.log(g_target / g_ref)

    return voc_ref + delta_voc


def interpolate_voc_temperature(
    voc_ref: float,
    temp_ref: float,
    temp_target: float,
    beta: float
) -> float:
    """
    Interpolate open-circuit voltage for temperature using linear relationship.

    Per IEC 61853-1 Clause 7.4:
    Voc(T) = Voc_ref * [1 + beta * (T - T_ref)]

    Args:
        voc_ref: Reference Voc at temp_ref in Volts
        temp_ref: Reference temperature in °C
        temp_target: Target temperature in °C
        beta: Temperature coefficient in %/°C (typically negative)

    Returns:
        Interpolated Voc in Volts

    Reference:
        IEC 61853-1:2011 Clause 7.4
    """
    beta_decimal = beta / 100.0
    return voc_ref * (1 + beta_decimal * (temp_target - temp_ref))


def interpolate_pmax_polynomial(
    irradiances: NDArray[np.float64],
    pmax_values: NDArray[np.float64],
    g_target: float,
    degree: int = 2
) -> float:
    """
    Interpolate maximum power using polynomial fit vs irradiance.

    Per IEC 61853-1 Clause 7.5, Pmax can be fitted with a polynomial:
    Pmax(G) = a0 + a1*G + a2*G²

    Args:
        irradiances: Array of irradiance values in W/m²
        pmax_values: Array of corresponding Pmax values in W
        g_target: Target irradiance in W/m²
        degree: Polynomial degree (default: 2 for quadratic)

    Returns:
        Interpolated Pmax in Watts

    Raises:
        ValueError: If insufficient data points

    Reference:
        IEC 61853-1:2011 Clause 7.5
    """
    valid = ~np.isnan(pmax_values)
    if np.sum(valid) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} valid points for degree-{degree} polynomial")

    coeffs = np.polyfit(irradiances[valid], pmax_values[valid], degree)
    return np.polyval(coeffs, g_target)


def interpolate_pmax_temperature(
    pmax_ref: float,
    temp_ref: float,
    temp_target: float,
    gamma: float
) -> float:
    """
    Interpolate maximum power for temperature.

    Pmax(T) = Pmax_ref * [1 + gamma * (T - T_ref)]

    Args:
        pmax_ref: Reference Pmax at temp_ref in Watts
        temp_ref: Reference temperature in °C
        temp_target: Target temperature in °C
        gamma: Temperature coefficient in %/°C (typically negative)

    Returns:
        Interpolated Pmax in Watts

    Reference:
        IEC 61853-1:2011 Clause 7.5
    """
    gamma_decimal = gamma / 100.0
    return pmax_ref * (1 + gamma_decimal * (temp_target - temp_ref))


# =============================================================================
# Power Matrix Operations
# =============================================================================

def create_interpolated_surface(
    power_matrix: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    method: str = 'cubic'
) -> interpolate.RectBivariateSpline:
    """
    Create a 2D interpolation surface from power matrix.

    Args:
        power_matrix: Pmax values array of shape (n_irradiances, n_temperatures)
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        method: Interpolation method ('linear', 'cubic')

    Returns:
        RectBivariateSpline interpolator

    Reference:
        IEC 61853-1:2011 Clause 8
    """
    g_arr = np.array(irradiances, dtype=np.float64)
    t_arr = np.array(temperatures, dtype=np.float64)

    # Handle NaN values by interpolating first
    matrix_filled = _fill_nan_values(power_matrix, g_arr, t_arr)

    kx = 3 if method == 'cubic' else 1
    ky = 3 if method == 'cubic' else 1

    return interpolate.RectBivariateSpline(g_arr, t_arr, matrix_filled, kx=kx, ky=ky)


def _fill_nan_values(
    matrix: NDArray[np.float64],
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Fill NaN values in matrix using nearest neighbor interpolation.

    Args:
        matrix: 2D array with potential NaN values
        x_values: X-axis values
        y_values: Y-axis values

    Returns:
        Matrix with NaN values filled
    """
    result = matrix.copy()
    nan_mask = np.isnan(result)

    if not np.any(nan_mask):
        return result

    # Create meshgrid
    xx, yy = np.meshgrid(x_values, y_values, indexing='ij')

    # Get valid points
    valid_mask = ~nan_mask
    valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
    valid_values = result[valid_mask]

    # Interpolate NaN points
    nan_points = np.column_stack([xx[nan_mask], yy[nan_mask]])

    if len(valid_points) > 0 and len(nan_points) > 0:
        filled_values = interpolate.griddata(
            valid_points, valid_values, nan_points,
            method='nearest'
        )
        result[nan_mask] = filled_values

    return result


def interpolate_power_at_conditions(
    power_matrix: NDArray[np.float64],
    isc_matrix: NDArray[np.float64],
    voc_matrix: NDArray[np.float64],
    g_target: float,
    t_target: float,
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES,
    module_area: float = 1.7
) -> InterpolationResult:
    """
    Interpolate all electrical parameters at specific G and T conditions.

    Uses bilinear interpolation on the power matrix for accurate results
    within the measured range.

    Args:
        power_matrix: Pmax values array
        isc_matrix: Isc values array
        voc_matrix: Voc values array
        g_target: Target irradiance in W/m²
        t_target: Target temperature in °C
        irradiances: List of irradiance levels
        temperatures: List of temperature levels
        module_area: Module area in m² for efficiency calculation

    Returns:
        InterpolationResult with all interpolated parameters

    Reference:
        IEC 61853-1:2011 Clause 8
    """
    g_arr = np.array(irradiances, dtype=np.float64)
    t_arr = np.array(temperatures, dtype=np.float64)

    # Create interpolators
    pmax_interp = create_interpolated_surface(power_matrix, irradiances, temperatures)
    isc_interp = create_interpolated_surface(isc_matrix, irradiances, temperatures)
    voc_interp = create_interpolated_surface(voc_matrix, irradiances, temperatures)

    # Interpolate values
    pmax = float(pmax_interp(g_target, t_target)[0, 0])
    isc = float(isc_interp(g_target, t_target)[0, 0])
    voc = float(voc_interp(g_target, t_target)[0, 0])

    # Calculate efficiency
    efficiency = (pmax / (g_target * module_area)) * 100 if g_target > 0 else 0.0

    # Determine if interpolation or extrapolation
    in_range = (
        g_arr.min() <= g_target <= g_arr.max() and
        t_arr.min() <= t_target <= t_arr.max()
    )
    method = "bilinear_interpolation" if in_range else "bilinear_extrapolation"

    return InterpolationResult(
        irradiance=g_target,
        temperature=t_target,
        pmax=max(0, pmax),  # Ensure non-negative
        isc=max(0, isc),
        voc=max(0, voc),
        efficiency=max(0, efficiency),
        method=method
    )


# =============================================================================
# CSER Calculations
# =============================================================================

def calculate_cser(
    power_matrix: NDArray[np.float64],
    hourly_irradiance: NDArray[np.float64],
    hourly_temperature: NDArray[np.float64],
    pmax_stc: float,
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES
) -> Dict[str, float]:
    """
    Calculate Climate Specific Energy Rating (CSER).

    Per IEC 61853-3, CSER is calculated by integrating the power output
    over a typical meteorological year using the power matrix:

    CSER = Σ(P(Gi, Ti) × Δt) / Pmax_STC

    Args:
        power_matrix: Pmax values array of shape (n_irradiances, n_temperatures)
        hourly_irradiance: Hourly POA irradiance values for TMY (8760 values) in W/m²
        hourly_temperature: Hourly cell temperature values for TMY (8760 values) in °C
        pmax_stc: Maximum power at STC in Watts
        irradiances: List of irradiance levels
        temperatures: List of temperature levels

    Returns:
        Dictionary with:
            - annual_energy: Total annual energy in kWh
            - cser_rating: CSER value (kWh/kWp)
            - specific_yield: Energy per rated power (kWh/kWp)
            - capacity_factor: Ratio of actual to theoretical max output
            - performance_ratio: Ratio of actual to ideal energy

    Reference:
        IEC 61853-3:2018 Clause 5
    """
    if len(hourly_irradiance) != len(hourly_temperature):
        raise ValueError("Irradiance and temperature arrays must have same length")

    # Create interpolation surface
    pmax_interp = create_interpolated_surface(power_matrix, irradiances, temperatures)

    # Calculate hourly power output
    hourly_power = np.zeros(len(hourly_irradiance))

    for i, (g, t) in enumerate(zip(hourly_irradiance, hourly_temperature)):
        if g > 0:  # Only calculate for positive irradiance
            # Clip to valid range for interpolation
            g_clipped = np.clip(g, min(irradiances), max(irradiances))
            t_clipped = np.clip(t, min(temperatures), max(temperatures))

            power = pmax_interp(g_clipped, t_clipped)[0, 0]
            hourly_power[i] = max(0, power)  # Ensure non-negative

    # Calculate metrics
    annual_energy = np.sum(hourly_power) / 1000  # Convert Wh to kWh

    # CSER rating (kWh per kWp of STC rating)
    pmax_stc_kw = pmax_stc / 1000
    cser_rating = annual_energy / pmax_stc_kw if pmax_stc_kw > 0 else 0.0

    # Specific yield (same as CSER for single module)
    specific_yield = cser_rating

    # Capacity factor
    hours_per_year = 8760
    theoretical_max = pmax_stc * hours_per_year / 1000  # kWh
    capacity_factor = annual_energy / theoretical_max if theoretical_max > 0 else 0.0

    # Performance ratio (ratio of actual energy to ideal energy at STC)
    total_irradiation = np.sum(hourly_irradiance) / 1000  # kWh/m²
    ideal_energy = (pmax_stc / 1000) * total_irradiation  # Assumes 1 kW/m² = 1000 W/m²
    performance_ratio = annual_energy / ideal_energy if ideal_energy > 0 else 0.0

    return {
        'annual_energy': round(annual_energy, 2),
        'cser_rating': round(cser_rating, 1),
        'specific_yield': round(specific_yield, 1),
        'capacity_factor': round(capacity_factor, 4),
        'performance_ratio': round(performance_ratio, 4)
    }


def calculate_monthly_energy(
    power_matrix: NDArray[np.float64],
    hourly_irradiance: NDArray[np.float64],
    hourly_temperature: NDArray[np.float64],
    irradiances: List[int] = IRRADIANCES,
    temperatures: List[int] = TEMPERATURES
) -> List[float]:
    """
    Calculate monthly energy production.

    Args:
        power_matrix: Pmax values array
        hourly_irradiance: Hourly POA irradiance (8760 values) in W/m²
        hourly_temperature: Hourly cell temperature (8760 values) in °C
        irradiances: List of irradiance levels
        temperatures: List of temperature levels

    Returns:
        List of 12 monthly energy values in kWh

    Reference:
        IEC 61853-3:2018 Clause 5
    """
    # Hours per month (non-leap year)
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    pmax_interp = create_interpolated_surface(power_matrix, irradiances, temperatures)

    monthly_energy = []
    hour_idx = 0

    for month_hours in hours_per_month:
        month_power = 0.0

        for _ in range(month_hours):
            if hour_idx < len(hourly_irradiance):
                g = hourly_irradiance[hour_idx]
                t = hourly_temperature[hour_idx]

                if g > 0:
                    g_clipped = np.clip(g, min(irradiances), max(irradiances))
                    t_clipped = np.clip(t, min(temperatures), max(temperatures))
                    power = pmax_interp(g_clipped, t_clipped)[0, 0]
                    month_power += max(0, power)

                hour_idx += 1

        monthly_energy.append(round(month_power / 1000, 2))  # kWh

    return monthly_energy


# =============================================================================
# Utility Functions
# =============================================================================

def validate_power_matrix(
    power_matrix: NDArray[np.float64],
    min_valid_points: int = 22
) -> Tuple[bool, str]:
    """
    Validate power matrix data quality.

    Args:
        power_matrix: Pmax values array
        min_valid_points: Minimum required valid (non-NaN) points

    Returns:
        Tuple of (is_valid, message)

    Reference:
        IEC 61853-1:2011 Clause 6
    """
    total_points = power_matrix.size
    valid_points = np.sum(~np.isnan(power_matrix))

    if valid_points < min_valid_points:
        return False, f"Insufficient data: {valid_points}/{min_valid_points} required points"

    # Check for negative values
    if np.any(power_matrix[~np.isnan(power_matrix)] < 0):
        return False, "Power matrix contains negative values"

    # Check for reasonable power progression (higher G should give higher P at same T)
    for t_idx in range(power_matrix.shape[1]):
        col = power_matrix[:, t_idx]
        valid = ~np.isnan(col)
        if np.sum(valid) > 1:
            if not np.all(np.diff(col[valid]) >= -0.1):  # Allow small tolerance
                return False, f"Power does not increase with irradiance at T index {t_idx}"

    return True, "Power matrix validation passed"


def calculate_efficiency(
    pmax: float,
    irradiance: float,
    area: float
) -> float:
    """
    Calculate module efficiency.

    η = Pmax / (G × A) × 100

    Args:
        pmax: Maximum power in Watts
        irradiance: Irradiance in W/m²
        area: Module area in m²

    Returns:
        Efficiency in percent
    """
    if irradiance <= 0 or area <= 0:
        return 0.0
    return (pmax / (irradiance * area)) * 100


def calculate_fill_factor(
    pmax: float,
    isc: float,
    voc: float
) -> float:
    """
    Calculate fill factor.

    FF = Pmax / (Isc × Voc)

    Args:
        pmax: Maximum power in Watts
        isc: Short-circuit current in Amperes
        voc: Open-circuit voltage in Volts

    Returns:
        Fill factor (0-1)
    """
    if isc <= 0 or voc <= 0:
        return 0.0
    return pmax / (isc * voc)
