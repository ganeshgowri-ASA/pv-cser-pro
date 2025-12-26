"""
Climate Specific Energy Rating (CSER) Calculator Module

This module implements the IEC 61853-3 energy rating methodology for
photovoltaic modules. It calculates the expected annual energy yield
under specific climatic conditions.

The CSER provides a dimensionless rating that indicates how well a
module performs relative to a reference condition, enabling comparison
across different module technologies and climate zones.

Key Calculations:
    - Module temperature modeling (NOCT-based)
    - Power interpolation from P(G,T) matrix
    - Spectral correction factors
    - Incidence angle modifier (IAM) correction
    - Annual energy simulation (8760 hours)
    - CSER rating calculation

References:
    - IEC 61853-1:2011 - Irradiance and temperature performance
    - IEC 61853-2:2016 - Spectral responsivity, IAM, module temperature
    - IEC 61853-3:2018 - Energy rating of PV modules
    - IEC 61853-4:2018 - Standard reference climatic profiles
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from .climate_profiles import ClimateProfile, load_standard_profile
from .interpolation import bilinear_interpolate, interpolate_power_matrix, InterpolationError


# Standard Test Conditions (STC)
STC_IRRADIANCE = 1000.0  # W/m^2
STC_TEMPERATURE = 25.0   # C

# Nominal Operating Cell Temperature conditions
NOCT_IRRADIANCE = 800.0  # W/m^2
NOCT_AMBIENT = 20.0      # C
NOCT_WIND = 1.0          # m/s


@dataclass
class ModuleParameters:
    """
    PV module parameters for energy rating calculations.

    Contains all parameters needed for IEC 61853-3 calculations,
    including power matrix, temperature coefficients, and optical properties.

    Attributes:
        P_stc: Rated power at STC in Watts.
        noct: Nominal Operating Cell Temperature in C (typically 45-48).
        gamma_pmax: Temperature coefficient of Pmax in %/C (typically -0.3 to -0.5).
        gamma_voc: Temperature coefficient of Voc in %/C.
        gamma_isc: Temperature coefficient of Isc in %/C.
        G_grid: Irradiance grid points for power matrix (W/m^2).
        T_grid: Temperature grid points for power matrix (C).
        P_matrix: Power matrix P(G,T) in Watts.
        iam_coefficients: Incidence angle modifier coefficients.
        spectral_mismatch: Spectral mismatch factor (default 1.0).
        efficiency_stc: Module efficiency at STC (fraction, 0-1).
    """
    P_stc: float
    noct: float = 45.0
    gamma_pmax: float = -0.4  # %/C
    gamma_voc: float = -0.3   # %/C
    gamma_isc: float = 0.05   # %/C
    G_grid: NDArray[np.float64] = field(
        default_factory=lambda: np.array([100, 200, 400, 600, 800, 1000, 1100])
    )
    T_grid: NDArray[np.float64] = field(
        default_factory=lambda: np.array([15, 25, 50, 75])
    )
    P_matrix: Optional[NDArray[np.float64]] = None
    iam_coefficients: Dict[str, float] = field(default_factory=dict)
    spectral_mismatch: float = 1.0
    efficiency_stc: float = 0.20

    def __post_init__(self):
        """Generate default power matrix if not provided."""
        self.G_grid = np.asarray(self.G_grid, dtype=np.float64)
        self.T_grid = np.asarray(self.T_grid, dtype=np.float64)

        if self.P_matrix is None:
            self.P_matrix = self._generate_default_power_matrix()
        else:
            self.P_matrix = np.asarray(self.P_matrix, dtype=np.float64)

        # Set default IAM coefficients if not provided
        if not self.iam_coefficients:
            # Default ASHRAE IAM model coefficient
            self.iam_coefficients = {'b0': 0.05}

    def _generate_default_power_matrix(self) -> NDArray[np.float64]:
        """
        Generate a default power matrix based on STC power and temperature coefficient.

        Uses the simplified model:
        P(G, T) = P_stc * (G / 1000) * (1 + gamma_pmax/100 * (T - 25))

        Returns:
            2D array with shape (len(T_grid), len(G_grid)).
        """
        P_matrix = np.zeros((len(self.T_grid), len(self.G_grid)))

        for i, T in enumerate(self.T_grid):
            for j, G in enumerate(self.G_grid):
                temp_factor = 1 + (self.gamma_pmax / 100) * (T - 25)
                irr_factor = G / STC_IRRADIANCE
                P_matrix[i, j] = self.P_stc * irr_factor * temp_factor

        return P_matrix


@dataclass
class CSERResult:
    """
    Result of CSER calculation.

    Contains all outputs from an IEC 61853-3 energy rating calculation.

    Attributes:
        cser: Climate Specific Energy Rating (dimensionless).
        annual_energy_kwh: Predicted annual energy yield in kWh.
        reference_energy_kwh: Reference annual energy in kWh.
        performance_ratio: System performance ratio (0-1).
        capacity_factor: Capacity factor (0-1).
        specific_yield: Specific yield in kWh/kWp.
        hourly_power: 8760-element array of hourly power output (W).
        hourly_module_temp: 8760-element array of module temperature (C).
        climate_profile: Name of climate profile used.
        average_efficiency: Average operating efficiency.
        temperature_losses: Fraction of energy lost to temperature (0-1).
        spectral_losses: Fraction of energy lost to spectral effects (0-1).
        iam_losses: Fraction of energy lost to IAM effects (0-1).
    """
    cser: float
    annual_energy_kwh: float
    reference_energy_kwh: float
    performance_ratio: float
    capacity_factor: float
    specific_yield: float
    hourly_power: NDArray[np.float64]
    hourly_module_temp: NDArray[np.float64]
    climate_profile: str
    average_efficiency: float = 0.0
    temperature_losses: float = 0.0
    spectral_losses: float = 0.0
    iam_losses: float = 0.0


def calculate_module_temperature(
    irradiance: Union[float, NDArray[np.float64]],
    ambient_temp: Union[float, NDArray[np.float64]],
    noct: float = 45.0,
    efficiency: float = 0.20,
    wind_speed: Optional[Union[float, NDArray[np.float64]]] = None
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate module temperature using the NOCT model.

    Implements the standard NOCT-based temperature model as specified
    in IEC 61853-2. The model accounts for irradiance heating and
    the effect of electrical efficiency on absorbed heat.

    The formula is:
        T_module = T_amb + (NOCT - 20) * (G / 800) * (1 - eta)

    Where:
        - T_amb: Ambient temperature (C)
        - NOCT: Nominal Operating Cell Temperature (C)
        - G: Irradiance (W/m^2)
        - eta: Module efficiency (fraction)

    Args:
        irradiance: In-plane irradiance in W/m^2.
        ambient_temp: Ambient temperature in C.
        noct: Nominal Operating Cell Temperature in C.
        efficiency: Module electrical efficiency (fraction, 0-1).
        wind_speed: Optional wind speed in m/s for wind correction.

    Returns:
        Module temperature in C.

    Example:
        >>> T_mod = calculate_module_temperature(800, 25, noct=45, efficiency=0.20)
        >>> print(f"Module temperature: {T_mod:.1f} C")
        Module temperature: 45.0 C

    References:
        IEC 61853-2:2016, Section 7.3 - Module operating temperature
    """
    G = np.asarray(irradiance, dtype=np.float64)
    T_amb = np.asarray(ambient_temp, dtype=np.float64)

    # Base NOCT model
    # Factor (1 - efficiency) accounts for heat not converted to electricity
    delta_T = (noct - NOCT_AMBIENT) * (G / NOCT_IRRADIANCE) * (1 - efficiency)

    # Wind speed correction (optional)
    if wind_speed is not None:
        ws = np.asarray(wind_speed, dtype=np.float64)
        # Empirical wind correction factor
        # At NOCT conditions, wind is 1 m/s
        wind_factor = np.maximum(0.5, 1 - 0.05 * (ws - NOCT_WIND))
        delta_T *= wind_factor

    T_module = T_amb + delta_T

    # Return scalar if input was scalar
    if np.ndim(T_module) == 0:
        return float(T_module)
    return T_module


def calculate_hourly_power(
    irradiance: Union[float, NDArray[np.float64]],
    module_temp: Union[float, NDArray[np.float64]],
    module: ModuleParameters
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate power output using bilinear interpolation from power matrix.

    Interpolates the measured power from the P(G,T) matrix for the
    given operating conditions. This is the core calculation method
    as specified in IEC 61853-3.

    Args:
        irradiance: In-plane irradiance in W/m^2.
        module_temp: Module temperature in C.
        module: ModuleParameters object with power matrix.

    Returns:
        DC power output in Watts.

    Example:
        >>> module = ModuleParameters(P_stc=400)
        >>> power = calculate_hourly_power(800, 45, module)
        >>> print(f"Power output: {power:.1f} W")

    References:
        IEC 61853-3:2018, Section 6.3 - Energy rating calculation
    """
    return interpolate_power_matrix(
        irradiance,
        module_temp,
        module.G_grid,
        module.T_grid,
        module.P_matrix,
        extrapolate=True
    )


def apply_spectral_correction(
    power: Union[float, NDArray[np.float64]],
    air_mass: Union[float, NDArray[np.float64]] = 1.5,
    precipitable_water: float = 2.0,
    module_type: str = 'cSi'
) -> Union[float, NDArray[np.float64]]:
    """
    Apply spectral mismatch correction factor.

    Corrects the power output for differences between the actual
    solar spectrum and the reference AM1.5G spectrum used for STC.

    Different PV technologies have different spectral responses:
    - c-Si: Moderate spectral sensitivity
    - CdTe: High sensitivity to blue-rich spectra
    - a-Si: High sensitivity to spectral variations
    - CIGS: Moderate spectral sensitivity

    Args:
        power: DC power in Watts.
        air_mass: Atmospheric air mass (typical range 1.0-5.0).
        precipitable_water: Precipitable water vapor in cm.
        module_type: Module technology ('cSi', 'CdTe', 'aSi', 'CIGS').

    Returns:
        Spectrally corrected power in Watts.

    Note:
        This is a simplified model. For accurate spectral correction,
        use the full IEC 61853-2 spectral responsivity method.

    References:
        IEC 61853-2:2016, Section 7.1 - Spectral responsivity
        IEC 61853-3:2018, Section 6.3.3 - Spectral correction
    """
    P = np.asarray(power, dtype=np.float64)
    AM = np.asarray(air_mass, dtype=np.float64)

    # Spectral correction coefficients by technology
    # These are approximate values based on typical module behavior
    coefficients = {
        'cSi': {'a1': 0.0, 'a2': 0.02, 'b1': 0.0, 'b2': 0.01},
        'CdTe': {'a1': 0.05, 'a2': 0.03, 'b1': 0.02, 'b2': 0.02},
        'aSi': {'a1': 0.08, 'a2': 0.04, 'b1': 0.03, 'b2': 0.03},
        'CIGS': {'a1': 0.02, 'a2': 0.02, 'b1': 0.01, 'b2': 0.01}
    }

    coef = coefficients.get(module_type, coefficients['cSi'])

    # Simplified spectral mismatch factor
    # SMF = 1 + a1*(AM - 1.5) + a2*(AM - 1.5)^2 + b1*(PW - 2) + b2*(PW - 2)^2
    delta_AM = AM - 1.5
    delta_PW = precipitable_water - 2.0

    spectral_factor = 1 + (
        coef['a1'] * delta_AM +
        coef['a2'] * delta_AM**2 +
        coef['b1'] * delta_PW +
        coef['b2'] * delta_PW**2
    )

    # Clamp to reasonable range
    spectral_factor = np.clip(spectral_factor, 0.9, 1.1)

    corrected = P * spectral_factor

    if np.ndim(corrected) == 0:
        return float(corrected)
    return corrected


def apply_iam_correction(
    power: Union[float, NDArray[np.float64]],
    aoi: Union[float, NDArray[np.float64]],
    iam_model: str = 'ashrae',
    b0: float = 0.05
) -> Union[float, NDArray[np.float64]]:
    """
    Apply Incidence Angle Modifier (IAM) correction.

    The IAM accounts for reflection losses at the module surface
    that increase as the angle of incidence increases.

    Supported models:
    - ASHRAE: IAM = 1 - b0 * (1/cos(AOI) - 1)
    - Physical: IAM = 1 - b0 * (1/cos(AOI) - 1) for AOI < 80
    - Martin-Ruiz: Polynomial approximation

    Args:
        power: DC power in Watts.
        aoi: Angle of incidence in degrees (0 = normal incidence).
        iam_model: IAM model to use ('ashrae', 'physical', 'martin-ruiz').
        b0: ASHRAE model coefficient (typical 0.04-0.06).

    Returns:
        IAM-corrected power in Watts.

    References:
        IEC 61853-2:2016, Section 7.2 - Incidence angle modifier
    """
    P = np.asarray(power, dtype=np.float64)
    theta = np.asarray(aoi, dtype=np.float64)

    # Convert to radians
    theta_rad = np.deg2rad(theta)

    # Handle edge cases
    cos_theta = np.cos(theta_rad)
    cos_theta = np.clip(cos_theta, 0.01, 1.0)  # Avoid division by zero

    if iam_model == 'ashrae':
        # ASHRAE model
        iam = 1 - b0 * (1 / cos_theta - 1)
    elif iam_model == 'physical':
        # Physical model with cutoff at high angles
        iam = np.where(
            theta < 80,
            1 - b0 * (1 / cos_theta - 1),
            0.0
        )
    elif iam_model == 'martin-ruiz':
        # Martin-Ruiz polynomial approximation
        ar = 0.16  # Glass air-refraction
        iam = 1 - np.exp(-(cos_theta / ar) * (1 - np.exp(-cos_theta / ar)))
    else:
        # Default: no correction
        iam = np.ones_like(theta)

    # Clamp IAM to valid range
    iam = np.clip(iam, 0.0, 1.0)

    corrected = P * iam

    if np.ndim(corrected) == 0:
        return float(corrected)
    return corrected


def calculate_annual_energy(
    climate: ClimateProfile,
    module: ModuleParameters,
    apply_spectral: bool = True,
    apply_iam: bool = True,
    iam_aoi: Optional[NDArray[np.float64]] = None
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate annual energy yield for 8760 hours.

    Performs a complete hourly simulation over one year (8760 hours)
    using the climate profile and module parameters.

    Args:
        climate: ClimateProfile with hourly irradiance and temperature.
        module: ModuleParameters with power matrix and coefficients.
        apply_spectral: Apply spectral correction factor.
        apply_iam: Apply incidence angle modifier correction.
        iam_aoi: Optional 8760-element array of angles of incidence.
            If not provided, a default profile is generated.

    Returns:
        Tuple of:
            - Annual energy in kWh
            - Hourly power array (8760 elements, Watts)
            - Hourly module temperature array (8760 elements, C)

    Example:
        >>> climate = load_standard_profile('tropical')
        >>> module = ModuleParameters(P_stc=400, noct=45)
        >>> energy, power, temp = calculate_annual_energy(climate, module)
        >>> print(f"Annual yield: {energy:.1f} kWh")

    References:
        IEC 61853-3:2018, Section 6 - Energy rating method
    """
    # Get hourly data
    G = climate.hourly_irradiance
    T_amb = climate.hourly_temperature

    # Calculate module temperature for each hour
    T_module = calculate_module_temperature(
        G, T_amb,
        noct=module.noct,
        efficiency=module.efficiency_stc
    )

    # Calculate hourly power from power matrix
    hourly_power = np.zeros(8760, dtype=np.float64)

    for hour in range(8760):
        if G[hour] > 0:
            try:
                hourly_power[hour] = calculate_hourly_power(
                    G[hour], T_module[hour], module
                )
            except (InterpolationError, ValueError):
                # Fall back to simple model for out-of-range conditions
                temp_factor = 1 + (module.gamma_pmax / 100) * (T_module[hour] - 25)
                irr_factor = G[hour] / STC_IRRADIANCE
                hourly_power[hour] = module.P_stc * irr_factor * temp_factor
                hourly_power[hour] = max(0, hourly_power[hour])

    # Apply spectral correction
    if apply_spectral:
        # Estimate air mass from hour of day (simplified)
        hours = np.arange(8760)
        hour_of_day = hours % 24

        # Simple air mass approximation
        # AM = 1/cos(zenith), with zenith varying by hour
        zenith = np.abs(hour_of_day - 12) * 7.5  # degrees, rough approximation
        zenith = np.clip(zenith, 0, 85)
        air_mass = 1 / np.cos(np.deg2rad(zenith))
        air_mass = np.clip(air_mass, 1.0, 5.0)

        pw = climate.spectral_characteristics.get('precipitable_water_cm', 2.0)

        hourly_power = apply_spectral_correction(
            hourly_power, air_mass, pw, 'cSi'
        )

    # Apply IAM correction
    if apply_iam:
        if iam_aoi is None:
            # Generate default AOI profile
            hours = np.arange(8760)
            hour_of_day = hours % 24
            # Simple AOI approximation (degrees from normal)
            iam_aoi = np.abs(hour_of_day - 12) * 5
            iam_aoi = np.clip(iam_aoi, 0, 85)

        b0 = module.iam_coefficients.get('b0', 0.05)
        hourly_power = apply_iam_correction(hourly_power, iam_aoi, 'ashrae', b0)

    # Ensure non-negative power
    hourly_power = np.maximum(0, hourly_power)

    # Calculate annual energy (kWh)
    # Each hour represents 1 hour, so sum gives Wh, divide by 1000 for kWh
    annual_energy_kwh = np.sum(hourly_power) / 1000.0

    return annual_energy_kwh, hourly_power, T_module


def calculate_cser_rating(
    annual_energy_kwh: float,
    reference_energy_kwh: float
) -> float:
    """
    Calculate the Climate Specific Energy Rating (CSER).

    The CSER is a dimensionless ratio of actual energy output to
    reference energy output. It provides a technology-neutral
    comparison metric.

    CSER = E_actual / E_reference

    Where:
        - E_actual: Calculated annual energy for the climate
        - E_reference: Reference energy (typically at STC over 1000 kWh/m^2)

    Args:
        annual_energy_kwh: Calculated annual energy in kWh.
        reference_energy_kwh: Reference annual energy in kWh.

    Returns:
        CSER rating (dimensionless, typically 0.8-1.2).

    Example:
        >>> cser = calculate_cser_rating(1200, 1000)
        >>> print(f"CSER: {cser:.3f}")
        CSER: 1.200

    References:
        IEC 61853-3:2018, Section 7 - Energy rating
    """
    if reference_energy_kwh <= 0:
        raise ValueError("Reference energy must be positive")

    return annual_energy_kwh / reference_energy_kwh


def calculate_performance_ratio(
    annual_energy_kwh: float,
    P_stc: float,
    annual_irradiation_kwh_m2: float
) -> float:
    """
    Calculate the Performance Ratio (PR).

    The PR is the ratio of actual energy output to the theoretical
    maximum energy if the module always operated at STC efficiency.

    PR = E_actual / (P_stc * H / G_stc)

    Where:
        - E_actual: Actual annual energy (kWh)
        - P_stc: Rated power at STC (kW)
        - H: Annual irradiation (kWh/m^2)
        - G_stc: STC irradiance (1 kW/m^2)

    Args:
        annual_energy_kwh: Actual annual energy yield in kWh.
        P_stc: Module rated power at STC in Watts.
        annual_irradiation_kwh_m2: Annual irradiation in kWh/m^2.

    Returns:
        Performance ratio (0-1, typically 0.7-0.9).

    Example:
        >>> pr = calculate_performance_ratio(1500, 400, 2000)
        >>> print(f"Performance Ratio: {pr:.1%}")
        Performance Ratio: 93.8%

    References:
        IEC 61724-1:2017 - PV system performance monitoring
    """
    if P_stc <= 0:
        raise ValueError("P_stc must be positive")
    if annual_irradiation_kwh_m2 <= 0:
        raise ValueError("Annual irradiation must be positive")

    P_stc_kw = P_stc / 1000.0
    theoretical_energy = P_stc_kw * annual_irradiation_kwh_m2

    return annual_energy_kwh / theoretical_energy


def calculate_capacity_factor(
    annual_energy_kwh: float,
    P_stc: float
) -> float:
    """
    Calculate the Capacity Factor (CF).

    The CF is the ratio of actual energy output to the theoretical
    maximum if the module operated at full rated power 24/7.

    CF = E_actual / (P_stc * 8760)

    Args:
        annual_energy_kwh: Actual annual energy yield in kWh.
        P_stc: Module rated power at STC in Watts.

    Returns:
        Capacity factor (0-1, typically 0.10-0.25 for fixed-tilt).

    Example:
        >>> cf = calculate_capacity_factor(600, 400)
        >>> print(f"Capacity Factor: {cf:.1%}")
        Capacity Factor: 17.1%
    """
    if P_stc <= 0:
        raise ValueError("P_stc must be positive")

    P_stc_kw = P_stc / 1000.0
    max_annual_energy = P_stc_kw * 8760  # hours per year

    return annual_energy_kwh / max_annual_energy


def calculate_specific_yield(
    annual_energy_kwh: float,
    P_stc: float
) -> float:
    """
    Calculate the Specific Yield (kWh/kWp).

    The specific yield normalizes energy output by system size,
    enabling comparison across different sized systems.

    SY = E_actual / P_stc

    Args:
        annual_energy_kwh: Actual annual energy yield in kWh.
        P_stc: Module rated power at STC in Watts.

    Returns:
        Specific yield in kWh/kWp.

    Example:
        >>> sy = calculate_specific_yield(600, 400)
        >>> print(f"Specific Yield: {sy:.0f} kWh/kWp")
        Specific Yield: 1500 kWh/kWp
    """
    if P_stc <= 0:
        raise ValueError("P_stc must be positive")

    P_stc_kw = P_stc / 1000.0
    return annual_energy_kwh / P_stc_kw


def run_cser_analysis(
    climate: Union[ClimateProfile, str],
    module: ModuleParameters,
    apply_corrections: bool = True
) -> CSERResult:
    """
    Run complete CSER analysis for a module and climate profile.

    This is the main entry point for IEC 61853-3 energy rating
    calculations. It performs all necessary calculations and
    returns a comprehensive result object.

    Args:
        climate: ClimateProfile object or profile name string.
        module: ModuleParameters with module specifications.
        apply_corrections: Apply spectral and IAM corrections.

    Returns:
        CSERResult with all calculation outputs.

    Example:
        >>> module = ModuleParameters(P_stc=400, noct=45, gamma_pmax=-0.40)
        >>> result = run_cser_analysis('tropical', module)
        >>> print(f"CSER: {result.cser:.3f}")
        >>> print(f"Annual Energy: {result.annual_energy_kwh:.1f} kWh")
        >>> print(f"Performance Ratio: {result.performance_ratio:.1%}")

    References:
        IEC 61853-3:2018 - Energy rating of PV modules
    """
    # Load climate profile if string
    if isinstance(climate, str):
        climate = load_standard_profile(climate)

    # Calculate annual energy
    annual_energy_kwh, hourly_power, hourly_temp = calculate_annual_energy(
        climate,
        module,
        apply_spectral=apply_corrections,
        apply_iam=apply_corrections
    )

    # Calculate reference energy (STC power * annual irradiation)
    reference_energy_kwh = (module.P_stc / 1000.0) * climate.annual_irradiation

    # Calculate metrics
    cser = calculate_cser_rating(annual_energy_kwh, reference_energy_kwh)

    performance_ratio = calculate_performance_ratio(
        annual_energy_kwh,
        module.P_stc,
        climate.annual_irradiation
    )

    capacity_factor = calculate_capacity_factor(annual_energy_kwh, module.P_stc)

    specific_yield = calculate_specific_yield(annual_energy_kwh, module.P_stc)

    # Calculate average efficiency
    # Efficiency = P_out / (G * A), where P_out = P_stc * eta_stc at STC
    # Average efficiency relative to STC
    daytime_mask = climate.hourly_irradiance > 50
    if np.any(daytime_mask):
        avg_power = np.mean(hourly_power[daytime_mask])
        avg_irradiance = np.mean(climate.hourly_irradiance[daytime_mask])
        theoretical_power = module.P_stc * (avg_irradiance / STC_IRRADIANCE)
        avg_efficiency = avg_power / theoretical_power if theoretical_power > 0 else 0
    else:
        avg_efficiency = 0

    # Estimate temperature losses
    # Compare to isothermal case at 25 C
    temp_loss_factor = np.mean(1 + (module.gamma_pmax / 100) * (hourly_temp[daytime_mask] - 25))
    temperature_losses = 1 - temp_loss_factor if daytime_mask.any() else 0

    return CSERResult(
        cser=cser,
        annual_energy_kwh=annual_energy_kwh,
        reference_energy_kwh=reference_energy_kwh,
        performance_ratio=performance_ratio,
        capacity_factor=capacity_factor,
        specific_yield=specific_yield,
        hourly_power=hourly_power,
        hourly_module_temp=hourly_temp,
        climate_profile=climate.name,
        average_efficiency=avg_efficiency,
        temperature_losses=max(0, temperature_losses),
        spectral_losses=0.02 if apply_corrections else 0,  # Approximate
        iam_losses=0.03 if apply_corrections else 0  # Approximate
    )


def compare_climates(
    module: ModuleParameters,
    climate_codes: Optional[List[str]] = None
) -> Dict[str, CSERResult]:
    """
    Compare module performance across multiple climate profiles.

    Runs CSER analysis for each specified climate and returns
    a dictionary of results for comparison.

    Args:
        module: ModuleParameters to analyze.
        climate_codes: List of climate profile codes. If None, uses
            all six standard IEC 61853-4 profiles.

    Returns:
        Dictionary mapping climate names to CSERResult objects.

    Example:
        >>> module = ModuleParameters(P_stc=400)
        >>> results = compare_climates(module)
        >>> for climate, result in results.items():
        ...     print(f"{climate}: CSER={result.cser:.3f}")
    """
    if climate_codes is None:
        climate_codes = ['tropical', 'desert', 'temperate', 'cold', 'marine', 'arctic']

    results = {}
    for code in climate_codes:
        result = run_cser_analysis(code, module)
        results[code] = result

    return results


@lru_cache(maxsize=32)
def get_cached_cser(
    climate_code: str,
    P_stc: float,
    noct: float,
    gamma_pmax: float
) -> float:
    """
    Get cached CSER value for common module/climate combinations.

    Uses LRU cache for performance when repeatedly calculating
    the same combinations.

    Args:
        climate_code: Climate profile code.
        P_stc: Rated power at STC in Watts.
        noct: NOCT in C.
        gamma_pmax: Temperature coefficient of Pmax in %/C.

    Returns:
        CSER rating.
    """
    module = ModuleParameters(
        P_stc=P_stc,
        noct=noct,
        gamma_pmax=gamma_pmax
    )
    result = run_cser_analysis(climate_code, module)
    return result.cser
