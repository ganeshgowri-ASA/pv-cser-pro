"""
IEC 61853 Constants and Reference Values.

This module contains all standard constants, test conditions, and reference values
defined in the IEC 61853 series of standards for PV module characterization.

Reference:
    IEC 61853-1:2011 - Irradiance and temperature performance measurements
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-4:2018 - Standard reference climatic profiles
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# IEC 61853-1: Power Matrix Test Conditions
# =============================================================================

# Standard irradiance levels for power matrix measurements (W/m²)
# Per IEC 61853-1 Table 1
IRRADIANCES: List[int] = [100, 200, 400, 600, 800, 1000, 1100]

# Standard temperature levels for power matrix measurements (°C)
# Per IEC 61853-1 Table 1
TEMPERATURES: List[int] = [15, 25, 50, 75]

# Number of measurement points in power matrix
MATRIX_ROWS: int = len(IRRADIANCES)  # 7 irradiance levels
MATRIX_COLS: int = len(TEMPERATURES)  # 4 temperature levels
MATRIX_POINTS: int = MATRIX_ROWS * MATRIX_COLS  # 28 total points (22 minimum required)


@dataclass(frozen=True)
class TestCondition:
    """
    Standard test condition definition.

    Attributes:
        name: Full name of the test condition
        abbreviation: Standard abbreviation
        irradiance: Irradiance level in W/m²
        temperature: Cell temperature in °C (or ambient for NOCT)
        description: Brief description of the condition
    """
    name: str
    abbreviation: str
    irradiance: float
    temperature: float
    description: str


# =============================================================================
# Standard Test Conditions (IEC 61853-1)
# =============================================================================

STC = TestCondition(
    name="Standard Test Conditions",
    abbreviation="STC",
    irradiance=1000.0,
    temperature=25.0,
    description="Reference conditions for PV module rating"
)

NOCT = TestCondition(
    name="Nominal Operating Cell Temperature",
    abbreviation="NOCT",
    irradiance=800.0,
    temperature=20.0,  # Ambient temperature; cell temp typically 45-47°C
    description="Conditions for determining NOCT (ambient temp reference)"
)

LIC = TestCondition(
    name="Low Irradiance Conditions",
    abbreviation="LIC",
    irradiance=200.0,
    temperature=25.0,
    description="Performance at low light levels"
)

HTC = TestCondition(
    name="High Temperature Conditions",
    abbreviation="HTC",
    irradiance=1000.0,
    temperature=75.0,
    description="Performance at elevated temperature"
)

LTC = TestCondition(
    name="Low Temperature Conditions",
    abbreviation="LTC",
    irradiance=500.0,
    temperature=15.0,
    description="Performance at reduced temperature"
)

# Dictionary of all standard test conditions
STANDARD_CONDITIONS: Dict[str, TestCondition] = {
    "STC": STC,
    "NOCT": NOCT,
    "LIC": LIC,
    "HTC": HTC,
    "LTC": LTC,
}


# =============================================================================
# Physical Constants
# =============================================================================

# Boltzmann constant (J/K)
BOLTZMANN_CONSTANT: float = 1.380649e-23

# Elementary charge (C)
ELEMENTARY_CHARGE: float = 1.602176634e-19

# Thermal voltage at 25°C (V)
# V_T = k*T/q where T = 298.15 K
THERMAL_VOLTAGE_25C: float = 0.02569  # ~25.69 mV


# =============================================================================
# Typical Temperature Coefficients
# =============================================================================

@dataclass(frozen=True)
class TemperatureCoefficientRanges:
    """
    Typical ranges for temperature coefficients by technology.

    Attributes:
        alpha_isc: Short-circuit current temperature coefficient (%/°C)
        beta_voc: Open-circuit voltage temperature coefficient (%/°C)
        gamma_pmax: Maximum power temperature coefficient (%/°C)
    """
    alpha_isc: Tuple[float, float]  # (min, max)
    beta_voc: Tuple[float, float]
    gamma_pmax: Tuple[float, float]


# Typical coefficient ranges by technology
TEMP_COEFF_MONO_SI = TemperatureCoefficientRanges(
    alpha_isc=(0.03, 0.06),      # %/°C (positive)
    beta_voc=(-0.35, -0.28),     # %/°C (negative)
    gamma_pmax=(-0.50, -0.38)    # %/°C (negative)
)

TEMP_COEFF_POLY_SI = TemperatureCoefficientRanges(
    alpha_isc=(0.03, 0.06),
    beta_voc=(-0.35, -0.30),
    gamma_pmax=(-0.48, -0.40)
)

TEMP_COEFF_CDTE = TemperatureCoefficientRanges(
    alpha_isc=(0.04, 0.08),
    beta_voc=(-0.25, -0.20),
    gamma_pmax=(-0.32, -0.25)
)

TEMP_COEFF_CIGS = TemperatureCoefficientRanges(
    alpha_isc=(0.01, 0.04),
    beta_voc=(-0.35, -0.25),
    gamma_pmax=(-0.45, -0.35)
)

TEMP_COEFFICIENTS_BY_TECHNOLOGY: Dict[str, TemperatureCoefficientRanges] = {
    "Mono-Si": TEMP_COEFF_MONO_SI,
    "Poly-Si": TEMP_COEFF_POLY_SI,
    "CdTe": TEMP_COEFF_CDTE,
    "CIGS": TEMP_COEFF_CIGS,
}


# =============================================================================
# IEC 61853-4: Climate Profiles
# =============================================================================

class ClimateProfile(Enum):
    """
    Standard reference climatic profiles per IEC 61853-4.

    Each profile represents typical conditions for different geographic regions.
    """
    SUBTROPICAL_COASTAL = "subtropical_coastal"
    SUBTROPICAL_ARID = "subtropical_arid"
    TEMPERATE_COASTAL = "temperate_coastal"
    TEMPERATE_CONTINENTAL = "temperate_continental"
    HIGH_ELEVATION = "high_elevation"
    TROPICAL = "tropical"


@dataclass(frozen=True)
class ClimateProfileData:
    """
    Climate profile characteristics.

    Attributes:
        name: Display name
        description: Brief description of climate type
        annual_ghi: Typical annual global horizontal irradiance (kWh/m²)
        avg_temp: Average annual temperature (°C)
        temp_range: Typical temperature range (min, max) in °C
        representative_location: Example location for this climate
    """
    name: str
    description: str
    annual_ghi: Tuple[int, int]  # (min, max) kWh/m²/year
    avg_temp: float
    temp_range: Tuple[float, float]
    representative_location: str


CLIMATE_PROFILES: Dict[ClimateProfile, ClimateProfileData] = {
    ClimateProfile.SUBTROPICAL_COASTAL: ClimateProfileData(
        name="Subtropical Coastal",
        description="Hot, humid coastal regions with moderate irradiance",
        annual_ghi=(1700, 1900),
        avg_temp=24.0,
        temp_range=(15.0, 35.0),
        representative_location="Miami, USA / Brisbane, Australia"
    ),
    ClimateProfile.SUBTROPICAL_ARID: ClimateProfileData(
        name="Subtropical Arid",
        description="Hot, dry desert regions with high irradiance",
        annual_ghi=(2000, 2200),
        avg_temp=26.0,
        temp_range=(10.0, 45.0),
        representative_location="Phoenix, USA / Riyadh, Saudi Arabia"
    ),
    ClimateProfile.TEMPERATE_COASTAL: ClimateProfileData(
        name="Temperate Coastal",
        description="Moderate, humid regions with lower irradiance",
        annual_ghi=(1000, 1200),
        avg_temp=12.0,
        temp_range=(0.0, 25.0),
        representative_location="London, UK / Seattle, USA"
    ),
    ClimateProfile.TEMPERATE_CONTINENTAL: ClimateProfileData(
        name="Temperate Continental",
        description="Moderate regions with seasonal variation",
        annual_ghi=(1200, 1400),
        avg_temp=10.0,
        temp_range=(-10.0, 30.0),
        representative_location="Berlin, Germany / Denver, USA"
    ),
    ClimateProfile.HIGH_ELEVATION: ClimateProfileData(
        name="High Elevation",
        description="Cold, high UV regions at altitude",
        annual_ghi=(1800, 2000),
        avg_temp=8.0,
        temp_range=(-15.0, 25.0),
        representative_location="La Paz, Bolivia / Lhasa, Tibet"
    ),
    ClimateProfile.TROPICAL: ClimateProfileData(
        name="Tropical",
        description="Hot, humid equatorial regions with stable conditions",
        annual_ghi=(1600, 1800),
        avg_temp=27.0,
        temp_range=(22.0, 35.0),
        representative_location="Singapore / Nairobi, Kenya"
    ),
}


# =============================================================================
# Validation Ranges
# =============================================================================

# Valid irradiance range (W/m²)
IRRADIANCE_MIN: float = 0.0
IRRADIANCE_MAX: float = 1500.0

# Valid temperature range (°C)
TEMPERATURE_MIN: float = -40.0
TEMPERATURE_MAX: float = 90.0

# Valid power range (W)
POWER_MIN: float = 0.0
POWER_MAX: float = 1000.0  # Per module; adjust for system

# Valid voltage range (V)
VOLTAGE_MIN: float = 0.0
VOLTAGE_MAX: float = 100.0  # Typical module Voc

# Valid current range (A)
CURRENT_MIN: float = 0.0
CURRENT_MAX: float = 20.0  # Typical module Isc


# =============================================================================
# Interpolation Constants
# =============================================================================

# Number of points for polynomial fitting
POLY_FIT_DEGREE: int = 2  # Quadratic for power vs irradiance

# Ideality factor range for Voc calculation
IDEALITY_FACTOR_MIN: float = 1.0
IDEALITY_FACTOR_MAX: float = 2.0
IDEALITY_FACTOR_TYPICAL: float = 1.3  # Typical for crystalline Si


# =============================================================================
# Display and Formatting
# =============================================================================

# Units for display
UNITS: Dict[str, str] = {
    "irradiance": "W/m²",
    "temperature": "°C",
    "power": "W",
    "voltage": "V",
    "current": "A",
    "efficiency": "%",
    "fill_factor": "%",
    "energy": "kWh",
    "energy_density": "kWh/m²",
}

# Decimal places for formatting
DECIMAL_PLACES: Dict[str, int] = {
    "irradiance": 0,
    "temperature": 1,
    "power": 2,
    "voltage": 3,
    "current": 3,
    "efficiency": 2,
    "fill_factor": 1,
    "coefficient": 4,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_irradiance_array() -> np.ndarray:
    """Return irradiance levels as numpy array."""
    return np.array(IRRADIANCES, dtype=np.float64)


def get_temperature_array() -> np.ndarray:
    """Return temperature levels as numpy array."""
    return np.array(TEMPERATURES, dtype=np.float64)


def create_empty_power_matrix() -> np.ndarray:
    """
    Create an empty power matrix with NaN values.

    Returns:
        np.ndarray: Matrix of shape (7, 4) filled with NaN
    """
    return np.full((MATRIX_ROWS, MATRIX_COLS), np.nan, dtype=np.float64)


def get_climate_profile_names() -> List[str]:
    """Return list of climate profile display names."""
    return [profile.name for profile in CLIMATE_PROFILES.values()]


def get_climate_profile_by_name(name: str) -> ClimateProfile:
    """
    Get ClimateProfile enum by display name.

    Args:
        name: Display name of the climate profile

    Returns:
        ClimateProfile enum value

    Raises:
        ValueError: If name not found
    """
    for profile, data in CLIMATE_PROFILES.items():
        if data.name == name:
            return profile
    raise ValueError(f"Climate profile '{name}' not found")
