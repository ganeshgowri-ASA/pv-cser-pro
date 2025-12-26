"""
Physical and standard constants for IEC 61853 calculations.

This module defines standard test conditions and reference values
as specified in IEC 61853 series.

References:
    IEC 61853-1:2011 - Irradiance and temperature performance measurements
    IEC 61853-3:2018 - Energy rating of PV modules
    IEC 61853-4:2018 - Standard reference climatic profiles
"""

from typing import Final

# =============================================================================
# Standard Test Conditions (STC) - IEC 61853-1
# =============================================================================

#: Reference irradiance at STC [W/m^2]
STC_IRRADIANCE: Final[float] = 1000.0

#: Reference cell temperature at STC [C]
STC_TEMPERATURE: Final[float] = 25.0

#: Reference air mass for STC spectrum
STC_AIR_MASS: Final[float] = 1.5

# =============================================================================
# Nominal Module Operating Temperature (NMOT) - IEC 61853-2
# =============================================================================

#: NMOT reference irradiance [W/m^2]
NMOT_IRRADIANCE: Final[float] = 800.0

#: NMOT reference ambient temperature [C]
NMOT_AMBIENT_TEMPERATURE: Final[float] = 20.0

#: NMOT reference wind speed [m/s]
NMOT_WIND_SPEED: Final[float] = 1.0

# =============================================================================
# Reference Spectrum - IEC 60904-3
# =============================================================================

#: Standard reference spectrum identifier
REFERENCE_SPECTRUM: Final[str] = "AM1.5G"

#: Total irradiance of reference spectrum [W/m^2]
REFERENCE_SPECTRUM_IRRADIANCE: Final[float] = 1000.0

# =============================================================================
# Energy Rating Calculation Parameters - IEC 61853-3
# =============================================================================

#: Time step for hourly energy calculations [hours]
IEC_TIME_STEPS: Final[float] = 1.0

#: Number of hours in a year for annual energy calculations
HOURS_PER_YEAR: Final[int] = 8760

#: Minimum irradiance threshold for energy calculations [W/m^2]
MIN_IRRADIANCE_THRESHOLD: Final[float] = 0.0

#: Maximum valid irradiance [W/m^2]
MAX_IRRADIANCE: Final[float] = 1400.0

#: Minimum valid cell temperature [C]
MIN_CELL_TEMPERATURE: Final[float] = -40.0

#: Maximum valid cell temperature [C]
MAX_CELL_TEMPERATURE: Final[float] = 90.0

# =============================================================================
# Power Matrix Standard Conditions - IEC 61853-1
# =============================================================================

#: Standard irradiance levels for power matrix [W/m^2]
STANDARD_IRRADIANCE_LEVELS: tuple[float, ...] = (
    100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1100.0
)

#: Standard temperature levels for power matrix [C]
STANDARD_TEMPERATURE_LEVELS: tuple[float, ...] = (
    15.0, 25.0, 50.0, 75.0
)

# =============================================================================
# Climate Profile Parameters - IEC 61853-4
# =============================================================================

#: Number of standard climate profiles in IEC 61853-4
NUM_STANDARD_CLIMATE_PROFILES: Final[int] = 6

#: Climate profile time resolution [minutes]
CLIMATE_PROFILE_TIME_RESOLUTION: Final[int] = 60
