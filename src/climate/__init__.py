"""
Climate Module for IEC 61853-3 and IEC 61853-4

This package provides climate-specific energy rating functionality
for photovoltaic modules according to IEC 61853 standards.

Main Components:
    - Climate Profiles (IEC 61853-4): Standard reference climates
    - CSER Calculator (IEC 61853-3): Energy rating calculations
    - Interpolation: Bilinear interpolation for power matrices

Standard Climate Profiles (IEC 61853-4):
    1. Tropical (TRO) - Hot-Humid
    2. Desert (DES) - Hot-Dry
    3. Temperate (TEM) - Moderate
    4. Cold (CLD) - Cold-Moderate
    5. Marine (MAR) - Marine-Moderate
    6. Arctic (ARC) - Very Cold

Example Usage:
    >>> from src.climate import load_standard_profile, run_cser_analysis, ModuleParameters
    >>>
    >>> # Load a climate profile
    >>> climate = load_standard_profile('tropical')
    >>>
    >>> # Define module parameters
    >>> module = ModuleParameters(
    ...     P_stc=400,
    ...     noct=45,
    ...     gamma_pmax=-0.40
    ... )
    >>>
    >>> # Run CSER analysis
    >>> result = run_cser_analysis(climate, module)
    >>> print(f"CSER: {result.cser:.3f}")
    >>> print(f"Annual Energy: {result.annual_energy_kwh:.1f} kWh")

References:
    - IEC 61853-1:2011 - Irradiance and temperature performance
    - IEC 61853-2:2016 - Spectral responsivity, IAM, module temperature
    - IEC 61853-3:2018 - Energy rating of PV modules
    - IEC 61853-4:2018 - Standard reference climatic profiles
"""

# Climate Profiles (IEC 61853-4)
from .climate_profiles import (
    ClimateProfile,
    CustomProfileBuilder,
    IEC_PROFILE_CODES,
    IEC_PROFILE_NAMES,
    IEC_STANDARD_PROFILES,
    get_standard_profiles_metadata,
    load_all_standard_profiles,
    load_standard_profile,
    validate_climate_data,
)

# CSER Calculator (IEC 61853-3)
from .cser_calculator import (
    CSERResult,
    ModuleParameters,
    STC_IRRADIANCE,
    STC_TEMPERATURE,
    apply_iam_correction,
    apply_spectral_correction,
    calculate_annual_energy,
    calculate_capacity_factor,
    calculate_cser_rating,
    calculate_hourly_power,
    calculate_module_temperature,
    calculate_performance_ratio,
    calculate_specific_yield,
    compare_climates,
    get_cached_cser,
    run_cser_analysis,
)

# Interpolation
from .interpolation import (
    InterpolationError,
    bilinear_interpolate,
    create_interpolation_function,
    extrapolate_power,
    interpolate_power_matrix,
    validate_power_matrix,
)

__all__ = [
    # Climate Profiles
    "ClimateProfile",
    "CustomProfileBuilder",
    "IEC_PROFILE_CODES",
    "IEC_PROFILE_NAMES",
    "IEC_STANDARD_PROFILES",
    "get_standard_profiles_metadata",
    "load_all_standard_profiles",
    "load_standard_profile",
    "validate_climate_data",
    # CSER Calculator
    "CSERResult",
    "ModuleParameters",
    "STC_IRRADIANCE",
    "STC_TEMPERATURE",
    "apply_iam_correction",
    "apply_spectral_correction",
    "calculate_annual_energy",
    "calculate_capacity_factor",
    "calculate_cser_rating",
    "calculate_hourly_power",
    "calculate_module_temperature",
    "calculate_performance_ratio",
    "calculate_specific_yield",
    "compare_climates",
    "get_cached_cser",
    "run_cser_analysis",
    # Interpolation
    "InterpolationError",
    "bilinear_interpolate",
    "create_interpolation_function",
    "extrapolate_power",
    "interpolate_power_matrix",
    "validate_power_matrix",
]
