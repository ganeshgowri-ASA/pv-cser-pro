"""
Climate module for PV-CSER Pro.

Implements IEC 61853-4 climate profiles and CSER calculations.

Classes:
    ClimateProfile: Standard and custom climate profile definitions
    CSERCalculator: Climate Specific Energy Rating calculator

References:
    IEC 61853-4:2018 - Photovoltaic (PV) module performance testing and
    energy rating - Part 4: Standard reference climatic profiles
"""

from .climate_profiles import (
    ClimateProfile,
    ClimateType,
    IEC_61853_4_PROFILES,
    get_climate_profile,
    list_available_profiles,
)
from .cser_calculator import (
    CSERCalculator,
    CSERResult,
    calculate_cser,
)

__all__ = [
    "ClimateProfile",
    "ClimateType",
    "IEC_61853_4_PROFILES",
    "get_climate_profile",
    "list_available_profiles",
    "CSERCalculator",
    "CSERResult",
    "calculate_cser",
]
