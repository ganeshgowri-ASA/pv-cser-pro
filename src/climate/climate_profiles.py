"""
Climate profiles for IEC 61853-4 calculations.

Defines standard climate profiles according to IEC 61853-4 and
provides functionality for custom climate profile management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClimateProfile:
    """
    Climate profile data container.

    Contains hourly meteorological data for a full year (8760 hours)
    according to IEC 61853-4 specifications.
    """

    # Profile identification
    name: str
    profile_type: str  # 'standard' or 'custom'
    description: str = ""

    # Location data
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0  # m
    timezone: str = "UTC"

    # Hourly data arrays (8760 values each)
    ghi: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # W/m²
    dni: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # W/m²
    dhi: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # W/m²
    ambient_temp: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # °C
    wind_speed: np.ndarray = field(default_factory=lambda: np.zeros(8760))  # m/s

    # Optional spectral data
    spectral_data: Optional[Dict[str, np.ndarray]] = None

    # Climate characteristics
    annual_ghi: float = 0.0  # kWh/m²
    avg_temp: float = 0.0    # °C

    def __post_init__(self):
        """Calculate derived values after initialization."""
        if len(self.ghi) == 8760 and np.sum(self.ghi) > 0:
            self.annual_ghi = np.sum(self.ghi) / 1000  # kWh/m²
            self.avg_temp = np.mean(self.ambient_temp)

    @property
    def poa_irradiance(self) -> np.ndarray:
        """Get plane of array irradiance (simplified: using GHI)."""
        return self.ghi

    def get_monthly_ghi(self) -> List[float]:
        """Calculate monthly GHI values."""
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        monthly = []
        idx = 0
        for hours in hours_per_month:
            monthly.append(np.sum(self.ghi[idx:idx+hours]) / 1000)  # kWh/m²
            idx += hours
        return monthly

    def get_monthly_temp(self) -> List[float]:
        """Calculate monthly average temperatures."""
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        monthly = []
        idx = 0
        for hours in hours_per_month:
            monthly.append(np.mean(self.ambient_temp[idx:idx+hours]))
            idx += hours
        return monthly

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "name": self.name,
            "profile_type": self.profile_type,
            "description": self.description,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "annual_ghi": self.annual_ghi,
            "avg_temp": self.avg_temp,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        ghi: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
    ) -> "ClimateProfile":
        """Create ClimateProfile from dictionary and arrays."""
        if wind_speed is None:
            wind_speed = np.ones(8760)

        return cls(
            name=data.get("name", "Custom"),
            profile_type=data.get("profile_type", "custom"),
            description=data.get("description", ""),
            location=data.get("location", ""),
            latitude=data.get("latitude", 0.0),
            longitude=data.get("longitude", 0.0),
            ghi=ghi,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
        )


def generate_synthetic_profile(
    profile_name: str,
    annual_ghi: float,
    avg_temp: float,
    temp_amplitude: float = 15.0,
) -> ClimateProfile:
    """
    Generate a synthetic climate profile.

    Args:
        profile_name: Name of the profile
        annual_ghi: Annual GHI in kWh/m²
        avg_temp: Annual average temperature in °C
        temp_amplitude: Seasonal temperature variation in °C

    Returns:
        ClimateProfile with synthetic data
    """
    hours = np.arange(8760)

    # Day of year (0-364)
    doy = (hours // 24) % 365

    # Hour of day (0-23)
    hod = hours % 24

    # Solar elevation approximation (simplified)
    solar_noon = 12
    day_length_factor = 1 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365)

    # GHI pattern
    # Peak at solar noon, zero at night
    hour_angle = (hod - solar_noon) / 12 * np.pi
    ghi_daily_pattern = np.maximum(0, np.cos(hour_angle))

    # Seasonal variation
    seasonal_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (doy - 172) / 365)

    # Scale to achieve target annual GHI
    ghi_raw = ghi_daily_pattern * seasonal_factor
    scale_factor = (annual_ghi * 1000) / np.sum(ghi_raw)
    ghi = ghi_raw * scale_factor

    # Temperature pattern
    # Daily cycle with seasonal variation
    temp_daily = -5 * np.cos(2 * np.pi * (hod - 14) / 24)  # Peak at 2PM
    temp_seasonal = temp_amplitude * np.cos(2 * np.pi * (doy - 200) / 365)
    ambient_temp = avg_temp + temp_daily + temp_seasonal

    # Wind speed (simplified pattern)
    wind_speed = 2.0 + 1.5 * np.cos(2 * np.pi * (hod - 14) / 24) + \
                 0.5 * np.random.random(8760)

    return ClimateProfile(
        name=profile_name,
        profile_type="synthetic",
        description=f"Synthetic profile: GHI={annual_ghi} kWh/m², Tavg={avg_temp}°C",
        ghi=ghi,
        ambient_temp=ambient_temp,
        wind_speed=wind_speed,
    )


# IEC 61853-4 Standard Climate Profiles
# These are simplified representations of the 6 standard profiles
IEC_CLIMATE_PROFILES = {
    "Subtropical Arid": {
        "description": "Hot desert climate (e.g., Phoenix, Alice Springs)",
        "annual_ghi": 2400,
        "avg_temp": 25,
        "temp_amplitude": 12,
        "characteristics": {
            "high_irradiance": True,
            "hot_temperatures": True,
            "low_humidity": True,
        }
    },
    "Subtropical Coastal": {
        "description": "Humid subtropical (e.g., Miami, Brisbane)",
        "annual_ghi": 1900,
        "avg_temp": 23,
        "temp_amplitude": 8,
        "characteristics": {
            "moderate_irradiance": True,
            "warm_temperatures": True,
            "high_humidity": True,
        }
    },
    "Tropical Humid": {
        "description": "Tropical rainforest/monsoon (e.g., Singapore, Lagos)",
        "annual_ghi": 1700,
        "avg_temp": 27,
        "temp_amplitude": 3,
        "characteristics": {
            "moderate_irradiance": True,
            "high_temperatures": True,
            "very_high_humidity": True,
        }
    },
    "Temperate Coastal": {
        "description": "Marine west coast (e.g., London, Seattle)",
        "annual_ghi": 1100,
        "avg_temp": 11,
        "temp_amplitude": 8,
        "characteristics": {
            "low_irradiance": True,
            "mild_temperatures": True,
            "high_cloud_cover": True,
        }
    },
    "Temperate Continental": {
        "description": "Humid continental (e.g., Munich, Chicago)",
        "annual_ghi": 1200,
        "avg_temp": 9,
        "temp_amplitude": 18,
        "characteristics": {
            "moderate_irradiance": True,
            "variable_temperatures": True,
            "seasonal_variation": True,
        }
    },
    "High Elevation": {
        "description": "Highland climate (e.g., Denver, La Paz)",
        "annual_ghi": 2000,
        "avg_temp": 10,
        "temp_amplitude": 10,
        "characteristics": {
            "high_irradiance": True,
            "cool_temperatures": True,
            "low_air_mass": True,
        }
    },
}


class ClimateProfileManager:
    """
    Manager for climate profiles.

    Provides methods for loading, creating, and managing
    climate profiles for CSER calculations.
    """

    def __init__(self):
        """Initialize manager with standard profiles."""
        self._profiles: Dict[str, ClimateProfile] = {}
        self._load_standard_profiles()

    def _load_standard_profiles(self) -> None:
        """Load IEC 61853-4 standard profiles."""
        for name, params in IEC_CLIMATE_PROFILES.items():
            profile = generate_synthetic_profile(
                profile_name=name,
                annual_ghi=params["annual_ghi"],
                avg_temp=params["avg_temp"],
                temp_amplitude=params["temp_amplitude"],
            )
            profile.description = params["description"]
            profile.profile_type = "standard"
            self._profiles[name] = profile

    def get_profile(self, name: str) -> Optional[ClimateProfile]:
        """Get a climate profile by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles."""
        return [
            {
                "name": name,
                "type": profile.profile_type,
                "annual_ghi": profile.annual_ghi,
                "avg_temp": profile.avg_temp,
                "description": profile.description,
            }
            for name, profile in self._profiles.items()
        ]

    def get_standard_profiles(self) -> List[str]:
        """Get list of standard profile names."""
        return [
            name for name, profile in self._profiles.items()
            if profile.profile_type == "standard"
        ]

    def add_custom_profile(
        self,
        name: str,
        ghi: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        location: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
    ) -> ClimateProfile:
        """
        Add a custom climate profile.

        Args:
            name: Profile name
            ghi: Hourly GHI array (8760 values)
            ambient_temp: Hourly temperature array (8760 values)
            wind_speed: Hourly wind speed array (8760 values)
            location: Location description
            latitude: Latitude
            longitude: Longitude

        Returns:
            Created ClimateProfile
        """
        if len(ghi) != 8760 or len(ambient_temp) != 8760:
            raise ValueError("Arrays must have 8760 hourly values")

        if wind_speed is None:
            wind_speed = np.ones(8760)

        profile = ClimateProfile(
            name=name,
            profile_type="custom",
            location=location,
            latitude=latitude,
            longitude=longitude,
            ghi=ghi,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
        )

        self._profiles[name] = profile
        return profile

    def compare_profiles(
        self,
        profile_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple climate profiles.

        Args:
            profile_names: List of profile names to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for name in profile_names:
            profile = self._profiles.get(name)
            if profile:
                comparison[name] = {
                    "annual_ghi": profile.annual_ghi,
                    "avg_temp": profile.avg_temp,
                    "peak_ghi": float(np.max(profile.ghi)),
                    "min_temp": float(np.min(profile.ambient_temp)),
                    "max_temp": float(np.max(profile.ambient_temp)),
                }

        return comparison

    def get_profile_for_location(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[str]:
        """
        Suggest a standard profile based on location.

        Args:
            latitude: Latitude
            longitude: Longitude

        Returns:
            Suggested profile name or None
        """
        abs_lat = abs(latitude)

        if abs_lat < 15:
            return "Tropical Humid"
        elif abs_lat < 30:
            if abs_lat < 25:
                return "Subtropical Coastal"
            else:
                return "Subtropical Arid"
        elif abs_lat < 50:
            return "Temperate Continental"
        else:
            return "Temperate Coastal"
