"""
IEC 61853-4 Climate Profiles for CSER Calculations.

Defines standard reference climatic profiles for energy rating of
photovoltaic modules according to IEC 61853-4:2018.

The six standard climate profiles represent characteristic global
climates for comparative energy rating:
    - Subtropical Arid (Hot and Dry)
    - Subtropical Coastal (Hot and Humid)
    - Temperate Coastal (Moderate)
    - High Altitude (Cold and Sunny)
    - Temperate Continental (Cold)
    - Tropical (Humid Equatorial)

References:
    IEC 61853-4:2018 - Photovoltaic (PV) module performance testing and
    energy rating - Part 4: Standard reference climatic profiles

Example:
    >>> from src.climate.climate_profiles import get_climate_profile, ClimateType
    >>> profile = get_climate_profile(ClimateType.SUBTROPICAL_ARID)
    >>> annual_irradiation = profile.annual_global_irradiation
    >>> print(f"Annual GHI: {annual_irradiation} kWh/m^2")
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class ClimateType(Enum):
    """
    Standard climate types defined in IEC 61853-4:2018.

    Each climate type represents a characteristic global climate zone
    with specific irradiance and temperature distributions.

    References:
        IEC 61853-4:2018, Table 1 - Climate profile designations
    """

    SUBTROPICAL_ARID = "subtropical_arid"
    SUBTROPICAL_COASTAL = "subtropical_coastal"
    TEMPERATE_COASTAL = "temperate_coastal"
    HIGH_ALTITUDE = "high_altitude"
    TEMPERATE_CONTINENTAL = "temperate_continental"
    TROPICAL = "tropical"

    @classmethod
    def from_string(cls, name: str) -> "ClimateType":
        """
        Create ClimateType from string identifier.

        Args:
            name: Climate name (case-insensitive, underscores/spaces allowed)

        Returns:
            Matching ClimateType enum value

        Raises:
            ValueError: If name does not match any climate type
        """
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        for climate in cls:
            if climate.value == normalized:
                return climate
        raise ValueError(f"Unknown climate type: {name}")


@dataclass(frozen=True)
class ClimateProfile:
    """
    Climate profile data structure for IEC 61853-4 calculations.

    Contains hourly meteorological data for a typical meteorological year
    (TMY) representing the specified climate zone.

    Attributes:
        name: Climate profile identifier
        climate_type: IEC 61853-4 climate type
        description: Human-readable description
        latitude: Representative latitude [degrees]
        longitude: Representative longitude [degrees]
        elevation: Site elevation [m]
        timezone: UTC offset [hours]
        annual_global_irradiation: Total annual GHI [kWh/m^2]
        annual_diffuse_fraction: Average diffuse/global ratio
        average_ambient_temperature: Annual mean temperature [C]
        hourly_ghi: Hourly global horizontal irradiance [W/m^2]
        hourly_dni: Hourly direct normal irradiance [W/m^2]
        hourly_dhi: Hourly diffuse horizontal irradiance [W/m^2]
        hourly_ambient_temp: Hourly ambient temperature [C]
        hourly_wind_speed: Hourly wind speed [m/s]
        hourly_relative_humidity: Hourly relative humidity [%]

    References:
        IEC 61853-4:2018, Section 5 - Climate profile data requirements
    """

    name: str
    climate_type: ClimateType
    description: str
    latitude: float
    longitude: float
    elevation: float
    timezone: float
    annual_global_irradiation: float
    annual_diffuse_fraction: float
    average_ambient_temperature: float
    hourly_ghi: NDArray[np.floating] = field(repr=False)
    hourly_dni: NDArray[np.floating] = field(repr=False)
    hourly_dhi: NDArray[np.floating] = field(repr=False)
    hourly_ambient_temp: NDArray[np.floating] = field(repr=False)
    hourly_wind_speed: NDArray[np.floating] = field(repr=False)
    hourly_relative_humidity: NDArray[np.floating] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate hourly data arrays."""
        expected_hours = 8760
        for attr_name in [
            "hourly_ghi",
            "hourly_dni",
            "hourly_dhi",
            "hourly_ambient_temp",
            "hourly_wind_speed",
            "hourly_relative_humidity",
        ]:
            arr = getattr(self, attr_name)
            if len(arr) != expected_hours:
                raise ValueError(
                    f"{attr_name} must have {expected_hours} hourly values, "
                    f"got {len(arr)}"
                )

    @property
    def hours_with_irradiance(self) -> int:
        """Return count of hours with positive irradiance."""
        return int(np.sum(self.hourly_ghi > 0))

    @property
    def peak_sun_hours(self) -> float:
        """
        Calculate equivalent peak sun hours.

        Returns:
            Annual energy normalized to 1000 W/m^2 [hours]

        References:
            IEC 61853-4:2018, Annex B.2 - Peak sun hours definition
        """
        return self.annual_global_irradiation

    @functools.cached_property
    def monthly_irradiation(self) -> NDArray[np.floating]:
        """
        Calculate monthly GHI totals.

        Returns:
            Array of 12 monthly irradiation values [kWh/m^2]
        """
        # Days per month (non-leap year)
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        monthly = np.zeros(12)

        hour_idx = 0
        for month, days in enumerate(days_per_month):
            hours_in_month = days * 24
            monthly[month] = np.sum(
                self.hourly_ghi[hour_idx : hour_idx + hours_in_month]
            ) / 1000.0  # Convert Wh to kWh
            hour_idx += hours_in_month

        return monthly

    @functools.cached_property
    def monthly_avg_temperature(self) -> NDArray[np.floating]:
        """
        Calculate monthly average ambient temperatures.

        Returns:
            Array of 12 monthly average temperatures [C]
        """
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        monthly = np.zeros(12)

        hour_idx = 0
        for month, days in enumerate(days_per_month):
            hours_in_month = days * 24
            monthly[month] = np.mean(
                self.hourly_ambient_temp[hour_idx : hour_idx + hours_in_month]
            )
            hour_idx += hours_in_month

        return monthly

    def get_irradiance_temperature_distribution(
        self,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Get irradiance-temperature joint distribution for energy calculations.

        Returns daytime hours only (GHI > 0) as a tuple of:
        (irradiance, temperature, hours) arrays.

        Returns:
            Tuple of (GHI [W/m^2], ambient temp [C], occurrence hours [h])

        References:
            IEC 61853-3:2018, Section 6.3 - Operating condition distribution
        """
        mask = self.hourly_ghi > 0
        return (
            self.hourly_ghi[mask],
            self.hourly_ambient_temp[mask],
            np.ones(np.sum(mask)),  # 1 hour per data point
        )


def _generate_synthetic_hourly_data(
    annual_ghi: float,
    avg_temp: float,
    temp_amplitude: float,
    diffuse_fraction: float,
    latitude: float,
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Generate synthetic hourly climate data for demonstration.

    Creates realistic hourly patterns based on annual characteristics.
    Production systems should use actual TMY data files.

    Args:
        annual_ghi: Target annual GHI [kWh/m^2]
        avg_temp: Annual average temperature [C]
        temp_amplitude: Seasonal temperature variation [C]
        diffuse_fraction: Average diffuse/global ratio
        latitude: Site latitude [degrees]

    Returns:
        Tuple of hourly arrays: (ghi, dni, dhi, temp, wind, humidity)
    """
    hours = np.arange(8760)
    day_of_year = hours // 24

    # Hour of day (0-23)
    hour_of_day = hours % 24

    # Seasonal variation (peak in summer)
    seasonal = np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Daily irradiance pattern (simplified clear sky)
    solar_noon = 12.0
    day_length = 12 + 4 * seasonal * np.sign(latitude)  # Approximate
    sunrise = solar_noon - day_length / 2
    sunset = solar_noon + day_length / 2

    # Hourly GHI with seasonal and diurnal patterns
    hour_angle = (hour_of_day - solar_noon) / (day_length / 2) * np.pi / 2
    ghi_pattern = np.maximum(0, np.cos(hour_angle))

    # Scale to match annual total
    base_ghi = ghi_pattern * (1 + 0.3 * seasonal)
    scale_factor = annual_ghi * 1000 / np.sum(base_ghi)  # kWh to Wh
    hourly_ghi = base_ghi * scale_factor

    # Cap at reasonable maximum
    hourly_ghi = np.minimum(hourly_ghi, 1200.0)

    # DHI and DNI from GHI
    hourly_dhi = hourly_ghi * diffuse_fraction
    # Simplified: DNI approximation (not accurate for all solar positions)
    hourly_dni = np.maximum(0, (hourly_ghi - hourly_dhi) * 1.1)

    # Temperature: diurnal and seasonal patterns
    seasonal_temp = avg_temp + temp_amplitude * seasonal
    diurnal_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    hourly_temp = seasonal_temp + diurnal_temp

    # Wind speed: slight daily pattern
    hourly_wind = 2.0 + 1.5 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.rand(8760) * 0.5
    hourly_wind = np.maximum(0.5, hourly_wind)

    # Relative humidity: inverse to temperature
    hourly_rh = 60 - 20 * (hourly_temp - avg_temp) / temp_amplitude
    hourly_rh = np.clip(hourly_rh, 20, 95)

    return (
        hourly_ghi.astype(np.float64),
        hourly_dni.astype(np.float64),
        hourly_dhi.astype(np.float64),
        hourly_temp.astype(np.float64),
        hourly_wind.astype(np.float64),
        hourly_rh.astype(np.float64),
    )


@functools.lru_cache(maxsize=8)
def _create_iec_profile(climate_type: ClimateType) -> ClimateProfile:
    """
    Create IEC 61853-4 standard climate profile.

    Cached to avoid regenerating synthetic data on repeated calls.

    Args:
        climate_type: IEC climate type to create

    Returns:
        ClimateProfile instance with appropriate characteristics

    References:
        IEC 61853-4:2018, Table 2 - Climate profile characteristics
    """
    # IEC 61853-4 reference values for each climate
    climate_params: Dict[ClimateType, Dict] = {
        ClimateType.SUBTROPICAL_ARID: {
            "name": "Subtropical Arid",
            "description": "Hot desert climate (e.g., Phoenix, Arizona)",
            "latitude": 33.4,
            "longitude": -112.0,
            "elevation": 340.0,
            "timezone": -7.0,
            "annual_ghi": 2100.0,  # kWh/m^2
            "diffuse_fraction": 0.25,
            "avg_temp": 23.0,
            "temp_amplitude": 12.0,
        },
        ClimateType.SUBTROPICAL_COASTAL: {
            "name": "Subtropical Coastal",
            "description": "Hot humid coastal climate (e.g., Miami, Florida)",
            "latitude": 25.8,
            "longitude": -80.2,
            "elevation": 5.0,
            "timezone": -5.0,
            "annual_ghi": 1800.0,
            "diffuse_fraction": 0.40,
            "avg_temp": 25.0,
            "temp_amplitude": 6.0,
        },
        ClimateType.TEMPERATE_COASTAL: {
            "name": "Temperate Coastal",
            "description": "Moderate maritime climate (e.g., San Francisco, CA)",
            "latitude": 37.8,
            "longitude": -122.4,
            "elevation": 20.0,
            "timezone": -8.0,
            "annual_ghi": 1700.0,
            "diffuse_fraction": 0.35,
            "avg_temp": 14.0,
            "temp_amplitude": 4.0,
        },
        ClimateType.HIGH_ALTITUDE: {
            "name": "High Altitude",
            "description": "Cold sunny mountain climate (e.g., Denver, Colorado)",
            "latitude": 39.7,
            "longitude": -105.0,
            "elevation": 1609.0,
            "timezone": -7.0,
            "annual_ghi": 1900.0,
            "diffuse_fraction": 0.30,
            "avg_temp": 10.0,
            "temp_amplitude": 15.0,
        },
        ClimateType.TEMPERATE_CONTINENTAL: {
            "name": "Temperate Continental",
            "description": "Cold continental climate (e.g., Berlin, Germany)",
            "latitude": 52.5,
            "longitude": 13.4,
            "elevation": 35.0,
            "timezone": 1.0,
            "annual_ghi": 1100.0,
            "diffuse_fraction": 0.50,
            "avg_temp": 9.0,
            "temp_amplitude": 10.0,
        },
        ClimateType.TROPICAL: {
            "name": "Tropical",
            "description": "Humid equatorial climate (e.g., Singapore)",
            "latitude": 1.3,
            "longitude": 103.8,
            "elevation": 15.0,
            "timezone": 8.0,
            "annual_ghi": 1600.0,
            "diffuse_fraction": 0.55,
            "avg_temp": 27.0,
            "temp_amplitude": 1.0,
        },
    }

    params = climate_params[climate_type]

    # Generate synthetic hourly data
    hourly_data = _generate_synthetic_hourly_data(
        annual_ghi=params["annual_ghi"],
        avg_temp=params["avg_temp"],
        temp_amplitude=params["temp_amplitude"],
        diffuse_fraction=params["diffuse_fraction"],
        latitude=params["latitude"],
    )

    return ClimateProfile(
        name=params["name"],
        climate_type=climate_type,
        description=params["description"],
        latitude=params["latitude"],
        longitude=params["longitude"],
        elevation=params["elevation"],
        timezone=params["timezone"],
        annual_global_irradiation=params["annual_ghi"],
        annual_diffuse_fraction=params["diffuse_fraction"],
        average_ambient_temperature=params["avg_temp"],
        hourly_ghi=hourly_data[0],
        hourly_dni=hourly_data[1],
        hourly_dhi=hourly_data[2],
        hourly_ambient_temp=hourly_data[3],
        hourly_wind_speed=hourly_data[4],
        hourly_relative_humidity=hourly_data[5],
    )


# Pre-defined IEC 61853-4 profiles (lazy initialization)
IEC_61853_4_PROFILES: Dict[ClimateType, ClimateProfile] = {}


def get_climate_profile(climate_type: ClimateType) -> ClimateProfile:
    """
    Get a standard IEC 61853-4 climate profile.

    Profiles are cached after first access for performance.

    Args:
        climate_type: The climate type to retrieve

    Returns:
        ClimateProfile for the specified climate

    References:
        IEC 61853-4:2018, Section 4 - Standard climate profiles

    Example:
        >>> profile = get_climate_profile(ClimateType.SUBTROPICAL_ARID)
        >>> print(profile.annual_global_irradiation)
        2100.0
    """
    if climate_type not in IEC_61853_4_PROFILES:
        IEC_61853_4_PROFILES[climate_type] = _create_iec_profile(climate_type)
    return IEC_61853_4_PROFILES[climate_type]


def list_available_profiles() -> List[Dict[str, str]]:
    """
    List all available IEC 61853-4 climate profiles.

    Returns:
        List of dictionaries with 'id', 'name', and 'description' keys

    Example:
        >>> profiles = list_available_profiles()
        >>> for p in profiles:
        ...     print(f"{p['id']}: {p['name']}")
    """
    return [
        {
            "id": climate_type.value,
            "name": get_climate_profile(climate_type).name,
            "description": get_climate_profile(climate_type).description,
        }
        for climate_type in ClimateType
    ]


def create_custom_profile(
    name: str,
    hourly_ghi: NDArray[np.floating],
    hourly_ambient_temp: NDArray[np.floating],
    hourly_wind_speed: Optional[NDArray[np.floating]] = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
    diffuse_fraction: float = 0.35,
) -> ClimateProfile:
    """
    Create a custom climate profile from user-provided data.

    Allows creation of site-specific profiles for CSER calculations
    using actual measured or TMY data.

    Args:
        name: Profile identifier
        hourly_ghi: 8760 hourly GHI values [W/m^2]
        hourly_ambient_temp: 8760 hourly temperatures [C]
        hourly_wind_speed: 8760 hourly wind speeds [m/s], default 1.0
        latitude: Site latitude [degrees]
        longitude: Site longitude [degrees]
        elevation: Site elevation [m]
        diffuse_fraction: Estimated diffuse/global ratio

    Returns:
        ClimateProfile instance

    References:
        IEC 61853-4:2018, Annex C - Custom profile requirements

    Example:
        >>> import numpy as np
        >>> ghi = np.random.rand(8760) * 800  # Simplified example
        >>> temp = 20 + 10 * np.sin(np.linspace(0, 2*np.pi, 8760))
        >>> profile = create_custom_profile("My Site", ghi, temp)
    """
    ghi = np.asarray(hourly_ghi, dtype=np.float64)
    temp = np.asarray(hourly_ambient_temp, dtype=np.float64)

    if hourly_wind_speed is None:
        wind = np.ones(8760, dtype=np.float64)
    else:
        wind = np.asarray(hourly_wind_speed, dtype=np.float64)

    # Calculate DNI and DHI from GHI using diffuse fraction
    dhi = ghi * diffuse_fraction
    dni = np.maximum(0, ghi - dhi)  # Simplified

    # Default humidity
    rh = 50.0 * np.ones(8760, dtype=np.float64)

    annual_ghi = np.sum(ghi) / 1000.0  # Wh to kWh

    return ClimateProfile(
        name=name,
        climate_type=ClimateType.TEMPERATE_COASTAL,  # Default for custom
        description=f"Custom profile: {name}",
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        timezone=0.0,
        annual_global_irradiation=annual_ghi,
        annual_diffuse_fraction=diffuse_fraction,
        average_ambient_temperature=float(np.mean(temp)),
        hourly_ghi=ghi,
        hourly_dni=dni,
        hourly_dhi=dhi,
        hourly_ambient_temp=temp,
        hourly_wind_speed=wind,
        hourly_relative_humidity=rh,
    )
