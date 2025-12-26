"""
Climate Profiles Module for IEC 61853-4 Standard Reference Climates

This module implements the six standard climate profiles defined in
IEC 61853-4:2018 for photovoltaic module energy rating.

The standard profiles represent typical climatic conditions found
globally and are used as reference conditions for CSER calculations.

Standard Profiles (IEC 61853-4):
    1. Tropical (Hot-Humid) - High irradiance, high humidity
    2. Desert (Hot-Dry) - Very high irradiance, low humidity
    3. Temperate (Moderate) - Moderate conditions, four seasons
    4. Cold (Cold-Moderate) - Low irradiance, cold temperatures
    5. Marine (Marine-Moderate) - Coastal, moderate conditions
    6. Arctic (Very Cold) - Very low temperatures, snow reflection

References:
    - IEC 61853-4:2018 - Standard reference climatic profiles
    - IEC 61853-3:2018 - Energy rating of PV modules
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Path to standard profiles JSON
_PROFILES_PATH = Path(__file__).parent.parent.parent / "data" / "climate_profiles" / "iec61853_4_standards.json"


@dataclass
class ClimateProfile:
    """
    Represents a climate profile for CSER calculations.

    This class encapsulates all climate data needed for IEC 61853-3
    energy rating calculations, including hourly irradiance and
    temperature distributions.

    Attributes:
        code: Short identifier (e.g., 'TRO', 'DES', 'TEM').
        name: Full descriptive name.
        description: Detailed description of the climate.
        annual_irradiation: Total annual irradiation in kWh/m^2.
        average_ambient_temperature: Mean ambient temperature in C.
        average_module_temperature: Mean module temperature in C.
        average_wind_speed: Mean wind speed in m/s.
        average_relative_humidity: Mean relative humidity in %.
        hourly_irradiance: 8760-element array of hourly irradiance (W/m^2).
        hourly_temperature: 8760-element array of hourly temperature (C).
        irradiance_distribution: Dict of irradiance level probabilities.
        temperature_distribution: Dict of temperature level probabilities.
        spectral_characteristics: Dict of spectral parameters.
        location_example: Example locations for this climate.
    """
    code: str
    name: str
    description: str = ""
    annual_irradiation: float = 0.0  # kWh/m^2
    average_ambient_temperature: float = 25.0  # C
    average_module_temperature: float = 40.0  # C
    average_wind_speed: float = 2.0  # m/s
    average_relative_humidity: float = 50.0  # %
    hourly_irradiance: NDArray[np.float64] = field(default_factory=lambda: np.zeros(8760))
    hourly_temperature: NDArray[np.float64] = field(default_factory=lambda: np.ones(8760) * 25.0)
    irradiance_distribution: Dict[str, float] = field(default_factory=dict)
    temperature_distribution: Dict[str, float] = field(default_factory=dict)
    spectral_characteristics: Dict[str, float] = field(default_factory=dict)
    location_example: str = ""

    def __post_init__(self):
        """Validate and convert arrays after initialization."""
        self.hourly_irradiance = np.asarray(self.hourly_irradiance, dtype=np.float64)
        self.hourly_temperature = np.asarray(self.hourly_temperature, dtype=np.float64)

        if len(self.hourly_irradiance) != 8760:
            # Generate synthetic data if not provided
            self.hourly_irradiance = self._generate_hourly_irradiance()
        if len(self.hourly_temperature) != 8760:
            self.hourly_temperature = self._generate_hourly_temperature()

    def _generate_hourly_irradiance(self) -> NDArray[np.float64]:
        """
        Generate synthetic hourly irradiance from distribution.

        Uses the irradiance distribution to create a realistic
        8760-hour irradiance profile.
        """
        if not self.irradiance_distribution:
            # Default solar day pattern
            return _generate_default_irradiance(self.annual_irradiation)

        return _generate_irradiance_from_distribution(
            self.irradiance_distribution,
            self.annual_irradiation
        )

    def _generate_hourly_temperature(self) -> NDArray[np.float64]:
        """
        Generate synthetic hourly temperature from distribution.

        Uses the temperature distribution to create a realistic
        8760-hour temperature profile.
        """
        if not self.temperature_distribution:
            # Default seasonal pattern
            return _generate_default_temperature(self.average_ambient_temperature)

        return _generate_temperature_from_distribution(
            self.temperature_distribution,
            self.average_ambient_temperature
        )

    def get_annual_energy_potential(self) -> float:
        """
        Calculate the annual irradiation from hourly data.

        Returns:
            Annual irradiation in kWh/m^2.
        """
        return float(np.sum(self.hourly_irradiance) / 1000.0)

    def get_daytime_hours(self, threshold: float = 50.0) -> int:
        """
        Count hours with irradiance above threshold.

        Args:
            threshold: Minimum irradiance to count as daytime (W/m^2).

        Returns:
            Number of hours with irradiance above threshold.
        """
        return int(np.sum(self.hourly_irradiance > threshold))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert hourly data to a pandas DataFrame.

        Returns:
            DataFrame with columns for hour, irradiance, and temperature.
        """
        return pd.DataFrame({
            'hour': range(8760),
            'month': [(h // 730) + 1 for h in range(8760)],
            'irradiance_wm2': self.hourly_irradiance,
            'ambient_temperature_c': self.hourly_temperature
        })

    def to_dict(self) -> Dict:
        """
        Convert profile to dictionary for serialization.

        Returns:
            Dictionary representation of the profile.
        """
        return {
            'code': self.code,
            'name': self.name,
            'description': self.description,
            'annual_irradiation': self.annual_irradiation,
            'average_ambient_temperature': self.average_ambient_temperature,
            'average_module_temperature': self.average_module_temperature,
            'average_wind_speed': self.average_wind_speed,
            'average_relative_humidity': self.average_relative_humidity,
            'irradiance_distribution': self.irradiance_distribution,
            'temperature_distribution': self.temperature_distribution,
            'spectral_characteristics': self.spectral_characteristics,
            'location_example': self.location_example
        }


# Standard IEC 61853-4 profile codes
IEC_PROFILE_CODES = {
    'tropical': 'TRO',
    'desert': 'DES',
    'temperate': 'TEM',
    'cold': 'CLD',
    'marine': 'MAR',
    'arctic': 'ARC'
}

# Reverse mapping
IEC_PROFILE_NAMES = {v: k for k, v in IEC_PROFILE_CODES.items()}


@lru_cache(maxsize=1)
def _load_profiles_json() -> Dict:
    """
    Load the standard profiles JSON file with caching.

    Returns:
        Dictionary of profile data.

    Raises:
        FileNotFoundError: If the profiles JSON file is not found.
    """
    if not _PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Standard profiles file not found: {_PROFILES_PATH}"
        )

    with open(_PROFILES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_standard_profiles_metadata() -> Dict[str, Dict]:
    """
    Get metadata for all standard IEC 61853-4 profiles.

    Returns:
        Dictionary mapping profile keys to their metadata.

    Example:
        >>> metadata = get_standard_profiles_metadata()
        >>> print(metadata['tropical']['name'])
        'Tropical Hot-Humid'
    """
    data = _load_profiles_json()
    return data.get('profiles', {})


def load_standard_profile(profile_code: str) -> ClimateProfile:
    """
    Load a standard IEC 61853-4 climate profile.

    Args:
        profile_code: Profile identifier. Can be:
            - Full name: 'tropical', 'desert', 'temperate', 'cold', 'marine', 'arctic'
            - Short code: 'TRO', 'DES', 'TEM', 'CLD', 'MAR', 'ARC'

    Returns:
        ClimateProfile object with 8760 hourly data points.

    Raises:
        ValueError: If profile_code is not recognized.
        FileNotFoundError: If profiles data file is not found.

    Example:
        >>> profile = load_standard_profile('tropical')
        >>> print(profile.annual_irradiation)
        2100
        >>> print(len(profile.hourly_irradiance))
        8760

    References:
        IEC 61853-4:2018, Annex A - Standard reference climatic profiles
    """
    # Normalize profile code
    code_lower = profile_code.lower()

    # Check if it's a short code
    if profile_code.upper() in IEC_PROFILE_NAMES:
        code_lower = IEC_PROFILE_NAMES[profile_code.upper()]

    # Load profiles data
    data = _load_profiles_json()
    profiles = data.get('profiles', {})

    if code_lower not in profiles:
        valid_codes = list(profiles.keys()) + list(IEC_PROFILE_CODES.values())
        raise ValueError(
            f"Unknown profile code: '{profile_code}'. "
            f"Valid codes: {valid_codes}"
        )

    profile_data = profiles[code_lower]

    # Create ClimateProfile
    return ClimateProfile(
        code=profile_data.get('code', IEC_PROFILE_CODES.get(code_lower, code_lower.upper())),
        name=profile_data.get('name', code_lower.title()),
        description=profile_data.get('description', ''),
        annual_irradiation=profile_data.get('annual_irradiation', 1000),
        average_ambient_temperature=profile_data.get('average_ambient_temperature', 25),
        average_module_temperature=profile_data.get('average_module_temperature', 40),
        average_wind_speed=profile_data.get('average_wind_speed', 2),
        average_relative_humidity=profile_data.get('average_relative_humidity', 50),
        irradiance_distribution=profile_data.get('irradiance_distribution', {}),
        temperature_distribution=profile_data.get('temperature_distribution', {}),
        spectral_characteristics=profile_data.get('spectral_characteristics', {}),
        location_example=profile_data.get('location_example', '')
    )


def load_all_standard_profiles() -> Dict[str, ClimateProfile]:
    """
    Load all six standard IEC 61853-4 climate profiles.

    Returns:
        Dictionary mapping profile names to ClimateProfile objects.

    Example:
        >>> profiles = load_all_standard_profiles()
        >>> for name, profile in profiles.items():
        ...     print(f"{name}: {profile.annual_irradiation} kWh/m^2")
    """
    profiles = {}
    for name in IEC_PROFILE_CODES.keys():
        profiles[name] = load_standard_profile(name)
    return profiles


def validate_climate_data(
    hourly_irradiance: NDArray[np.float64],
    hourly_temperature: NDArray[np.float64],
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate hourly climate data for CSER calculations.

    Checks that the climate data meets the requirements for
    IEC 61853-3 energy rating calculations.

    Args:
        hourly_irradiance: 8760-element array of irradiance (W/m^2).
        hourly_temperature: 8760-element array of temperature (C).
        strict: If True, apply stricter validation rules.

    Returns:
        Tuple of (is_valid, list_of_warnings_or_errors).

    Example:
        >>> G = np.random.uniform(0, 1000, 8760)
        >>> T = np.random.uniform(10, 40, 8760)
        >>> valid, messages = validate_climate_data(G, T)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check array lengths
    if len(hourly_irradiance) != 8760:
        errors.append(
            f"Irradiance array must have 8760 elements, got {len(hourly_irradiance)}"
        )
    if len(hourly_temperature) != 8760:
        errors.append(
            f"Temperature array must have 8760 elements, got {len(hourly_temperature)}"
        )

    if errors:
        return False, errors

    # Check for NaN values
    nan_G = np.sum(np.isnan(hourly_irradiance))
    nan_T = np.sum(np.isnan(hourly_temperature))
    if nan_G > 0:
        errors.append(f"Irradiance contains {nan_G} NaN values")
    if nan_T > 0:
        errors.append(f"Temperature contains {nan_T} NaN values")

    # Check irradiance range
    G_min = np.min(hourly_irradiance)
    G_max = np.max(hourly_irradiance)
    if G_min < 0:
        errors.append(f"Irradiance has negative values (min={G_min:.1f})")
    if G_max > 1500:
        if strict:
            errors.append(f"Irradiance exceeds 1500 W/m^2 (max={G_max:.1f})")
        else:
            warnings.append(f"Very high irradiance detected (max={G_max:.1f} W/m^2)")

    # Check temperature range
    T_min = np.min(hourly_temperature)
    T_max = np.max(hourly_temperature)
    if T_min < -50:
        if strict:
            errors.append(f"Temperature below -50 C (min={T_min:.1f})")
        else:
            warnings.append(f"Very low temperature detected (min={T_min:.1f} C)")
    if T_max > 60:
        if strict:
            errors.append(f"Temperature exceeds 60 C (max={T_max:.1f})")
        else:
            warnings.append(f"Very high temperature detected (max={T_max:.1f} C)")

    # Check for reasonable annual irradiation
    annual_kwh = np.sum(hourly_irradiance) / 1000
    if annual_kwh < 500:
        warnings.append(f"Very low annual irradiation ({annual_kwh:.0f} kWh/m^2)")
    elif annual_kwh > 3000:
        warnings.append(f"Very high annual irradiation ({annual_kwh:.0f} kWh/m^2)")

    # Check for daylight hours
    daylight_hours = np.sum(hourly_irradiance > 50)
    if daylight_hours < 2000:
        warnings.append(f"Low daylight hours ({daylight_hours})")
    elif daylight_hours > 5000:
        warnings.append(f"High daylight hours ({daylight_hours})")

    is_valid = len(errors) == 0
    messages = errors + warnings

    return is_valid, messages


class CustomProfileBuilder:
    """
    Builder class for creating custom climate profiles.

    Provides a fluent interface for constructing ClimateProfile objects
    with custom parameters and hourly data.

    Example:
        >>> builder = CustomProfileBuilder()
        >>> profile = (builder
        ...     .set_name("My Location")
        ...     .set_annual_irradiation(1800)
        ...     .set_average_temperature(22)
        ...     .from_monthly_averages(monthly_G, monthly_T)
        ...     .build())
    """

    def __init__(self):
        """Initialize the builder with default values."""
        self._code = "CUS"
        self._name = "Custom Profile"
        self._description = "User-defined climate profile"
        self._annual_irradiation = 1500.0
        self._avg_temp = 20.0
        self._avg_module_temp = 35.0
        self._wind_speed = 2.0
        self._humidity = 50.0
        self._hourly_G: Optional[NDArray[np.float64]] = None
        self._hourly_T: Optional[NDArray[np.float64]] = None
        self._location = ""

    def set_code(self, code: str) -> 'CustomProfileBuilder':
        """Set the profile code."""
        self._code = code
        return self

    def set_name(self, name: str) -> 'CustomProfileBuilder':
        """Set the profile name."""
        self._name = name
        return self

    def set_description(self, description: str) -> 'CustomProfileBuilder':
        """Set the profile description."""
        self._description = description
        return self

    def set_location(self, location: str) -> 'CustomProfileBuilder':
        """Set example location."""
        self._location = location
        return self

    def set_annual_irradiation(self, kwh_per_m2: float) -> 'CustomProfileBuilder':
        """Set annual irradiation in kWh/m^2."""
        self._annual_irradiation = kwh_per_m2
        return self

    def set_average_temperature(self, temp_c: float) -> 'CustomProfileBuilder':
        """Set average ambient temperature in C."""
        self._avg_temp = temp_c
        return self

    def set_average_module_temperature(self, temp_c: float) -> 'CustomProfileBuilder':
        """Set average module temperature in C."""
        self._avg_module_temp = temp_c
        return self

    def set_wind_speed(self, speed_ms: float) -> 'CustomProfileBuilder':
        """Set average wind speed in m/s."""
        self._wind_speed = speed_ms
        return self

    def set_humidity(self, humidity_pct: float) -> 'CustomProfileBuilder':
        """Set average relative humidity in %."""
        self._humidity = humidity_pct
        return self

    def set_hourly_data(
        self,
        irradiance: NDArray[np.float64],
        temperature: NDArray[np.float64]
    ) -> 'CustomProfileBuilder':
        """
        Set hourly irradiance and temperature arrays directly.

        Args:
            irradiance: 8760-element array of hourly irradiance (W/m^2).
            temperature: 8760-element array of hourly temperature (C).
        """
        if len(irradiance) != 8760:
            raise ValueError(f"Irradiance must have 8760 elements, got {len(irradiance)}")
        if len(temperature) != 8760:
            raise ValueError(f"Temperature must have 8760 elements, got {len(temperature)}")

        self._hourly_G = np.asarray(irradiance, dtype=np.float64)
        self._hourly_T = np.asarray(temperature, dtype=np.float64)
        self._annual_irradiation = np.sum(self._hourly_G) / 1000.0
        self._avg_temp = np.mean(self._hourly_T)

        return self

    def from_monthly_averages(
        self,
        monthly_irradiance: List[float],
        monthly_temperature: List[float]
    ) -> 'CustomProfileBuilder':
        """
        Generate hourly data from monthly average values.

        Args:
            monthly_irradiance: 12-element list of monthly average daily irradiance (Wh/m^2/day).
            monthly_temperature: 12-element list of monthly average temperature (C).

        Returns:
            Self for chaining.
        """
        if len(monthly_irradiance) != 12:
            raise ValueError("monthly_irradiance must have 12 elements")
        if len(monthly_temperature) != 12:
            raise ValueError("monthly_temperature must have 12 elements")

        self._hourly_G = _generate_hourly_from_monthly(monthly_irradiance, 'irradiance')
        self._hourly_T = _generate_hourly_from_monthly(monthly_temperature, 'temperature')
        self._annual_irradiation = np.sum(self._hourly_G) / 1000.0
        self._avg_temp = np.mean(self._hourly_T)

        return self

    def from_tmy_file(self, filepath: Union[str, Path]) -> 'CustomProfileBuilder':
        """
        Load hourly data from a TMY (Typical Meteorological Year) file.

        Supports CSV files with columns for GHI and temperature.

        Args:
            filepath: Path to TMY CSV file.

        Returns:
            Self for chaining.
        """
        df = pd.read_csv(filepath)

        # Try common column names
        ghi_cols = ['GHI', 'ghi', 'irradiance', 'Irradiance', 'G', 'solar_radiation']
        temp_cols = ['Temperature', 'temperature', 'temp', 'Temp', 'T', 'air_temp', 'ambient_temp']

        ghi_col = None
        temp_col = None

        for col in ghi_cols:
            if col in df.columns:
                ghi_col = col
                break

        for col in temp_cols:
            if col in df.columns:
                temp_col = col
                break

        if ghi_col is None:
            raise ValueError(f"Could not find irradiance column. Tried: {ghi_cols}")
        if temp_col is None:
            raise ValueError(f"Could not find temperature column. Tried: {temp_cols}")

        # Extract data
        ghi = df[ghi_col].values[:8760]
        temp = df[temp_col].values[:8760]

        if len(ghi) < 8760:
            raise ValueError(f"TMY file has only {len(ghi)} rows, need 8760")

        self._hourly_G = np.asarray(ghi, dtype=np.float64)
        self._hourly_T = np.asarray(temp, dtype=np.float64)
        self._annual_irradiation = np.sum(self._hourly_G) / 1000.0
        self._avg_temp = np.mean(self._hourly_T)

        return self

    def build(self) -> ClimateProfile:
        """
        Build and return the ClimateProfile.

        Returns:
            Constructed ClimateProfile object.

        Raises:
            ValueError: If validation fails.
        """
        # Generate hourly data if not set
        if self._hourly_G is None:
            self._hourly_G = _generate_default_irradiance(self._annual_irradiation)
        if self._hourly_T is None:
            self._hourly_T = _generate_default_temperature(self._avg_temp)

        # Validate
        is_valid, messages = validate_climate_data(
            self._hourly_G, self._hourly_T, strict=False
        )

        if not is_valid:
            raise ValueError(f"Invalid climate data: {messages}")

        return ClimateProfile(
            code=self._code,
            name=self._name,
            description=self._description,
            annual_irradiation=self._annual_irradiation,
            average_ambient_temperature=self._avg_temp,
            average_module_temperature=self._avg_module_temp,
            average_wind_speed=self._wind_speed,
            average_relative_humidity=self._humidity,
            hourly_irradiance=self._hourly_G,
            hourly_temperature=self._hourly_T,
            location_example=self._location
        )


def _generate_default_irradiance(annual_kwh: float) -> NDArray[np.float64]:
    """
    Generate synthetic hourly irradiance data.

    Creates a realistic solar irradiance profile based on
    typical diurnal and seasonal patterns.

    Args:
        annual_kwh: Target annual irradiation in kWh/m^2.

    Returns:
        8760-element array of hourly irradiance values (W/m^2).
    """
    hours = np.arange(8760)

    # Day of year (0-364)
    doy = hours // 24

    # Hour of day (0-23)
    hod = hours % 24

    # Seasonal factor (1 in summer, 0.3 in winter for northern hemisphere)
    seasonal = 0.65 + 0.35 * np.cos(2 * np.pi * (doy - 172) / 365)

    # Diurnal factor (peak at solar noon ~12:00)
    diurnal = np.maximum(0, np.sin(np.pi * (hod - 6) / 12))

    # Combine factors
    pattern = seasonal * diurnal

    # Scale to match target annual irradiation
    current_annual = np.sum(pattern) / 1000
    if current_annual > 0:
        scale = annual_kwh / current_annual
    else:
        scale = 1.0

    irradiance = pattern * scale * 1000 / np.max(pattern) if np.max(pattern) > 0 else pattern

    # Cap at reasonable maximum
    irradiance = np.clip(irradiance, 0, 1200)

    # Rescale to hit target
    current = np.sum(irradiance) / 1000
    if current > 0:
        irradiance *= annual_kwh / current

    return irradiance.astype(np.float64)


def _generate_default_temperature(avg_temp: float) -> NDArray[np.float64]:
    """
    Generate synthetic hourly temperature data.

    Creates a realistic temperature profile with diurnal
    and seasonal variations.

    Args:
        avg_temp: Target average temperature in C.

    Returns:
        8760-element array of hourly temperature values (C).
    """
    hours = np.arange(8760)

    # Day of year
    doy = hours // 24

    # Hour of day
    hod = hours % 24

    # Seasonal variation (+/- 10 C)
    seasonal = 10 * np.cos(2 * np.pi * (doy - 172) / 365)

    # Diurnal variation (+/- 5 C, peak at 14:00)
    diurnal = 5 * np.sin(2 * np.pi * (hod - 8) / 24)

    # Combine
    temperature = avg_temp + seasonal + diurnal

    return temperature.astype(np.float64)


def _generate_irradiance_from_distribution(
    distribution: Dict[str, float],
    annual_kwh: float
) -> NDArray[np.float64]:
    """
    Generate hourly irradiance from an irradiance distribution.

    Args:
        distribution: Dict mapping irradiance levels to probabilities.
        annual_kwh: Target annual irradiation in kWh/m^2.

    Returns:
        8760-element array of hourly irradiance values.
    """
    # Parse distribution
    levels = []
    probs = []

    for key, prob in distribution.items():
        # Extract irradiance level from key (e.g., "G_800" -> 800)
        level = float(key.replace('G_', '').replace('g_', ''))
        levels.append(level)
        probs.append(prob)

    levels = np.array(levels)
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize

    # Start with default pattern
    base = _generate_default_irradiance(annual_kwh)

    # Adjust distribution
    # This is a simplified approach - in practice, more sophisticated
    # methods would be used to match the exact distribution
    daylight_mask = base > 50
    daylight_hours = np.sum(daylight_mask)

    if daylight_hours > 0:
        # Assign irradiance levels based on probability
        irradiance = np.zeros(8760)
        daylight_indices = np.where(daylight_mask)[0]
        np.random.shuffle(daylight_indices)

        idx = 0
        for level, prob in zip(levels, probs):
            count = int(prob * daylight_hours)
            irradiance[daylight_indices[idx:idx+count]] = level
            idx += count

        # Fill remaining with weighted average
        if idx < len(daylight_indices):
            avg_level = np.average(levels, weights=probs)
            irradiance[daylight_indices[idx:]] = avg_level

        # Rescale to match annual target
        current = np.sum(irradiance) / 1000
        if current > 0:
            irradiance *= annual_kwh / current

        return irradiance

    return base


def _generate_temperature_from_distribution(
    distribution: Dict[str, float],
    avg_temp: float
) -> NDArray[np.float64]:
    """
    Generate hourly temperature from a temperature distribution.

    Args:
        distribution: Dict mapping temperature levels to probabilities.
        avg_temp: Target average temperature in C.

    Returns:
        8760-element array of hourly temperature values.
    """
    # Parse distribution
    levels = []
    probs = []

    for key, prob in distribution.items():
        # Extract temperature from key (e.g., "T_25" -> 25, "T_minus5" -> -5)
        temp_str = key.replace('T_', '').replace('t_', '')
        temp_str = temp_str.replace('minus', '-')
        level = float(temp_str)
        levels.append(level)
        probs.append(prob)

    levels = np.array(levels)
    probs = np.array(probs)
    probs = probs / np.sum(probs)

    # Generate base temperature pattern
    base = _generate_default_temperature(avg_temp)

    # Adjust to match distribution
    # Simple approach: scale and shift to match distribution range
    temp_range = np.max(levels) - np.min(levels)
    base_range = np.max(base) - np.min(base)

    if base_range > 0:
        scale = temp_range / base_range
        temperature = (base - np.mean(base)) * scale + avg_temp
    else:
        temperature = base

    return temperature.astype(np.float64)


def _generate_hourly_from_monthly(
    monthly_values: List[float],
    data_type: str
) -> NDArray[np.float64]:
    """
    Generate hourly data from monthly averages.

    Args:
        monthly_values: 12 monthly average values.
        data_type: 'irradiance' or 'temperature'.

    Returns:
        8760-element hourly array.
    """
    # Days per month
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    hourly = np.zeros(8760)
    hour_idx = 0

    for month, (days, value) in enumerate(zip(days_per_month, monthly_values)):
        hours_in_month = days * 24

        if data_type == 'irradiance':
            # For irradiance, create diurnal pattern
            for day in range(days):
                for hour in range(24):
                    # Simple diurnal pattern
                    if 6 <= hour <= 18:
                        factor = np.sin(np.pi * (hour - 6) / 12)
                        hourly[hour_idx] = value * factor / 0.636  # Normalize
                    else:
                        hourly[hour_idx] = 0
                    hour_idx += 1
        else:
            # For temperature, create diurnal variation
            for day in range(days):
                for hour in range(24):
                    diurnal = 5 * np.sin(2 * np.pi * (hour - 8) / 24)
                    hourly[hour_idx] = value + diurnal
                    hour_idx += 1

    return hourly[:8760]


# Dictionary of standard profiles for quick access
IEC_STANDARD_PROFILES: Dict[str, Dict] = {
    'tropical': {
        'code': 'TRO',
        'name': 'Tropical Hot-Humid',
        'annual_irradiation': 2100,
        'avg_temp': 28,
        'description': 'IEC 61853-4 Profile 1'
    },
    'desert': {
        'code': 'DES',
        'name': 'Desert Hot-Dry',
        'annual_irradiation': 2400,
        'avg_temp': 30,
        'description': 'IEC 61853-4 Profile 2'
    },
    'temperate': {
        'code': 'TEM',
        'name': 'Temperate Moderate',
        'annual_irradiation': 1200,
        'avg_temp': 12,
        'description': 'IEC 61853-4 Profile 3'
    },
    'cold': {
        'code': 'CLD',
        'name': 'Cold Moderate',
        'annual_irradiation': 1000,
        'avg_temp': 5,
        'description': 'IEC 61853-4 Profile 4'
    },
    'marine': {
        'code': 'MAR',
        'name': 'Marine Moderate',
        'annual_irradiation': 1400,
        'avg_temp': 15,
        'description': 'IEC 61853-4 Profile 5'
    },
    'arctic': {
        'code': 'ARC',
        'name': 'Arctic Very Cold',
        'annual_irradiation': 800,
        'avg_temp': -5,
        'description': 'IEC 61853-4 Profile 6'
    }
}
