"""
Helper functions for PV-CSER Pro application.

Provides utility functions for data validation, formatting,
interpolation, and statistical calculations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    min_rows: int = 1,
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame against specified requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of columns that must be numeric
        min_rows: Minimum number of rows required

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if df is None or df.empty:
        return False, ["DataFrame is empty or None"]

    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")

    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' must be numeric")
                elif df[col].isna().all():
                    errors.append(f"Column '{col}' contains only missing values")

    return len(errors) == 0, errors


def format_number(
    value: float,
    precision: int = 2,
    suffix: str = "",
    use_si_prefix: bool = False,
) -> str:
    """
    Format a number for display.

    Args:
        value: Number to format
        precision: Decimal precision
        suffix: Optional suffix (e.g., " W", " °C")
        use_si_prefix: Whether to use SI prefixes (k, M, G)

    Returns:
        Formatted string
    """
    if value is None or np.isnan(value):
        return "N/A"

    if use_si_prefix:
        if abs(value) >= 1e9:
            return f"{value/1e9:.{precision}f} G{suffix}"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{precision}f} M{suffix}"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{precision}f} k{suffix}"

    return f"{value:.{precision}f}{suffix}"


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers or arrays, handling division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value when division by zero

    Returns:
        Result of division or default value
    """
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                denominator != 0,
                numerator / denominator,
                default,
            )
            return np.nan_to_num(result, nan=default)
    else:
        if denominator == 0:
            return default
        return numerator / denominator


def interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xi: Union[float, np.ndarray],
    yi: Union[float, np.ndarray],
    method: str = "linear",
) -> Union[float, np.ndarray]:
    """
    Perform 2D interpolation on scattered data.

    Args:
        x: X coordinates of known points
        y: Y coordinates of known points
        z: Values at known points
        xi: X coordinate(s) for interpolation
        yi: Y coordinate(s) for interpolation
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Interpolated value(s)
    """
    # Create interpolation function
    if method == "cubic":
        try:
            interp_func = interpolate.CloughTocher2DInterpolator(
                list(zip(x.flatten(), y.flatten())),
                z.flatten(),
            )
        except Exception:
            # Fall back to linear if cubic fails
            interp_func = interpolate.LinearNDInterpolator(
                list(zip(x.flatten(), y.flatten())),
                z.flatten(),
            )
    elif method == "nearest":
        interp_func = interpolate.NearestNDInterpolator(
            list(zip(x.flatten(), y.flatten())),
            z.flatten(),
        )
    else:  # linear
        interp_func = interpolate.LinearNDInterpolator(
            list(zip(x.flatten(), y.flatten())),
            z.flatten(),
        )

    return interp_func(xi, yi)


def interpolate_power_matrix(
    irradiance_levels: np.ndarray,
    temperature_levels: np.ndarray,
    power_matrix: np.ndarray,
    target_irradiance: float,
    target_temperature: float,
) -> float:
    """
    Interpolate power from a power matrix.

    Args:
        irradiance_levels: Array of irradiance values (W/m²)
        temperature_levels: Array of temperature values (°C)
        power_matrix: 2D array of power values [irradiance x temperature]
        target_irradiance: Target irradiance for interpolation
        target_temperature: Target temperature for interpolation

    Returns:
        Interpolated power value
    """
    # Create regular grid interpolator
    interp_func = interpolate.RegularGridInterpolator(
        (irradiance_levels, temperature_levels),
        power_matrix,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Clip to valid range
    target_irradiance = np.clip(
        target_irradiance,
        irradiance_levels.min(),
        irradiance_levels.max(),
    )
    target_temperature = np.clip(
        target_temperature,
        temperature_levels.min(),
        temperature_levels.max(),
    )

    return float(interp_func((target_irradiance, target_temperature)))


def calculate_statistics(data: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a dataset.

    Args:
        data: Input data array

    Returns:
        Dictionary of statistical measures
    """
    arr = np.array(data)
    arr = arr[~np.isnan(arr)]  # Remove NaN values

    if len(arr) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "sum": np.nan,
        }

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "sum": float(np.sum(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def celsius_to_kelvin(celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


def parse_power_matrix_csv(file_content: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a power matrix from CSV content.

    Expected format:
    - First row: temperature values (header row, first cell empty or "Irradiance")
    - First column: irradiance values
    - Remaining cells: power values

    Args:
        file_content: CSV file content as bytes

    Returns:
        Tuple of (irradiance_levels, temperature_levels, power_matrix)
    """
    from io import BytesIO

    df = pd.read_csv(BytesIO(file_content), index_col=0)

    irradiance_levels = df.index.values.astype(float)
    temperature_levels = df.columns.values.astype(float)
    power_matrix = df.values.astype(float)

    return irradiance_levels, temperature_levels, power_matrix


def parse_power_matrix_excel(
    file_content: bytes,
    sheet_name: Union[str, int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a power matrix from Excel content.

    Args:
        file_content: Excel file content as bytes
        sheet_name: Sheet name or index to read

    Returns:
        Tuple of (irradiance_levels, temperature_levels, power_matrix)
    """
    from io import BytesIO

    df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name, index_col=0)

    irradiance_levels = df.index.values.astype(float)
    temperature_levels = df.columns.values.astype(float)
    power_matrix = df.values.astype(float)

    return irradiance_levels, temperature_levels, power_matrix


def generate_sample_power_matrix(
    pmax_stc: float = 400.0,
    temp_coeff: float = -0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sample power matrix for demonstration.

    Args:
        pmax_stc: Maximum power at STC (W)
        temp_coeff: Power temperature coefficient (%/°C)

    Returns:
        Tuple of (irradiance_levels, temperature_levels, power_matrix)
    """
    irradiance_levels = np.array([100, 200, 400, 600, 800, 1000, 1100])
    temperature_levels = np.array([15, 25, 50, 75])

    # Generate power matrix
    power_matrix = np.zeros((len(irradiance_levels), len(temperature_levels)))

    for i, g in enumerate(irradiance_levels):
        for j, t in enumerate(temperature_levels):
            # Base power scales linearly with irradiance
            p_base = pmax_stc * (g / 1000.0)
            # Apply temperature correction
            temp_factor = 1 + (temp_coeff / 100) * (t - 25)
            power_matrix[i, j] = p_base * temp_factor

    return irradiance_levels, temperature_levels, power_matrix


def validate_module_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate module specification data.

    Args:
        data: Dictionary of module data

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    required_fields = ["manufacturer", "model_name", "pmax_stc"]
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")

    # Validate numeric ranges
    numeric_validations = [
        ("pmax_stc", 0, 1000, "Pmax at STC"),
        ("voc_stc", 0, 100, "Voc at STC"),
        ("isc_stc", 0, 30, "Isc at STC"),
        ("vmp_stc", 0, 100, "Vmp at STC"),
        ("imp_stc", 0, 30, "Imp at STC"),
        ("temp_coeff_pmax", -1, 0, "Temperature coefficient (Pmax)"),
        ("module_area", 0.1, 5, "Module area"),
        ("nmot", 30, 60, "NMOT"),
    ]

    for field, min_val, max_val, name in numeric_validations:
        if field in data and data[field] is not None:
            val = data[field]
            if not isinstance(val, (int, float)):
                errors.append(f"{name} must be a number")
            elif val < min_val or val > max_val:
                errors.append(f"{name} should be between {min_val} and {max_val}")

    return len(errors) == 0, errors
