"""
Bilinear Interpolation Module for IEC 61853-3 Power Matrix

This module provides interpolation functions for calculating power output
from the P(G,T) power matrix as specified in IEC 61853-3.

The power matrix contains measured power values at discrete irradiance (G)
and temperature (T) points. Bilinear interpolation is used to estimate
power at any operating condition within or near the measured range.

References:
    - IEC 61853-1:2011 - Photovoltaic (PV) module performance testing
    - IEC 61853-3:2018 - Energy rating of PV modules
"""

from functools import lru_cache
from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray


class InterpolationError(Exception):
    """Exception raised for interpolation errors."""
    pass


def bilinear_interpolate(
    x: float,
    y: float,
    x_grid: NDArray[np.float64],
    y_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    extrapolate: bool = True,
    extrapolation_limit: float = 0.2
) -> float:
    """
    Perform bilinear interpolation on a 2D grid.

    Interpolates a value z at point (x, y) from a regular grid of z values.
    This is the primary interpolation method for the P(G,T) power matrix
    as specified in IEC 61853-3.

    Args:
        x: The x-coordinate (typically irradiance G in W/m^2).
        y: The y-coordinate (typically temperature T in C).
        x_grid: 1D array of x-coordinates of the grid points (ascending).
        y_grid: 1D array of y-coordinates of the grid points (ascending).
        z_grid: 2D array of z values with shape (len(y_grid), len(x_grid)).
        extrapolate: If True, allow limited extrapolation beyond grid bounds.
        extrapolation_limit: Maximum fraction beyond grid bounds for extrapolation.
            E.g., 0.2 allows 20% beyond the grid range.

    Returns:
        Interpolated z value at point (x, y).

    Raises:
        InterpolationError: If point is outside grid bounds and extrapolation
            is disabled or exceeds the limit.
        ValueError: If input arrays have invalid shapes or values.

    Example:
        >>> G_grid = np.array([200, 400, 600, 800, 1000])  # W/m^2
        >>> T_grid = np.array([15, 25, 50, 75])  # C
        >>> P_matrix = np.array([...])  # Power matrix
        >>> power = bilinear_interpolate(750, 40, G_grid, T_grid, P_matrix)

    References:
        IEC 61853-3:2018, Section 6.3 - Energy rating calculation
    """
    # Validate inputs
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    z_grid = np.asarray(z_grid, dtype=np.float64)

    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be 1D arrays")

    if z_grid.shape != (len(y_grid), len(x_grid)):
        raise ValueError(
            f"z_grid shape {z_grid.shape} must match "
            f"(len(y_grid), len(x_grid)) = ({len(y_grid)}, {len(x_grid)})"
        )

    # Check for NaN or Inf in inputs
    if np.isnan(x) or np.isnan(y):
        raise ValueError("x and y must not be NaN")

    if np.any(np.isnan(z_grid)):
        raise ValueError("z_grid contains NaN values")

    # Calculate grid bounds
    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Handle extrapolation bounds
    x_lower_bound = x_min - extrapolation_limit * x_range if extrapolate else x_min
    x_upper_bound = x_max + extrapolation_limit * x_range if extrapolate else x_max
    y_lower_bound = y_min - extrapolation_limit * y_range if extrapolate else y_min
    y_upper_bound = y_max + extrapolation_limit * y_range if extrapolate else y_max

    # Check bounds
    if x < x_lower_bound or x > x_upper_bound:
        raise InterpolationError(
            f"x={x} is outside interpolation bounds [{x_lower_bound:.2f}, {x_upper_bound:.2f}]"
        )
    if y < y_lower_bound or y > y_upper_bound:
        raise InterpolationError(
            f"y={y} is outside interpolation bounds [{y_lower_bound:.2f}, {y_upper_bound:.2f}]"
        )

    # Find bracketing indices
    i1, i2 = _find_bracket_indices(x, x_grid)
    j1, j2 = _find_bracket_indices(y, y_grid)

    # Get corner coordinates
    x1, x2 = x_grid[i1], x_grid[i2]
    y1, y2 = y_grid[j1], y_grid[j2]

    # Get corner values
    z11 = z_grid[j1, i1]  # (x1, y1)
    z12 = z_grid[j2, i1]  # (x1, y2)
    z21 = z_grid[j1, i2]  # (x2, y1)
    z22 = z_grid[j2, i2]  # (x2, y2)

    # Handle degenerate cases
    if x1 == x2 and y1 == y2:
        return float(z11)
    elif x1 == x2:
        # Linear interpolation in y only
        return float(_linear_interpolate(y, y1, y2, z11, z12))
    elif y1 == y2:
        # Linear interpolation in x only
        return float(_linear_interpolate(x, x1, x2, z11, z21))

    # Compute bilinear interpolation
    # Normalize coordinates to [0, 1]
    t = (x - x1) / (x2 - x1)
    u = (y - y1) / (y2 - y1)

    # Bilinear formula
    z = (1 - t) * (1 - u) * z11 + \
        t * (1 - u) * z21 + \
        (1 - t) * u * z12 + \
        t * u * z22

    return float(z)


def _find_bracket_indices(value: float, grid: NDArray[np.float64]) -> Tuple[int, int]:
    """
    Find the indices of grid points that bracket a value.

    Args:
        value: The value to bracket.
        grid: Sorted 1D array of grid points.

    Returns:
        Tuple of (lower_index, upper_index).
    """
    if len(grid) == 1:
        return 0, 0

    # Use searchsorted for efficiency
    idx = np.searchsorted(grid, value)

    # Handle edge cases
    if idx == 0:
        return 0, min(1, len(grid) - 1)
    elif idx >= len(grid):
        return len(grid) - 2, len(grid) - 1
    else:
        return idx - 1, idx


def _linear_interpolate(
    x: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float
) -> float:
    """
    Perform simple linear interpolation.

    Args:
        x: The x-coordinate to interpolate at.
        x1, x2: The bracketing x-coordinates.
        y1, y2: The y-values at x1 and x2.

    Returns:
        Interpolated y value at x.
    """
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def interpolate_power_matrix(
    irradiance: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    G_grid: NDArray[np.float64],
    T_grid: NDArray[np.float64],
    P_matrix: NDArray[np.float64],
    extrapolate: bool = True
) -> Union[float, NDArray[np.float64]]:
    """
    Interpolate power from a P(G,T) matrix for given irradiance and temperature.

    This function wraps bilinear_interpolate for convenient use with PV module
    power matrices. It handles both scalar and array inputs.

    Args:
        irradiance: Irradiance value(s) in W/m^2.
        temperature: Cell temperature value(s) in C.
        G_grid: Irradiance grid points (ascending order).
        T_grid: Temperature grid points (ascending order).
        P_matrix: Power matrix with shape (len(T_grid), len(G_grid)).
        extrapolate: Allow limited extrapolation beyond grid bounds.

    Returns:
        Interpolated power value(s) in Watts.

    Example:
        >>> G = np.array([200, 400, 600, 800, 1000, 1100])
        >>> T = np.array([15, 25, 50, 75])
        >>> P = np.array([[...], [...], [...], [...]])  # Power matrix
        >>> power = interpolate_power_matrix(850, 35, G, T, P)

    References:
        IEC 61853-3:2018, Section 6.3
    """
    irradiance = np.atleast_1d(irradiance)
    temperature = np.atleast_1d(temperature)

    # Broadcast to same shape if needed
    if irradiance.shape != temperature.shape:
        irradiance, temperature = np.broadcast_arrays(irradiance, temperature)

    # Allocate output
    power = np.zeros_like(irradiance, dtype=np.float64)

    # Interpolate each point
    for idx in np.ndindex(irradiance.shape):
        try:
            power[idx] = bilinear_interpolate(
                irradiance[idx],
                temperature[idx],
                G_grid,
                T_grid,
                P_matrix,
                extrapolate=extrapolate
            )
        except InterpolationError:
            # Return 0 for points outside valid range
            power[idx] = 0.0

    # Return scalar if input was scalar
    if power.size == 1:
        return float(power.flat[0])
    return power


@lru_cache(maxsize=128)
def create_interpolation_function(
    G_grid_tuple: Tuple[float, ...],
    T_grid_tuple: Tuple[float, ...],
    P_matrix_tuple: Tuple[Tuple[float, ...], ...]
):
    """
    Create a cached interpolation function for a power matrix.

    This function creates a closure that can be called efficiently
    multiple times for the same power matrix. Uses LRU caching for
    performance optimization.

    Args:
        G_grid_tuple: Tuple of irradiance grid points.
        T_grid_tuple: Tuple of temperature grid points.
        P_matrix_tuple: Tuple of tuples representing the power matrix.

    Returns:
        A function f(G, T) that returns interpolated power.

    Note:
        Arguments must be tuples (not arrays) to enable caching.
        Convert arrays using tuple(arr) and tuple(map(tuple, matrix)).
    """
    G_grid = np.array(G_grid_tuple)
    T_grid = np.array(T_grid_tuple)
    P_matrix = np.array(P_matrix_tuple)

    def interpolate(G: float, T: float) -> float:
        return bilinear_interpolate(G, T, G_grid, T_grid, P_matrix)

    return interpolate


def validate_power_matrix(
    G_grid: NDArray[np.float64],
    T_grid: NDArray[np.float64],
    P_matrix: NDArray[np.float64]
) -> Tuple[bool, Optional[str]]:
    """
    Validate a power matrix for IEC 61853 compliance.

    Checks that the power matrix meets the requirements specified in
    IEC 61853-1 for power measurements.

    Args:
        G_grid: Irradiance grid points in W/m^2.
        T_grid: Temperature grid points in C.
        P_matrix: Power matrix in Watts.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.

    References:
        IEC 61853-1:2011, Section 6 - Test matrix
    """
    # Check array dimensions
    if G_grid.ndim != 1:
        return False, "G_grid must be 1D array"
    if T_grid.ndim != 1:
        return False, "T_grid must be 1D array"
    if P_matrix.ndim != 2:
        return False, "P_matrix must be 2D array"

    # Check shape consistency
    if P_matrix.shape != (len(T_grid), len(G_grid)):
        return False, (
            f"P_matrix shape {P_matrix.shape} must match "
            f"(len(T_grid), len(G_grid)) = ({len(T_grid)}, {len(G_grid)})"
        )

    # Check for ascending order
    if not np.all(np.diff(G_grid) > 0):
        return False, "G_grid must be in ascending order"
    if not np.all(np.diff(T_grid) > 0):
        return False, "T_grid must be in ascending order"

    # Check for NaN values
    if np.any(np.isnan(P_matrix)):
        return False, "P_matrix contains NaN values"

    # Check for negative power values
    if np.any(P_matrix < 0):
        return False, "P_matrix contains negative power values"

    # IEC 61853-1 recommended grid points
    recommended_G = [100, 200, 400, 600, 800, 1000, 1100]
    recommended_T = [15, 25, 50, 75]

    # Check minimum coverage
    if len(G_grid) < 4:
        return False, "G_grid should have at least 4 points for accurate interpolation"
    if len(T_grid) < 3:
        return False, "T_grid should have at least 3 points for accurate interpolation"

    # Check irradiance range
    if G_grid[0] > 200:
        return False, f"G_grid should start at 200 W/m^2 or lower, got {G_grid[0]}"
    if G_grid[-1] < 1000:
        return False, f"G_grid should extend to at least 1000 W/m^2, got {G_grid[-1]}"

    # Check temperature range
    if T_grid[0] > 25:
        return False, f"T_grid should start at 25 C or lower, got {T_grid[0]}"
    if T_grid[-1] < 50:
        return False, f"T_grid should extend to at least 50 C, got {T_grid[-1]}"

    return True, None


def extrapolate_power(
    G: float,
    T: float,
    G_grid: NDArray[np.float64],
    T_grid: NDArray[np.float64],
    P_matrix: NDArray[np.float64],
    P_stc: float,
    gamma: float = -0.004
) -> float:
    """
    Extrapolate power beyond the measured matrix using temperature coefficients.

    For conditions outside the power matrix range, this function uses
    the temperature coefficient to extrapolate power values.

    Args:
        G: Irradiance in W/m^2.
        T: Cell temperature in C.
        G_grid: Irradiance grid points.
        T_grid: Temperature grid points.
        P_matrix: Power matrix.
        P_stc: Power at STC (1000 W/m^2, 25 C) in Watts.
        gamma: Temperature coefficient of Pmax (/C). Default -0.4%/C.

    Returns:
        Estimated power in Watts.

    Note:
        This is a simplified extrapolation. For accurate results outside
        the matrix range, additional measurements are recommended.
    """
    # Try direct interpolation first
    try:
        return bilinear_interpolate(G, T, G_grid, T_grid, P_matrix, extrapolate=False)
    except InterpolationError:
        pass

    # Extrapolate using temperature coefficient
    # P(G, T) = P_stc * (G / 1000) * (1 + gamma * (T - 25))
    P_extrapolated = P_stc * (G / 1000) * (1 + gamma * (T - 25))

    # Ensure non-negative power
    return max(0.0, P_extrapolated)
