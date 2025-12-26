"""
Interpolation utilities for IEC 61853 power matrix calculations.

Provides 2D bilinear interpolation for power matrices as specified
in IEC 61853-3 for energy rating calculations.

References:
    IEC 61853-1:2011 - Power matrix measurement specifications
    IEC 61853-3:2018 - Energy rating interpolation requirements

Example:
    >>> import numpy as np
    >>> from src.utils.interpolation import PowerMatrixInterpolator
    >>>
    >>> # Define power matrix (irradiance x temperature)
    >>> irradiance = np.array([200, 400, 600, 800, 1000])
    >>> temperature = np.array([15, 25, 50, 75])
    >>> power = np.array([
    ...     [60, 58, 52, 45],   # 200 W/m^2
    ...     [125, 120, 108, 94],  # 400 W/m^2
    ...     [190, 183, 165, 144], # 600 W/m^2
    ...     [255, 245, 221, 193], # 800 W/m^2
    ...     [320, 307, 277, 242]  # 1000 W/m^2
    ... ])
    >>>
    >>> interpolator = PowerMatrixInterpolator(irradiance, temperature, power)
    >>> power_at_condition = interpolator(700.0, 35.0)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from .constants import (
    MAX_CELL_TEMPERATURE,
    MAX_IRRADIANCE,
    MIN_CELL_TEMPERATURE,
    MIN_IRRADIANCE_THRESHOLD,
)


@dataclass(frozen=True)
class PowerMatrixSpec:
    """
    Specification for a power matrix.

    Attributes:
        irradiance_levels: Array of irradiance values [W/m^2]
        temperature_levels: Array of temperature values [C]
        power_values: 2D array of power values [W], shape (n_irr, n_temp)

    References:
        IEC 61853-1:2011, Section 7 - Power matrix requirements
    """

    irradiance_levels: NDArray[np.floating]
    temperature_levels: NDArray[np.floating]
    power_values: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate power matrix dimensions."""
        n_irr = len(self.irradiance_levels)
        n_temp = len(self.temperature_levels)
        expected_shape = (n_irr, n_temp)

        if self.power_values.shape != expected_shape:
            raise ValueError(
                f"Power matrix shape {self.power_values.shape} does not match "
                f"expected shape {expected_shape} based on irradiance and "
                f"temperature levels."
            )


class PowerMatrixInterpolator:
    """
    Bilinear interpolator for IEC 61853 power matrices.

    Uses scipy's RegularGridInterpolator for efficient 2D interpolation
    of power values from measured irradiance and temperature data.

    Attributes:
        irradiance: Sorted array of irradiance values [W/m^2]
        temperature: Sorted array of temperature values [C]
        power: 2D power matrix [W]
        method: Interpolation method ('linear' or 'cubic')

    References:
        IEC 61853-3:2018, Annex A - Interpolation methods

    Example:
        >>> interp = PowerMatrixInterpolator(irr, temp, power)
        >>> p = interp(800.0, 40.0)  # Power at 800 W/m^2, 40C
    """

    def __init__(
        self,
        irradiance: ArrayLike,
        temperature: ArrayLike,
        power: ArrayLike,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: Optional[float] = None,
    ) -> None:
        """
        Initialize the power matrix interpolator.

        Args:
            irradiance: 1D array of irradiance levels [W/m^2], must be sorted
            temperature: 1D array of temperature levels [C], must be sorted
            power: 2D array of power values [W], shape (n_irr, n_temp)
            method: Interpolation method, 'linear' or 'cubic'
            bounds_error: If True, raise error for out-of-bounds queries
            fill_value: Value for out-of-bounds queries (None extrapolates)

        Raises:
            ValueError: If arrays have incompatible shapes or invalid values
        """
        self._irradiance = np.asarray(irradiance, dtype=np.float64)
        self._temperature = np.asarray(temperature, dtype=np.float64)
        self._power = np.asarray(power, dtype=np.float64)
        self._method = method

        self._validate_inputs()

        # Create interpolator (irradiance, temperature) -> power
        self._interpolator = RegularGridInterpolator(
            (self._irradiance, self._temperature),
            self._power,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    def _validate_inputs(self) -> None:
        """Validate input arrays for interpolation."""
        # Check dimensions
        if self._irradiance.ndim != 1:
            raise ValueError("Irradiance must be 1D array")
        if self._temperature.ndim != 1:
            raise ValueError("Temperature must be 1D array")
        if self._power.ndim != 2:
            raise ValueError("Power must be 2D array")

        # Check shapes match
        expected_shape = (len(self._irradiance), len(self._temperature))
        if self._power.shape != expected_shape:
            raise ValueError(
                f"Power shape {self._power.shape} does not match expected "
                f"{expected_shape}"
            )

        # Check sorting
        if not np.all(np.diff(self._irradiance) > 0):
            raise ValueError("Irradiance values must be strictly increasing")
        if not np.all(np.diff(self._temperature) > 0):
            raise ValueError("Temperature values must be strictly increasing")

        # Check for valid values
        if np.any(self._irradiance < 0):
            raise ValueError("Irradiance values must be non-negative")
        if np.any(np.isnan(self._power)):
            raise ValueError("Power matrix contains NaN values")

    def __call__(
        self,
        irradiance: Union[float, ArrayLike],
        temperature: Union[float, ArrayLike],
    ) -> Union[float, NDArray[np.floating]]:
        """
        Interpolate power at given irradiance and temperature.

        Args:
            irradiance: Irradiance value(s) [W/m^2]
            temperature: Temperature value(s) [C]

        Returns:
            Interpolated power value(s) [W]

        Example:
            >>> power = interpolator(800.0, 40.0)
            >>> powers = interpolator([600, 800], [30, 40])
        """
        irr = np.atleast_1d(irradiance)
        temp = np.atleast_1d(temperature)

        # Broadcast to same shape
        irr, temp = np.broadcast_arrays(irr, temp)

        # Stack for interpolator input
        points = np.stack([irr.ravel(), temp.ravel()], axis=-1)
        result = self._interpolator(points)

        # Return scalar if inputs were scalar
        if np.isscalar(irradiance) and np.isscalar(temperature):
            return float(result[0])

        return result.reshape(irr.shape)

    @property
    def irradiance_range(self) -> tuple[float, float]:
        """Return (min, max) irradiance values [W/m^2]."""
        return float(self._irradiance[0]), float(self._irradiance[-1])

    @property
    def temperature_range(self) -> tuple[float, float]:
        """Return (min, max) temperature values [C]."""
        return float(self._temperature[0]), float(self._temperature[-1])

    @property
    def power_at_stc(self) -> float:
        """
        Return interpolated power at STC (1000 W/m^2, 25C).

        Returns:
            Power at Standard Test Conditions [W]

        References:
            IEC 61853-1:2011, Section 3.1 - STC definition
        """
        return float(self(1000.0, 25.0))


@functools.lru_cache(maxsize=128)
def bilinear_interpolate(
    x: float,
    y: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    q11: float,
    q12: float,
    q21: float,
    q22: float,
) -> float:
    """
    Perform bilinear interpolation for a single point.

    Uses the standard bilinear interpolation formula for a point (x, y)
    within the rectangle defined by corners (x1, y1) and (x2, y2).

    Args:
        x: X-coordinate of query point
        y: Y-coordinate of query point
        x1: Lower X bound
        x2: Upper X bound
        y1: Lower Y bound
        y2: Upper Y bound
        q11: Value at (x1, y1)
        q12: Value at (x1, y2)
        q21: Value at (x2, y1)
        q22: Value at (x2, y2)

    Returns:
        Interpolated value at (x, y)

    References:
        IEC 61853-3:2018, Annex A.2 - Bilinear interpolation

    Example:
        >>> value = bilinear_interpolate(
        ...     x=750, y=35,
        ...     x1=600, x2=800,
        ...     y1=25, y2=50,
        ...     q11=183, q12=165, q21=245, q22=221
        ... )
    """
    if x2 == x1 or y2 == y1:
        raise ValueError("Interpolation bounds must define non-zero intervals")

    # Compute weights
    dx = (x2 - x1)
    dy = (y2 - y1)
    wx = (x - x1) / dx
    wy = (y - y1) / dy

    # Bilinear interpolation formula
    result = (
        q11 * (1 - wx) * (1 - wy) +
        q21 * wx * (1 - wy) +
        q12 * (1 - wx) * wy +
        q22 * wx * wy
    )

    return result


def create_power_matrix(
    pmax_stc: float,
    irradiance_levels: Optional[ArrayLike] = None,
    temperature_levels: Optional[ArrayLike] = None,
    gamma_pmax: float = -0.004,
    efficiency_ratio: float = 0.95,
) -> PowerMatrixSpec:
    """
    Create a synthetic power matrix from STC power and temperature coefficient.

    Generates a power matrix following typical module behavior with
    linear irradiance scaling and temperature derating.

    Args:
        pmax_stc: Maximum power at STC [W]
        irradiance_levels: Irradiance values [W/m^2], default IEC levels
        temperature_levels: Temperature values [C], default IEC levels
        gamma_pmax: Power temperature coefficient [1/K], typically -0.003 to -0.005
        efficiency_ratio: Low-light efficiency ratio, typically 0.9 to 1.0

    Returns:
        PowerMatrixSpec with generated power values

    References:
        IEC 61853-1:2011, Section 8 - Expected power matrix behavior

    Example:
        >>> spec = create_power_matrix(pmax_stc=400.0, gamma_pmax=-0.0038)
        >>> power_at_800W_50C = spec.power_values[4, 2]
    """
    from .constants import STANDARD_IRRADIANCE_LEVELS, STANDARD_TEMPERATURE_LEVELS

    if irradiance_levels is None:
        irradiance_levels = STANDARD_IRRADIANCE_LEVELS
    if temperature_levels is None:
        temperature_levels = STANDARD_TEMPERATURE_LEVELS

    irr = np.asarray(irradiance_levels, dtype=np.float64)
    temp = np.asarray(temperature_levels, dtype=np.float64)

    # Reference conditions
    g_stc = 1000.0  # W/m^2
    t_stc = 25.0    # C

    # Create meshgrid for vectorized calculation
    G, T = np.meshgrid(irr, temp, indexing="ij")

    # Calculate power with irradiance scaling and temperature derating
    # P = Pmax * (G/G_stc) * [1 + gamma * (T - T_stc)]
    irradiance_factor = G / g_stc

    # Apply low-light efficiency correction
    low_light_correction = 1.0 - (1.0 - efficiency_ratio) * (1.0 - G / g_stc)

    # Temperature derating
    temperature_factor = 1.0 + gamma_pmax * (T - t_stc)

    power = pmax_stc * irradiance_factor * low_light_correction * temperature_factor

    # Ensure non-negative power
    power = np.maximum(power, 0.0)

    return PowerMatrixSpec(
        irradiance_levels=irr,
        temperature_levels=temp,
        power_values=power,
    )


class SplineInterpolator:
    """
    Cubic spline interpolator for smooth power matrix interpolation.

    Uses scipy's RectBivariateSpline for higher-order interpolation
    with continuous first derivatives.

    References:
        IEC 61853-3:2018, Annex A.3 - Spline interpolation (optional)
    """

    def __init__(
        self,
        irradiance: ArrayLike,
        temperature: ArrayLike,
        power: ArrayLike,
        degree: int = 3,
    ) -> None:
        """
        Initialize spline interpolator.

        Args:
            irradiance: 1D array of irradiance levels [W/m^2]
            temperature: 1D array of temperature levels [C]
            power: 2D array of power values [W]
            degree: Spline degree (1-5), default 3 (cubic)
        """
        self._irradiance = np.asarray(irradiance, dtype=np.float64)
        self._temperature = np.asarray(temperature, dtype=np.float64)
        self._power = np.asarray(power, dtype=np.float64)

        self._spline = RectBivariateSpline(
            self._irradiance,
            self._temperature,
            self._power,
            kx=degree,
            ky=degree,
        )

    def __call__(
        self,
        irradiance: Union[float, ArrayLike],
        temperature: Union[float, ArrayLike],
    ) -> Union[float, NDArray[np.floating]]:
        """
        Evaluate spline at given irradiance and temperature.

        Args:
            irradiance: Irradiance value(s) [W/m^2]
            temperature: Temperature value(s) [C]

        Returns:
            Interpolated power value(s) [W]
        """
        irr = np.atleast_1d(irradiance)
        temp = np.atleast_1d(temperature)

        # RectBivariateSpline returns 2D array for grid evaluation
        result = self._spline(irr, temp, grid=False)

        if np.isscalar(irradiance) and np.isscalar(temperature):
            return float(result)

        return result
