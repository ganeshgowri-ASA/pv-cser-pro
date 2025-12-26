"""
Power matrix data handling for IEC 61853-1.

Provides functionality for loading, validating, and processing
power matrix data (irradiance vs temperature performance).
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate


@dataclass
class PowerMatrixData:
    """Container for power matrix data."""

    irradiance_levels: np.ndarray  # W/m²
    temperature_levels: np.ndarray  # °C
    power_matrix: np.ndarray  # W (2D: irradiance x temperature)

    # Optional current and voltage matrices
    current_matrix: Optional[np.ndarray] = None
    voltage_matrix: Optional[np.ndarray] = None

    # Metadata
    measurement_date: Optional[str] = None
    measurement_location: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate matrix dimensions."""
        expected_shape = (len(self.irradiance_levels), len(self.temperature_levels))
        if self.power_matrix.shape != expected_shape:
            raise ValueError(
                f"Power matrix shape {self.power_matrix.shape} does not match "
                f"expected shape {expected_shape}"
            )

    def get_power_at_stc(self) -> float:
        """Get power at STC conditions (1000 W/m², 25°C)."""
        return self.interpolate_power(1000.0, 25.0)

    def interpolate_power(
        self,
        irradiance: Union[float, np.ndarray],
        temperature: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Interpolate power at given irradiance and temperature.

        Args:
            irradiance: Irradiance value(s) in W/m²
            temperature: Temperature value(s) in °C

        Returns:
            Interpolated power value(s)
        """
        interp_func = interpolate.RegularGridInterpolator(
            (self.irradiance_levels, self.temperature_levels),
            self.power_matrix,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Handle scalar inputs
        scalar_input = np.isscalar(irradiance) and np.isscalar(temperature)

        irradiance = np.atleast_1d(irradiance)
        temperature = np.atleast_1d(temperature)

        # Create coordinate pairs
        if len(irradiance) == len(temperature):
            points = np.column_stack([irradiance, temperature])
        else:
            # Create mesh for different-length arrays
            g, t = np.meshgrid(irradiance, temperature, indexing='ij')
            points = np.column_stack([g.ravel(), t.ravel()])

        result = interp_func(points)

        if scalar_input:
            return float(result[0])

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert power matrix to DataFrame."""
        return pd.DataFrame(
            self.power_matrix,
            index=self.irradiance_levels,
            columns=self.temperature_levels,
        )


class PowerMatrixHandler:
    """
    Handler for power matrix data operations.

    Provides methods for loading, validating, and processing
    power matrix data from various file formats.
    """

    # Standard IEC 61853-1 measurement conditions
    STANDARD_IRRADIANCE = np.array([100, 200, 400, 600, 800, 1000, 1100])  # W/m²
    STANDARD_TEMPERATURES = np.array([15, 25, 50, 75])  # °C

    def __init__(self):
        """Initialize handler."""
        self._loaded_matrix: Optional[PowerMatrixData] = None

    def load_from_csv(
        self,
        file_content: Union[bytes, str, BytesIO],
    ) -> PowerMatrixData:
        """
        Load power matrix from CSV file.

        Expected format:
        - First row: temperature values (with first cell as header)
        - First column: irradiance values
        - Remaining cells: power values

        Args:
            file_content: CSV file content

        Returns:
            PowerMatrixData instance
        """
        if isinstance(file_content, bytes):
            file_content = BytesIO(file_content)
        elif isinstance(file_content, str):
            file_content = BytesIO(file_content.encode())

        df = pd.read_csv(file_content, index_col=0)

        irradiance_levels = df.index.values.astype(float)
        temperature_levels = df.columns.values.astype(float)
        power_matrix = df.values.astype(float)

        self._loaded_matrix = PowerMatrixData(
            irradiance_levels=irradiance_levels,
            temperature_levels=temperature_levels,
            power_matrix=power_matrix,
        )

        return self._loaded_matrix

    def load_from_excel(
        self,
        file_content: Union[bytes, BytesIO],
        sheet_name: Union[str, int] = 0,
    ) -> PowerMatrixData:
        """
        Load power matrix from Excel file.

        Args:
            file_content: Excel file content
            sheet_name: Sheet name or index

        Returns:
            PowerMatrixData instance
        """
        if isinstance(file_content, bytes):
            file_content = BytesIO(file_content)

        df = pd.read_excel(file_content, sheet_name=sheet_name, index_col=0)

        irradiance_levels = df.index.values.astype(float)
        temperature_levels = df.columns.values.astype(float)
        power_matrix = df.values.astype(float)

        self._loaded_matrix = PowerMatrixData(
            irradiance_levels=irradiance_levels,
            temperature_levels=temperature_levels,
            power_matrix=power_matrix,
        )

        return self._loaded_matrix

    def validate_matrix(
        self,
        matrix_data: PowerMatrixData,
        pmax_stc: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate power matrix data.

        Args:
            matrix_data: PowerMatrixData to validate
            pmax_stc: Expected Pmax at STC for validation

        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        issues = []

        # Check for negative power values
        if np.any(matrix_data.power_matrix < 0):
            issues.append("ERROR: Power matrix contains negative values")

        # Check for NaN values
        nan_count = np.isnan(matrix_data.power_matrix).sum()
        if nan_count > 0:
            issues.append(f"WARNING: Power matrix contains {nan_count} NaN values")

        # Check irradiance range
        if matrix_data.irradiance_levels.min() > 100:
            issues.append("WARNING: Minimum irradiance > 100 W/m²")
        if matrix_data.irradiance_levels.max() < 1000:
            issues.append("WARNING: Maximum irradiance < 1000 W/m²")

        # Check temperature range
        if matrix_data.temperature_levels.min() > 25:
            issues.append("WARNING: Minimum temperature > 25°C")
        if matrix_data.temperature_levels.max() < 50:
            issues.append("WARNING: Maximum temperature < 50°C")

        # Check STC condition availability
        has_1000 = 1000 in matrix_data.irradiance_levels
        has_25 = 25 in matrix_data.temperature_levels
        if not has_1000:
            issues.append("WARNING: 1000 W/m² irradiance level not present")
        if not has_25:
            issues.append("WARNING: 25°C temperature level not present")

        # Validate against expected Pmax at STC
        if pmax_stc is not None and has_1000 and has_25:
            idx_g = np.where(matrix_data.irradiance_levels == 1000)[0][0]
            idx_t = np.where(matrix_data.temperature_levels == 25)[0][0]
            measured_pmax = matrix_data.power_matrix[idx_g, idx_t]

            deviation = abs(measured_pmax - pmax_stc) / pmax_stc * 100
            if deviation > 5:
                issues.append(
                    f"WARNING: Measured Pmax at STC ({measured_pmax:.1f}W) differs "
                    f"from specified Pmax ({pmax_stc:.1f}W) by {deviation:.1f}%"
                )

        # Check monotonicity with irradiance
        for j in range(len(matrix_data.temperature_levels)):
            power_col = matrix_data.power_matrix[:, j]
            if not np.all(np.diff(power_col[~np.isnan(power_col)]) >= -1):
                issues.append(
                    f"WARNING: Power not monotonic with irradiance at T="
                    f"{matrix_data.temperature_levels[j]}°C"
                )

        # Check temperature coefficient behavior (power should decrease with temp)
        for i in range(len(matrix_data.irradiance_levels)):
            power_row = matrix_data.power_matrix[i, :]
            valid = ~np.isnan(power_row)
            if np.sum(valid) >= 2:
                if np.any(np.diff(power_row[valid]) > 0):
                    issues.append(
                        f"WARNING: Power increases with temperature at G="
                        f"{matrix_data.irradiance_levels[i]} W/m²"
                    )

        is_valid = not any("ERROR" in issue for issue in issues)
        return is_valid, issues

    def normalize_matrix(
        self,
        matrix_data: PowerMatrixData,
    ) -> PowerMatrixData:
        """
        Normalize power matrix to rated power (Pmax at STC = 1.0).

        Args:
            matrix_data: PowerMatrixData to normalize

        Returns:
            Normalized PowerMatrixData
        """
        pmax_stc = matrix_data.get_power_at_stc()
        if pmax_stc <= 0 or np.isnan(pmax_stc):
            raise ValueError("Cannot normalize: invalid Pmax at STC")

        normalized_matrix = matrix_data.power_matrix / pmax_stc

        return PowerMatrixData(
            irradiance_levels=matrix_data.irradiance_levels.copy(),
            temperature_levels=matrix_data.temperature_levels.copy(),
            power_matrix=normalized_matrix,
            notes=f"Normalized to Pmax={pmax_stc:.1f}W",
        )

    def calculate_temperature_coefficients(
        self,
        matrix_data: PowerMatrixData,
        irradiance: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Calculate temperature coefficient from power matrix.

        Args:
            matrix_data: PowerMatrixData
            irradiance: Irradiance level for calculation

        Returns:
            Dictionary with temperature coefficients
        """
        # Find closest irradiance level
        idx = np.argmin(np.abs(matrix_data.irradiance_levels - irradiance))
        power_vs_temp = matrix_data.power_matrix[idx, :]
        temps = matrix_data.temperature_levels

        # Remove NaN values
        valid = ~np.isnan(power_vs_temp)
        power_valid = power_vs_temp[valid]
        temps_valid = temps[valid]

        if len(power_valid) < 2:
            return {"gamma_pmax": np.nan, "gamma_pmax_rel": np.nan}

        # Linear fit
        coeffs = np.polyfit(temps_valid, power_valid, 1)
        gamma_pmax = coeffs[0]  # W/°C

        # Get reference power at 25°C
        p_25 = matrix_data.interpolate_power(irradiance, 25.0)
        gamma_pmax_rel = (gamma_pmax / p_25) * 100 if p_25 > 0 else np.nan  # %/°C

        return {
            "gamma_pmax": gamma_pmax,
            "gamma_pmax_rel": gamma_pmax_rel,
        }

    @staticmethod
    def generate_sample_matrix(
        pmax_stc: float = 400.0,
        temp_coeff: float = -0.35,
    ) -> PowerMatrixData:
        """
        Generate a sample power matrix for demonstration.

        Args:
            pmax_stc: Maximum power at STC (W)
            temp_coeff: Power temperature coefficient (%/°C)

        Returns:
            PowerMatrixData with generated values
        """
        irradiance_levels = np.array([100, 200, 400, 600, 800, 1000, 1100])
        temperature_levels = np.array([15, 25, 50, 75])

        power_matrix = np.zeros((len(irradiance_levels), len(temperature_levels)))

        for i, g in enumerate(irradiance_levels):
            for j, t in enumerate(temperature_levels):
                # Base power scales linearly with irradiance
                p_base = pmax_stc * (g / 1000.0)

                # Apply temperature correction
                temp_factor = 1 + (temp_coeff / 100) * (t - 25)
                power_matrix[i, j] = p_base * temp_factor

                # Add slight non-linearity at low irradiance
                if g < 400:
                    power_matrix[i, j] *= 0.98

        return PowerMatrixData(
            irradiance_levels=irradiance_levels,
            temperature_levels=temperature_levels,
            power_matrix=power_matrix,
            notes="Sample power matrix for demonstration",
        )

    def export_to_csv(
        self,
        matrix_data: PowerMatrixData,
        filepath: str,
    ) -> None:
        """Export power matrix to CSV file."""
        df = matrix_data.to_dataframe()
        df.index.name = "Irradiance (W/m²)"
        df.columns.name = "Temperature (°C)"
        df.to_csv(filepath)

    def export_to_excel(
        self,
        matrix_data: PowerMatrixData,
        filepath: str,
    ) -> None:
        """Export power matrix to Excel file."""
        df = matrix_data.to_dataframe()
        df.index.name = "Irradiance (W/m²)"
        df.columns.name = "Temperature (°C)"
        df.to_excel(filepath, sheet_name="Power Matrix")
